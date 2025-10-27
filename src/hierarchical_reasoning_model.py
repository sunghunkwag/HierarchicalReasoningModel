# Hierarchical Reasoning Model with Adaptive Computation Time and Meta-Reinforcement Learning
# A novel architecture for dynamic multi-level reasoning with batch-wise ACT processing

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class HRMConfig:
    """Configuration class for Hierarchical Reasoning Model"""
    hidden_low: int
    hidden_high: int
    T: int                      # Low-level internal step limit
    max_cycles: int             # High-level cycle limit
    use_act: bool = True        # Whether to use Adaptive Computation Time
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    detach_low_to_high: bool = True   # Gradient blocking for low->high transfer
    detach_high_to_low: bool = False  # Gradient blocking for high->low feedback
    compute_cost_weight: float = 0.0  # Computation penalty weight
    # Additional hyperparameters can be added as needed

# ---------- Component Modules ----------
class InputNetwork(nn.Module):
    """Input encoding network"""
    def __init__(self, cfg: HRMConfig):
        super().__init__()
        self.enc = nn.Linear(1024, cfg.hidden_low)  # Example: 1024-dim input
    
    def forward(self, x):                           # x: [B, D]
        return self.enc(x)

class LowLevelModule(nn.Module):
    """Low-level processing module with top-down connections"""
    def __init__(self, cfg: HRMConfig):
        super().__init__()
        self.gru = nn.GRUCell(cfg.hidden_low, cfg.hidden_low)
        self.cross = nn.Linear(cfg.hidden_high, cfg.hidden_low)  # top-down signal
    
    def forward(self, z_L, z_H, x_tilde):
        # Combine top-down signal
        h = z_L + self.cross(z_H)
        h = self.gru(x_tilde, h)
        return h

class HighLevelModule(nn.Module):
    """High-level processing module"""
    def __init__(self, cfg: HRMConfig):
        super().__init__()
        self.gru = nn.GRUCell(cfg.hidden_low, cfg.hidden_high)
    
    def forward(self, z_H, z_L_summary):
        return self.gru(z_L_summary, z_H)

class OutputNetwork(nn.Module):
    """Final output network"""
    def __init__(self, cfg: HRMConfig):
        super().__init__()
        self.head = nn.Linear(cfg.hidden_high, 1)  # Replace with regression/logits as needed
    
    def forward(self, z_H):
        return self.head(z_H)

class QNetwork(nn.Module):
    """ACT policy: 'continue' vs 'stop' scoring"""
    def __init__(self, cfg: HRMConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.hidden_high, 2)
    
    def forward(self, z_H):
        return self.fc(z_H)  # [B, 2]

# Meta-RL controller and experience buffer with lightweight stubs
class MetaRLController(nn.Module):
    """Meta-reinforcement learning controller"""
    def __init__(self, cfg: HRMConfig):
        super().__init__()
        self.baseline = nn.Linear(cfg.hidden_high, 1)
    
    def compute_reward(self, loss_vec: torch.Tensor, cycles: torch.Tensor, cfg: HRMConfig) -> torch.Tensor:
        """
        loss_vec: [B], cycles: [B] -> reward: [B]
        Accuracy improvement - computation penalty
        """
        return (-loss_vec.detach()) - cfg.compute_cost_weight * cycles.float()
    
    def policy_loss(self, logits, action, advantage):
        # logits: [B,2], action: [B], advantage: [B]
        logp = F.log_softmax(logits, dim=-1).gather(1, action.view(-1,1)).squeeze(1)
        return -(logp * advantage.detach()).mean()

class ExperienceBuffer:
    """Simple experience buffer for meta-learning"""
    def __init__(self):
        self.items = []
    
    def add(self, **kw):
        self.items.append(kw)
    
    def clear(self):
        self.items.clear()
    
    def sample_all(self):
        return self.items

class MetaOptimizer:
    """Wrapper for multiple optimizers"""
    def __init__(self, optimizers: Dict[str, torch.optim.Optimizer]):
        self.opt = optimizers
    
    def step_all(self):
        for o in self.opt.values():
            o.step()
    
    def zero_all(self):
        for o in self.opt.values():
            o.zero_grad(set_to_none=True)

# ---------- Main Model ----------
class HierarchicalReasoningModel(nn.Module):
    """
    Hierarchical Reasoning Model with Adaptive Computation Time and Meta-Reinforcement Learning
    
    This model implements:
    - Two-level hierarchical processing (low-level and high-level)
    - Adaptive Computation Time (ACT) for dynamic reasoning cycles
    - Meta-reinforcement learning for learning optimal computation allocation
    - Batch-wise ACT processing for efficient training
    """
    
    def __init__(self, config: HRMConfig):
        super().__init__()
        self.cfg = config
        self.input_network = InputNetwork(config)
        self.low_level_module = LowLevelModule(config)
        self.high_level_module = HighLevelModule(config)
        self.output_network = OutputNetwork(config)
        self.q_network = QNetwork(config)
        self.meta_controller = MetaRLController(config)
        self.experience_buffer = ExperienceBuffer()

        # Parameter group optimizers (can be separated if needed)
        self.meta_optimizer = MetaOptimizer({
            "core": torch.optim.Adam(self.parameters())
        })
        self.to(config.device)

    # ---------- State Initialization ----------
    def initialize_states(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden states for both levels"""
        z_L = torch.zeros(batch_size, self.cfg.hidden_low, device=self.cfg.device)
        z_H = torch.zeros(batch_size, self.cfg.hidden_high, device=self.cfg.device)
        return z_L, z_H

    # ---------- ACT: Cycle Count Decision ----------
    @torch.no_grad()
    def adaptive_computation_time(self, z_H, cycle, hard_limit: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch-wise 'continue/stop' decision using simple greedy policy"""
        logits = self.q_network(z_H)                    # [B, 2]
        action = logits.argmax(dim=-1)                  # 0: continue, 1: stop
        if hard_limit is None:
            hard_limit = self.cfg.max_cycles
        # Force stop: if last cycle
        if cycle + 1 >= hard_limit:
            action = torch.ones_like(action)
        return action, logits

    # ---------- Per-sample Loss Helper ----------
    def _per_sample_loss(self, pred, y, criterion, criterion_noreduce=None):
        """
        Get loss vector [B] by attempting 'reduction="none"' and averaging over non-batch dimensions.
        """
        if criterion_noreduce is not None:
            losses = criterion_noreduce(pred, y)  # Expected shape: [B,...]
        else:
            # Attempt 1: Functional call with reduction specification
            try:
                losses = criterion(pred, y, reduction="none")
            except TypeError:
                # Attempt 2: Module-style criterion with temporary reduction change
                if hasattr(criterion, "reduction"):
                    old = criterion.reduction
                    try:
                        criterion.reduction = "none"
                        losses = criterion(pred, y)
                    finally:
                        criterion.reduction = old
                else:
                    # Attempt 3: Heuristic (classification/regression)
                    if pred.dim() > 1 and y.dtype in (torch.long, torch.int64):
                        losses = F.cross_entropy(pred, y, reduction="none")
                    else:
                        losses = F.mse_loss(pred, y, reduction="none")

        # Average over non-batch dimensions
        while losses.dim() > 1:
            losses = losses.mean(dim=-1)
        return losses  # [B]

    # ---------- Forward Pass (Inference) ----------
    def forward(self, x, num_cycles: Optional[int] = None):
        """
        Forward pass for inference
        
        Args:
            x: Input tensor [B, D]
            num_cycles: Fixed number of cycles (if None, uses ACT)
        
        Returns:
            output: Model output
            cycles_taken: Number of cycles used per sample
        """
        B = x.size(0)
        x_tilde = self.input_network(x)                 # [B, hidden_low]
        z_L, z_H = self.initialize_states(B)

        cycles_taken = torch.zeros(B, dtype=torch.long, device=self.cfg.device)
        cycle = 0
        
        while True:
            # T steps of low-level processing
            for _ in range(self.cfg.T):
                top = z_H.detach() if self.cfg.detach_high_to_low else z_H
                z_L = self.low_level_module(z_L, top, x_tilde)

            # Low->high aggregation (using simple pass-through here)
            z_L_summary = z_L.detach() if self.cfg.detach_low_to_high else z_L
            z_H = self.high_level_module(z_H, z_L_summary)

            cycles_taken += 1
            cycle += 1

            # Termination decision
            if num_cycles is not None:
                if cycle >= num_cycles:
                    break
            elif self.cfg.use_act:
                action, _ = self.adaptive_computation_time(z_H, cycle-1)
                # If all samples want to "stop", exit
                if (action == 1).all():
                    break
            else:
                if cycle >= self.cfg.max_cycles:
                    break

        output = self.output_network(z_H)               # [B, ...]
        return output, cycles_taken

    # ---------- Training Step with Batch-wise ACT ----------
    def training_step(self, x, y, criterion, criterion_noreduce=None):
        """
        Training step with batch-wise ACT processing and per-sample advantage calculation
        
        Args:
            x: Input batch [B, D]
            y: Target batch [B, ...]
            criterion: Loss function
            criterion_noreduce: Alternative loss function with reduction="none"
        
        Returns:
            Dictionary with loss components and metrics
        """
        cfg = self.cfg
        dev = cfg.device
        self.meta_optimizer.zero_all()

        B = x.size(0)
        x_tilde = self.input_network(x)                    # [B, hidden_low]
        z_L, z_H = self.initialize_states(B)               # [B,Hl], [B,Hh]
        cycles_taken = torch.zeros(B, dtype=torch.long, device=dev)

        # Per-sample termination management
        alive = torch.ones(B, dtype=torch.bool, device=dev)

        # Storage for "stop moment" snapshots of each sample
        final_logits = torch.zeros(B, 2, device=dev)
        final_actions = torch.full((B,), -1, dtype=torch.long, device=dev)
        final_z_H = torch.zeros(B, cfg.hidden_high, device=dev)

        cycle = 0
        # High-level cycle loop: until all stop or limit reached
        while alive.any() and cycle < cfg.max_cycles:
            # T steps of low-level processing (stopped samples keep fixed state)
            for _ in range(cfg.T):
                top = z_H.detach() if cfg.detach_high_to_low else z_H
                new_z_L = self.low_level_module(z_L, top, x_tilde)
                z_L = torch.where(alive.unsqueeze(1), new_z_L, z_L)

            # High-level update (stopped samples keep fixed state)
            z_L_summary = z_L.detach() if cfg.detach_low_to_high else z_L
            new_z_H = self.high_level_module(z_H, z_L_summary)
            z_H = torch.where(alive.unsqueeze(1), new_z_H, z_H)

            # Computation count increases only for alive samples
            cycles_taken = cycles_taken + alive.long()
            cycle += 1

            # ACT: per-sample stop decision
            if cfg.use_act:
                with torch.no_grad():
                    logits = self.q_network(z_H.detach())          # [B,2]
                    action = logits.argmax(dim=-1)                 # 0: continue, 1: stop

                    # Force stop remaining samples if at max cycles
                    if cycle >= cfg.max_cycles:
                        action = torch.where(alive, torch.ones_like(action), action)

                    just_stopped = alive & (action == 1)
                    if just_stopped.any():
                        final_logits[just_stopped] = logits[just_stopped]
                        final_actions[just_stopped] = action[just_stopped]
                        final_z_H[just_stopped] = z_H[just_stopped]
                        alive = alive & (~just_stopped)
            else:
                # ACT disabled: stop all at limit
                if cycle >= cfg.max_cycles:
                    with torch.no_grad():
                        logits = self.q_network(z_H.detach())
                    final_logits[alive] = logits[alive]
                    final_actions[alive] = 1
                    final_z_H[alive] = z_H[alive]
                    alive = torch.zeros_like(alive)

        # Safety fill for any remaining samples (should be rare)
        remaining = final_actions < 0
        if remaining.any():
            with torch.no_grad():
                logits = self.q_network(z_H.detach())
            final_logits[remaining] = logits[remaining]
            final_actions[remaining] = 1
            final_z_H[remaining] = z_H[remaining]

        # Final output and per-sample loss
        pred = self.output_network(final_z_H)              # [B,...]
        task_loss_vec = self._per_sample_loss(pred, y, criterion, criterion_noreduce)  # [B]
        task_loss = task_loss_vec.mean()

        # Reward/advantage calculation
        reward = self.meta_controller.compute_reward(task_loss_vec, cycles_taken, cfg)  # [B]
        baseline_pred = self.meta_controller.baseline(final_z_H).squeeze(1)             # [B]

        advantage = reward - baseline_pred.detach()  # [B]
        # Normalize advantage for policy stability
        adv_norm = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Policy loss (applied only to final decisions of each sample)
        policy_loss = self.meta_controller.policy_loss(final_logits, final_actions, adv_norm) if cfg.use_act else torch.tensor(0.0, device=dev)

        # Value function loss using Huber loss
        baseline_loss = F.smooth_l1_loss(baseline_pred, reward.detach())

        # Total loss and backpropagation
        loss = task_loss + policy_loss + baseline_loss
        loss.backward()
        self.meta_optimizer.step_all()

        return {
            "loss": float(loss.detach().cpu()),
            "task_loss": float(task_loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()) if cfg.use_act else 0.0,
            "baseline_loss": float(baseline_loss.detach().cpu()),
            "cycles_mean": float(cycles_taken.float().mean().cpu())
        }

    # ---------- Meta Feedback Collection ----------
    def collect_meta_feedback(self, *, cycle, logits, action, loss_value, cycles_tensor):
        """Simple logging for meta-learning training"""
        self.experience_buffer.add(
            cycle=cycle,
            logits=logits.detach(),
            action=action.detach(),
            loss_value=loss_value.detach(),
            cycles=cycles_tensor.detach()
        )
