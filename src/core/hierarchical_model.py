#!/usr/bin/env python3
"""
Advanced Hierarchical Reasoning Model Implementation

This module provides a comprehensive implementation of the Hierarchical Reasoning Model
with Adaptive Computation Time, Meta-Reinforcement Learning, and advanced optimizations.

Key Features:
- Hierarchical two-level processing with configurable depth
- Adaptive Computation Time with learned policies
- Meta-reinforcement learning for computation allocation
- Batch-wise processing with individual sample termination
- Advanced gradient management and optimization
- Comprehensive logging and monitoring
- Memory-efficient implementation with gradient checkpointing
- Support for multiple loss functions and regularization
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.checkpoint import checkpoint
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HierarchicalModelConfig:
    """
    Comprehensive configuration class for the Hierarchical Reasoning Model.
    
    This configuration supports various architectural choices, training strategies,
    and optimization techniques for maximum flexibility and performance.
    """
    # Core Architecture Parameters
    input_dim: int = 1024
    hidden_low: int = 256
    hidden_high: int = 512
    output_dim: int = 1
    
    # Hierarchical Processing
    T: int = 3                          # Low-level steps per high-level cycle
    max_cycles: int = 8                 # Maximum reasoning cycles
    min_cycles: int = 1                 # Minimum required cycles
    
    # Adaptive Computation Time
    use_act: bool = True                # Enable ACT
    act_threshold: float = 0.99         # ACT halting threshold
    act_penalty: float = 0.01           # Computation cost penalty
    act_epsilon: float = 0.01           # Numerical stability for ACT
    
    # Meta-Reinforcement Learning
    use_meta_rl: bool = True            # Enable meta-RL
    baseline_lr: float = 0.001          # Baseline learning rate
    policy_lr: float = 0.0003           # Policy learning rate
    entropy_weight: float = 0.01        # Policy entropy regularization
    gamma: float = 0.99                 # Discount factor for rewards
    
    # Gradient Management
    detach_low_to_high: bool = True     # Block gradients low->high
    detach_high_to_low: bool = False    # Block gradients high->low
    gradient_checkpointing: bool = False # Enable gradient checkpointing
    max_grad_norm: float = 1.0          # Gradient clipping threshold
    
    # Network Architecture Details
    dropout_rate: float = 0.1           # Dropout probability
    layer_norm: bool = True             # Use layer normalization
    residual_connections: bool = True   # Use residual connections
    attention_mechanism: bool = False   # Use attention for aggregation
    
    # Optimization
    weight_decay: float = 1e-4          # L2 regularization
    warmup_steps: int = 1000            # Learning rate warmup
    use_scheduler: bool = True          # Use learning rate scheduler
    
    # Training Dynamics
    teacher_forcing_ratio: float = 0.5  # Teacher forcing probability
    curriculum_learning: bool = False   # Enable curriculum learning
    noise_injection: float = 0.0        # Input noise for regularization
    
    # Hardware and Efficiency
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True        # Use mixed precision training
    compile_model: bool = False         # Use torch.compile (PyTorch 2.0+)
    
    # Monitoring and Debugging
    log_interval: int = 100             # Logging frequency
    save_checkpoints: bool = True       # Save model checkpoints
    track_gradients: bool = False       # Monitor gradient norms
    
    # Advanced Features
    multi_head_attention: bool = False  # Multi-head attention
    positional_encoding: bool = False   # Positional encodings
    adaptive_embedding: bool = False    # Adaptive input embedding
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.T < 1:
            raise ValueError("T (low-level steps) must be >= 1")
        if self.max_cycles < self.min_cycles:
            raise ValueError("max_cycles must be >= min_cycles")
        if self.hidden_low < 1 or self.hidden_high < 1:
            raise ValueError("Hidden dimensions must be positive")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError("Dropout rate must be in [0, 1)")
        
        logger.info(f"HRM Config initialized: {self.hidden_low}→{self.hidden_high}, "
                   f"T={self.T}, max_cycles={self.max_cycles}, ACT={self.use_act}")


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence modeling.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for feature aggregation.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply multi-head attention."""
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        return self.w_o(context)


class AdaptiveInputEmbedding(nn.Module):
    """
    Adaptive input embedding that learns to emphasize important features.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adaptive embedding using mixture of experts."""
        batch_size = x.size(0)
        
        # Compute expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        
        # Compute gating weights
        gate_weights = self.gate(x).unsqueeze(1)  # [B, 1, num_experts]
        
        # Weighted combination
        output = torch.bmm(expert_outputs, gate_weights.transpose(1, 2)).squeeze(2)
        
        return output


class EnhancedInputNetwork(nn.Module):
    """
    Enhanced input network with optional adaptive embedding and positional encoding.
    """
    
    def __init__(self, config: HierarchicalModelConfig):
        super().__init__()
        self.config = config
        
        if config.adaptive_embedding:
            self.embedding = AdaptiveInputEmbedding(
                config.input_dim, config.hidden_low
            )
        else:
            self.embedding = nn.Sequential(
                nn.Linear(config.input_dim, config.hidden_low),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            )
        
        if config.positional_encoding:
            self.pos_encoding = PositionalEncoding(config.hidden_low)
        
        if config.layer_norm:
            self.norm = nn.LayerNorm(config.hidden_low)
        
        # Input projection layers
        self.input_projection = nn.Sequential(
            nn.Linear(config.hidden_low, config.hidden_low),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_low, config.hidden_low)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through embedding and projection layers."""
        # Apply noise injection for regularization
        if self.training and self.config.noise_injection > 0:
            noise = torch.randn_like(x) * self.config.noise_injection
            x = x + noise
        
        # Input embedding
        x = self.embedding(x)
        
        # Positional encoding
        if self.config.positional_encoding:
            x = self.pos_encoding(x.unsqueeze(0)).squeeze(0)
        
        # Layer normalization
        if self.config.layer_norm:
            x = self.norm(x)
        
        # Input projection with residual connection
        if self.config.residual_connections:
            x = x + self.input_projection(x)
        else:
            x = self.input_projection(x)
        
        return x


class AdvancedLowLevelModule(nn.Module):
    """
    Advanced low-level processing module with attention and normalization.
    """
    
    def __init__(self, config: HierarchicalModelConfig):
        super().__init__()
        self.config = config
        
        # Core GRU cell
        self.gru = nn.GRUCell(config.hidden_low, config.hidden_low)
        
        # Top-down connection from high-level
        self.cross_connection = nn.Sequential(
            nn.Linear(config.hidden_high, config.hidden_low),
            nn.Tanh()
        )
        
        # Self-attention for internal processing
        if config.attention_mechanism:
            self.self_attention = MultiHeadAttention(
                config.hidden_low, num_heads=8, dropout=config.dropout_rate
            )
        
        # Layer normalization
        if config.layer_norm:
            self.norm1 = nn.LayerNorm(config.hidden_low)
            self.norm2 = nn.LayerNorm(config.hidden_low)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_low, config.hidden_low * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_low * 2, config.hidden_low)
        )
    
    def forward(self, z_L: torch.Tensor, z_H: torch.Tensor, 
                x_tilde: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through low-level module.
        
        Args:
            z_L: Low-level hidden state [B, hidden_low]
            z_H: High-level hidden state [B, hidden_high]
            x_tilde: Input embedding [B, hidden_low]
        
        Returns:
            Updated low-level hidden state [B, hidden_low]
        """
        # Top-down signal from high-level
        top_down = self.cross_connection(z_H)
        
        # Combine input, previous state, and top-down signal
        combined_input = x_tilde + top_down
        
        # GRU update
        h_new = self.gru(combined_input, z_L)
        
        # Apply dropout
        h_new = self.dropout(h_new)
        
        # Layer normalization and residual connection
        if self.config.layer_norm:
            h_new = self.norm1(h_new)
        
        if self.config.residual_connections:
            h_new = h_new + z_L
        
        # Self-attention (if enabled)
        if self.config.attention_mechanism:
            h_attended = self.self_attention(h_new, h_new, h_new)
            h_new = h_new + self.dropout(h_attended)
            
            if self.config.layer_norm:
                h_new = self.norm2(h_new)
        
        # Feed-forward network
        ffn_output = self.ffn(h_new)
        
        if self.config.residual_connections:
            h_new = h_new + self.dropout(ffn_output)
        else:
            h_new = self.dropout(ffn_output)
        
        return h_new


class AdvancedHighLevelModule(nn.Module):
    """
    Advanced high-level processing module with attention aggregation.
    """
    
    def __init__(self, config: HierarchicalModelConfig):
        super().__init__()
        self.config = config
        
        # Aggregation layer for low-level information
        if config.attention_mechanism:
            self.aggregation = MultiHeadAttention(
                config.hidden_low, num_heads=8, dropout=config.dropout_rate
            )
        else:
            self.aggregation = nn.Sequential(
                nn.Linear(config.hidden_low, config.hidden_high),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            )
        
        # Core GRU cell
        self.gru = nn.GRUCell(config.hidden_high, config.hidden_high)
        
        # Layer normalization
        if config.layer_norm:
            self.norm1 = nn.LayerNorm(config.hidden_high)
            self.norm2 = nn.LayerNorm(config.hidden_high)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Feed-forward processing
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_high, config.hidden_high * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_high * 2, config.hidden_high)
        )
        
        # Context integration
        self.context_gate = nn.Sequential(
            nn.Linear(config.hidden_high * 2, config.hidden_high),
            nn.Sigmoid()
        )
    
    def forward(self, z_H: torch.Tensor, z_L_summary: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through high-level module.
        
        Args:
            z_H: High-level hidden state [B, hidden_high]
            z_L_summary: Aggregated low-level information [B, hidden_low or hidden_high]
        
        Returns:
            Updated high-level hidden state [B, hidden_high]
        """
        # Aggregate low-level information
        if self.config.attention_mechanism:
            # Use attention for aggregation (treating z_L_summary as both key and value)
            aggregated = self.aggregation(
                z_L_summary.unsqueeze(1), 
                z_L_summary.unsqueeze(1), 
                z_L_summary.unsqueeze(1)
            ).squeeze(1)
        else:
            # Simple linear aggregation
            aggregated = self.aggregation(z_L_summary)
        
        # Context gating mechanism
        context = torch.cat([z_H, aggregated], dim=1)
        gate = self.context_gate(context)
        gated_input = aggregated * gate
        
        # GRU update
        h_new = self.gru(gated_input, z_H)
        
        # Dropout and normalization
        h_new = self.dropout(h_new)
        
        if self.config.layer_norm:
            h_new = self.norm1(h_new)
        
        # Residual connection
        if self.config.residual_connections:
            h_new = h_new + z_H
        
        # Feed-forward processing
        ffn_output = self.ffn(h_new)
        
        if self.config.residual_connections:
            h_new = h_new + self.dropout(ffn_output)
        else:
            h_new = self.dropout(ffn_output)
        
        if self.config.layer_norm:
            h_new = self.norm2(h_new)
        
        return h_new


class EnhancedOutputNetwork(nn.Module):
    """
    Enhanced output network with multiple prediction heads and uncertainty estimation.
    """
    
    def __init__(self, config: HierarchicalModelConfig):
        super().__init__()
        self.config = config
        
        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(config.hidden_high, config.hidden_high),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_high, config.hidden_high // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # Main prediction head
        self.prediction_head = nn.Linear(config.hidden_high // 2, config.output_dim)
        
        # Uncertainty estimation head (for regression tasks)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.hidden_high // 2, config.hidden_high // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_high // 4, config.output_dim),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
        # Layer normalization
        if config.layer_norm:
            self.norm = nn.LayerNorm(config.hidden_high)
    
    def forward(self, z_H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate predictions and uncertainty estimates.
        
        Args:
            z_H: High-level hidden state [B, hidden_high]
        
        Returns:
            Tuple of (predictions, uncertainties)
        """
        # Layer normalization
        if self.config.layer_norm:
            z_H = self.norm(z_H)
        
        # Feature processing
        features = self.feature_processor(z_H)
        
        # Predictions
        predictions = self.prediction_head(features)
        
        # Uncertainty estimates
        uncertainties = self.uncertainty_head(features)
        
        return predictions, uncertainties


class AdvancedACTController(nn.Module):
    """
    Advanced Adaptive Computation Time controller with sophisticated policy learning.
    """
    
    def __init__(self, config: HierarchicalModelConfig):
        super().__init__()
        self.config = config
        
        # Policy network for continue/stop decisions
        self.policy_network = nn.Sequential(
            nn.Linear(config.hidden_high, config.hidden_high // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_high // 2, config.hidden_high // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_high // 4, 2)  # [continue, stop]
        )
        
        # Value network for baseline estimation
        self.value_network = nn.Sequential(
            nn.Linear(config.hidden_high, config.hidden_high // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_high // 2, config.hidden_high // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_high // 4, 1)
        )
        
        # Halting probability network (for traditional ACT)
        self.halt_network = nn.Sequential(
            nn.Linear(config.hidden_high, config.hidden_high // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_high // 4, 1),
            nn.Sigmoid()
        )
    
    def compute_policy_action(self, z_H: torch.Tensor, 
                            cycle: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute policy-based continue/stop decisions.
        
        Args:
            z_H: High-level hidden state [B, hidden_high]
            cycle: Current cycle number
        
        Returns:
            Tuple of (actions, logits) where actions are 0 (continue) or 1 (stop)
        """
        # Add cycle information to the state
        batch_size = z_H.size(0)
        cycle_embed = torch.full((batch_size, 1), cycle / self.config.max_cycles, 
                               device=z_H.device, dtype=z_H.dtype)
        
        # Enhanced state representation
        enhanced_state = torch.cat([z_H, cycle_embed], dim=1)
        enhanced_input = F.linear(enhanced_state, 
                                torch.cat([torch.eye(self.config.hidden_high, device=z_H.device),
                                         torch.ones(self.config.hidden_high, 1, device=z_H.device)], dim=1))
        
        # Policy logits
        logits = self.policy_network(enhanced_input)
        
        # Sample actions (during training) or take greedy actions (during inference)
        if self.training:
            probs = F.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(-1)
        else:
            actions = logits.argmax(dim=-1)
        
        # Force stop at maximum cycles
        if cycle >= self.config.max_cycles - 1:
            actions = torch.ones_like(actions)
        
        return actions, logits
    
    def compute_halt_probability(self, z_H: torch.Tensor) -> torch.Tensor:
        """
        Compute halting probabilities for traditional ACT.
        
        Args:
            z_H: High-level hidden state [B, hidden_high]
        
        Returns:
            Halting probabilities [B, 1]
        """
        return self.halt_network(z_H)
    
    def compute_baseline(self, z_H: torch.Tensor) -> torch.Tensor:
        """
        Compute value function baseline for policy gradients.
        
        Args:
            z_H: High-level hidden state [B, hidden_high]
        
        Returns:
            Value estimates [B, 1]
        """
        return self.value_network(z_H)
    
    def compute_rewards(self, task_losses: torch.Tensor, 
                       cycles_used: torch.Tensor) -> torch.Tensor:
        """
        Compute rewards for meta-reinforcement learning.
        
        Args:
            task_losses: Per-sample task losses [B]
            cycles_used: Number of cycles used per sample [B]
        
        Returns:
            Rewards [B]
        """
        # Reward = negative loss - computation penalty
        accuracy_reward = -task_losses.detach()
        computation_penalty = self.config.act_penalty * cycles_used.float()
        
        rewards = accuracy_reward - computation_penalty
        
        return rewards


class AdvancedMetaLearningOptimizer:
    """
    Advanced optimizer for meta-learning with multiple components.
    """
    
    def __init__(self, model_params: Dict[str, nn.Parameter], config: HierarchicalModelConfig):
        self.config = config
        
        # Separate optimizers for different components
        self.optimizers = {
            'model': torch.optim.AdamW(
                model_params.get('model', []), 
                lr=config.baseline_lr,
                weight_decay=config.weight_decay
            ),
            'policy': torch.optim.Adam(
                model_params.get('policy', []), 
                lr=config.policy_lr
            ),
            'baseline': torch.optim.Adam(
                model_params.get('baseline', []), 
                lr=config.baseline_lr
            )
        }
        
        # Learning rate schedulers
        if config.use_scheduler:
            self.schedulers = {
                name: torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=10000
                ) for name, optimizer in self.optimizers.items()
            }
        else:
            self.schedulers = {}
    
    def zero_grad(self):
        """Zero gradients for all optimizers."""
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
    
    def step(self, max_grad_norm: Optional[float] = None):
        """Perform optimization step with gradient clipping."""
        if max_grad_norm is not None:
            for name, optimizer in self.optimizers.items():
                params = [p for group in optimizer.param_groups for p in group['params']]
                if params:
                    clip_grad_norm_(params, max_grad_norm)
        
        for optimizer in self.optimizers.values():
            optimizer.step()
    
    def scheduler_step(self):
        """Step all learning rate schedulers."""
        for scheduler in self.schedulers.values():
            scheduler.step()


class ComprehensiveHierarchicalModel(nn.Module):
    """
    Comprehensive implementation of the Hierarchical Reasoning Model
    with all advanced features and optimizations.
    """
    
    def __init__(self, config: HierarchicalModelConfig):
        super().__init__()
        self.config = config
        
        # Core components
        self.input_network = EnhancedInputNetwork(config)
        self.low_level_module = AdvancedLowLevelModule(config)
        self.high_level_module = AdvancedHighLevelModule(config)
        self.output_network = EnhancedOutputNetwork(config)
        self.act_controller = AdvancedACTController(config)
        
        # Training metrics storage
        self.training_metrics = defaultdict(list)
        self.step_count = 0
        
        # Initialize parameters
        self._initialize_parameters()
        
        # Move to device
        self.to(config.device)
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Mixed precision scaler
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info(f"Comprehensive HRM initialized with {self.count_parameters()} parameters")
    
    def _initialize_parameters(self):
        """Initialize model parameters using Xavier initialization."""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def _setup_optimizer(self):
        """Setup optimizers for different parameter groups."""
        # Group parameters by component
        model_params = []
        policy_params = []
        baseline_params = []
        
        for name, param in self.named_parameters():
            if 'act_controller.policy' in name:
                policy_params.append(param)
            elif 'act_controller.value' in name:
                baseline_params.append(param)
            else:
                model_params.append(param)
        
        param_groups = {
            'model': model_params,
            'policy': policy_params,
            'baseline': baseline_params
        }
        
        self.optimizer = AdvancedMetaLearningOptimizer(param_groups, self.config)
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def initialize_states(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden states for both processing levels.
        
        Args:
            batch_size: Batch size
        
        Returns:
            Tuple of (low_level_state, high_level_state)
        """
        device = next(self.parameters()).device
        
        z_L = torch.zeros(batch_size, self.config.hidden_low, device=device)
        z_H = torch.zeros(batch_size, self.config.hidden_high, device=device)
        
        return z_L, z_H
    
    def forward_single_cycle(self, z_L: torch.Tensor, z_H: torch.Tensor, 
                           x_tilde: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a single reasoning cycle.
        
        Args:
            z_L: Low-level hidden state [B, hidden_low]
            z_H: High-level hidden state [B, hidden_high] 
            x_tilde: Input embedding [B, hidden_low]
        
        Returns:
            Tuple of updated (z_L, z_H)
        """
        # Low-level processing (T steps)
        for _ in range(self.config.T):
            # Apply gradient detachment if configured
            z_H_input = z_H.detach() if self.config.detach_high_to_low else z_H
            
            if self.config.gradient_checkpointing and self.training:
                z_L = checkpoint(self.low_level_module, z_L, z_H_input, x_tilde)
            else:
                z_L = self.low_level_module(z_L, z_H_input, x_tilde)
        
        # High-level processing (1 step)
        z_L_input = z_L.detach() if self.config.detach_low_to_high else z_L
        
        if self.config.gradient_checkpointing and self.training:
            z_H = checkpoint(self.high_level_module, z_H, z_L_input)
        else:
            z_H = self.high_level_module(z_H, z_L_input)
        
        return z_L, z_H
    
    def forward(self, x: torch.Tensor, 
                num_cycles: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the hierarchical model.
        
        Args:
            x: Input tensor [B, input_dim]
            num_cycles: Fixed number of cycles (if None, uses ACT)
        
        Returns:
            Dictionary containing predictions, uncertainties, and metadata
        """
        batch_size = x.size(0)
        
        # Input processing
        x_tilde = self.input_network(x)
        
        # Initialize states
        z_L, z_H = self.initialize_states(batch_size)
        
        # Tracking variables
        cycles_used = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        halting_probs = []
        policy_logits = []
        value_estimates = []
        
        # Reasoning cycles
        for cycle in range(self.config.max_cycles):
            # Perform reasoning cycle
            z_L, z_H = self.forward_single_cycle(z_L, z_H, x_tilde)
            
            # Update cycle counts
            cycles_used += 1
            
            # Store value estimates
            if self.config.use_meta_rl:
                value_est = self.act_controller.compute_baseline(z_H.detach())
                value_estimates.append(value_est)
            
            # Decide whether to continue or stop
            if num_cycles is not None:
                # Fixed number of cycles
                if cycle >= num_cycles - 1:
                    break
            elif self.config.use_act:
                if self.config.use_meta_rl:
                    # Policy-based ACT
                    actions, logits = self.act_controller.compute_policy_action(z_H.detach(), cycle)
                    policy_logits.append(logits)
                    
                    # Check if all samples want to stop (or we're at max cycles)
                    if (actions == 1).all() or cycle >= self.config.max_cycles - 1:
                        break
                else:
                    # Traditional ACT with halting probabilities
                    halt_prob = self.act_controller.compute_halt_probability(z_H.detach())
                    halting_probs.append(halt_prob)
                    
                    # Check halting condition
                    if (halt_prob > self.config.act_threshold).all() or cycle >= self.config.max_cycles - 1:
                        break
            else:
                # No ACT, use fixed max cycles
                if cycle >= self.config.max_cycles - 1:
                    break
        
        # Generate final output
        predictions, uncertainties = self.output_network(z_H)
        
        # Prepare output dictionary
        output = {
            'predictions': predictions,
            'uncertainties': uncertainties,
            'cycles_used': cycles_used,
            'final_state_low': z_L,
            'final_state_high': z_H
        }
        
        # Add ACT-specific outputs
        if policy_logits:
            output['policy_logits'] = torch.stack(policy_logits, dim=1)
        if value_estimates:
            output['value_estimates'] = torch.stack(value_estimates, dim=1)
        if halting_probs:
            output['halting_probs'] = torch.stack(halting_probs, dim=1)
        
        return output
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                    uncertainties: Optional[torch.Tensor] = None,
                    cycles_used: Optional[torch.Tensor] = None,
                    policy_logits: Optional[torch.Tensor] = None,
                    value_estimates: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss including task loss, ACT penalty, and meta-RL terms.
        
        Args:
            predictions: Model predictions [B, output_dim]
            targets: Ground truth targets [B, output_dim]
            uncertainties: Prediction uncertainties [B, output_dim]
            cycles_used: Number of cycles used [B]
            policy_logits: Policy network logits [B, num_cycles, 2]
            value_estimates: Value function estimates [B, num_cycles, 1]
        
        Returns:
            Dictionary containing loss components
        """
        losses = {}
        
        # Task loss (with uncertainty weighting if available)
        if uncertainties is not None:
            # Uncertainty-weighted loss
            precision = 1.0 / (uncertainties + self.config.act_epsilon)
            task_loss = F.mse_loss(predictions, targets, reduction='none')
            task_loss = (task_loss * precision + torch.log(uncertainties)).mean()
        else:
            # Standard MSE loss
            task_loss = F.mse_loss(predictions, targets)
        
        losses['task_loss'] = task_loss
        
        # ACT penalty
        if cycles_used is not None:
            act_penalty = self.config.act_penalty * cycles_used.float().mean()
            losses['act_penalty'] = act_penalty
        else:
            losses['act_penalty'] = torch.tensor(0.0, device=predictions.device)
        
        # Meta-RL losses
        if policy_logits is not None and value_estimates is not None and cycles_used is not None:
            # Compute rewards
            task_losses_per_sample = F.mse_loss(predictions, targets, reduction='none').mean(dim=1)
            rewards = self.act_controller.compute_rewards(task_losses_per_sample, cycles_used)
            
            # Policy loss (REINFORCE with baseline)
            batch_size = policy_logits.size(0)
            num_cycles = policy_logits.size(1)
            
            policy_loss = 0.0
            value_loss = 0.0
            entropy_loss = 0.0
            
            for i in range(num_cycles):
                cycle_logits = policy_logits[:, i, :]  # [B, 2]
                cycle_values = value_estimates[:, i, :].squeeze(-1)  # [B]
                
                # Action probabilities
                probs = F.softmax(cycle_logits, dim=-1)
                log_probs = F.log_softmax(cycle_logits, dim=-1)
                
                # Sample actions (during training, use actual decisions made)
                if i < cycles_used.max():
                    # For cycles that were actually executed
                    mask = (cycles_used > i).float()
                    
                    # Advantage estimation
                    advantages = (rewards - cycle_values.detach()) * mask
                    
                    # Policy loss
                    action_log_probs = log_probs[:, 1]  # Log prob of stopping
                    policy_loss += -(action_log_probs * advantages).mean()
                    
                    # Entropy regularization
                    entropy = -(probs * log_probs).sum(dim=-1)
                    entropy_loss += -entropy.mean()
                    
                    # Value loss
                    value_targets = rewards * mask
                    value_loss += F.mse_loss(cycle_values, value_targets.detach())
            
            losses['policy_loss'] = policy_loss
            losses['value_loss'] = value_loss
            losses['entropy_loss'] = self.config.entropy_weight * entropy_loss
        else:
            losses['policy_loss'] = torch.tensor(0.0, device=predictions.device)
            losses['value_loss'] = torch.tensor(0.0, device=predictions.device)
            losses['entropy_loss'] = torch.tensor(0.0, device=predictions.device)
        
        # Total loss
        losses['total_loss'] = (losses['task_loss'] + 
                              losses['act_penalty'] + 
                              losses['policy_loss'] + 
                              losses['value_loss'] + 
                              losses['entropy_loss'])
        
        return losses
    
    def training_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            x: Input batch [B, input_dim]
            y: Target batch [B, output_dim]
        
        Returns:
            Dictionary of training metrics
        """
        self.train()
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if self.config.mixed_precision:
            with torch.cuda.amp.autocast():
                output = self.forward(x)
                loss_dict = self.compute_loss(
                    output['predictions'], y, 
                    output.get('uncertainties'),
                    output.get('cycles_used'),
                    output.get('policy_logits'),
                    output.get('value_estimates')
                )
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss_dict['total_loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            output = self.forward(x)
            loss_dict = self.compute_loss(
                output['predictions'], y,
                output.get('uncertainties'),
                output.get('cycles_used'),
                output.get('policy_logits'),
                output.get('value_estimates')
            )
            
            # Backward pass
            loss_dict['total_loss'].backward()
            self.optimizer.step(self.config.max_grad_norm)
        
        # Learning rate scheduling
        self.optimizer.scheduler_step()
        
        # Update step count
        self.step_count += 1
        
        # Convert losses to float for logging
        metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        metrics['cycles_mean'] = output['cycles_used'].float().mean().item()
        metrics['learning_rate'] = self.optimizer.optimizers['model'].param_groups[0]['lr']
        
        # Store metrics
        if self.step_count % self.config.log_interval == 0:
            for key, value in metrics.items():
                self.training_metrics[key].append(value)
            
            logger.info(f"Step {self.step_count}: Loss={metrics['total_loss']:.4f}, "
                       f"Cycles={metrics['cycles_mean']:.2f}, LR={metrics['learning_rate']:.6f}")
        
        return metrics
    
    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate the model on a batch of data.
        
        Args:
            x: Input batch [B, input_dim]
            y: Target batch [B, output_dim]
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.eval()
        
        with torch.no_grad():
            output = self.forward(x)
            loss_dict = self.compute_loss(
                output['predictions'], y,
                output.get('uncertainties'),
                output.get('cycles_used'),
                output.get('policy_logits'),
                output.get('value_estimates')
            )
        
        # Additional evaluation metrics
        predictions = output['predictions']
        mse = F.mse_loss(predictions, y)
        mae = F.l1_loss(predictions, y)
        
        metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        metrics.update({
            'mse': mse.item(),
            'mae': mae.item(),
            'cycles_mean': output['cycles_used'].float().mean().item(),
            'cycles_std': output['cycles_used'].float().std().item()
        })
        
        return metrics
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': {
                name: opt.state_dict() for name, opt in self.optimizer.optimizers.items()
            },
            'config': self.config,
            'step_count': self.step_count,
            'training_metrics': dict(self.training_metrics)
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        for name, state_dict in checkpoint['optimizer_state_dict'].items():
            if name in self.optimizer.optimizers:
                self.optimizer.optimizers[name].load_state_dict(state_dict)
        
        self.step_count = checkpoint.get('step_count', 0)
        self.training_metrics = defaultdict(list, checkpoint.get('training_metrics', {}))
        
        logger.info(f"Checkpoint loaded from {filepath}")


# Factory function for creating models
def create_hierarchical_model(config: Optional[HierarchicalModelConfig] = None) -> ComprehensiveHierarchicalModel:
    """
    Factory function to create a hierarchical reasoning model.
    
    Args:
        config: Model configuration (uses default if None)
    
    Returns:
        Initialized hierarchical model
    """
    if config is None:
        config = HierarchicalModelConfig()
    
    model = ComprehensiveHierarchicalModel(config)
    
    # Compile model if requested and available
    if config.compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            logger.info("Model compiled successfully")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")
    
    return model