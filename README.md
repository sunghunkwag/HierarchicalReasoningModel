# Hierarchical Reasoning Model (HRM)

[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A novel neural architecture that combines **Hierarchical Processing**, **Adaptive Computation Time (ACT)**, and **Meta-Reinforcement Learning** for dynamic multi-level reasoning. This implementation features batch-wise ACT processing for efficient training and per-sample advantage calculation for stable meta-learning.

## 🚀 Key Features

### 🧠 Hierarchical Architecture
- **Two-Level Processing**: Low-level (fast, detailed) and high-level (slow, abstract) reasoning modules
- **Bidirectional Information Flow**: Top-down control signals and bottom-up information aggregation
- **Recurrent State Management**: GRU-based memory for both processing levels

### ⚡ Adaptive Computation Time (ACT)
- **Dynamic Reasoning Cycles**: Automatically determines optimal computation steps per input
- **Batch-wise Processing**: Different samples can stop at different cycles for maximum efficiency
- **Policy-based Control**: Learned "continue/stop" decisions via reinforcement learning

### 🎯 Meta-Reinforcement Learning
- **Computation-Accuracy Trade-off**: Learns to balance performance vs. computational cost
- **Per-sample Rewards**: Individual advantage calculation for stable policy learning
- **Baseline Value Function**: Variance reduction for policy gradients

## 📋 Architecture Overview

```
Input → InputNetwork → Low-Level Module ↔ High-Level Module → OutputNetwork → Output
                           ↑                    ↓
                      [T iterations]     [N cycles with ACT]
                           ↑                    ↓
                    Meta-RL Controller ← QNetwork (continue/stop)
```

### Core Components:

1. **InputNetwork**: Encodes raw input into low-level hidden representations
2. **LowLevelModule**: Fast, detailed processing with GRU cells (T steps per cycle)
3. **HighLevelModule**: Abstract reasoning and control (1 step per cycle)
4. **QNetwork**: ACT policy network for continue/stop decisions
5. **MetaRLController**: Reward computation and policy optimization
6. **OutputNetwork**: Final prediction from high-level representations

## 🛠️ Installation

```bash
git clone https://github.com/sunghunkwag/HierarchicalReasoningModel.git
cd HierarchicalReasoningModel
pip install torch torchvision numpy
```

## 💡 Quick Start

### Basic Usage

```python
import torch
from src.hierarchical_reasoning_model import HierarchicalReasoningModel, HRMConfig

# Configure the model
config = HRMConfig(
    hidden_low=256,      # Low-level hidden dimension
    hidden_high=512,     # High-level hidden dimension  
    T=3,                 # Low-level steps per cycle
    max_cycles=5,        # Maximum reasoning cycles
    use_act=True,        # Enable Adaptive Computation Time
    compute_cost_weight=0.01  # Computation penalty weight
)

# Create model
model = HierarchicalReasoningModel(config)

# Forward pass
x = torch.randn(32, 1024)  # Batch of 32 samples, 1024-dim input
output, cycles_used = model(x)

print(f"Output shape: {output.shape}")
print(f"Average cycles used: {cycles_used.float().mean():.2f}")
```

### Training Example

```python
import torch.nn as nn
from torch.utils.data import DataLoader

# Setup
criterion = nn.MSELoss()
dataloader = DataLoader(your_dataset, batch_size=32)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (x, y) in enumerate(dataloader):
        metrics = model.training_step(x, y, criterion)
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}:")
            print(f"  Total Loss: {metrics['loss']:.4f}")
            print(f"  Task Loss: {metrics['task_loss']:.4f}")
            print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"  Avg Cycles: {metrics['cycles_mean']:.2f}")
```

## 🔧 Configuration Options

| Parameter | Description | Default |
|-----------|-------------|--------|
| `hidden_low` | Low-level hidden dimension | Required |
| `hidden_high` | High-level hidden dimension | Required |
| `T` | Low-level steps per high-level cycle | Required |
| `max_cycles` | Maximum reasoning cycles | Required |
| `use_act` | Enable Adaptive Computation Time | `True` |
| `device` | Computation device | `"cuda"` if available |
| `detach_low_to_high` | Gradient blocking low→high | `True` |
| `detach_high_to_low` | Gradient blocking high→low | `False` |
| `compute_cost_weight` | Computation penalty weight | `0.0` |

## 🧪 Key Innovations

### 1. Batch-wise ACT Processing
Unlike traditional ACT implementations that wait for all samples to complete, our approach allows different samples in a batch to stop at different cycles:

```python
# Each sample can stop independently
alive = torch.ones(B, dtype=torch.bool, device=device)
while alive.any() and cycle < max_cycles:
    # Update only alive samples
    z_H = torch.where(alive.unsqueeze(1), new_z_H, z_H)
    
    # Some samples may choose to stop
    just_stopped = alive & (action == 1)
    alive = alive & (~just_stopped)
```

### 2. Per-sample Advantage Calculation
Stable meta-learning through individual reward computation:

```python
# Per-sample loss and reward vectors
task_loss_vec = self._per_sample_loss(pred, y, criterion)  # [B]
reward = self.meta_controller.compute_reward(task_loss_vec, cycles_taken, cfg)  # [B]
advantage = reward - baseline_pred.detach()  # [B] vs [B]
```

### 3. Hierarchical Information Flow
Careful gradient management between processing levels:

```python
# Configurable gradient flow
top = z_H.detach() if cfg.detach_high_to_low else z_H
z_L_summary = z_L.detach() if cfg.detach_low_to_high else z_L
```

## 📊 Performance Characteristics

### Computational Efficiency
- **Dynamic Allocation**: Easy problems use fewer cycles, hard problems use more
- **No Wasted Computation**: Stopped samples don't participate in further processing
- **Gradient Efficiency**: Selective gradient blocking prevents unstable feedback loops

### Learning Stability
- **Normalized Advantages**: Prevents policy gradient explosion
- **Smooth L1 Baseline Loss**: Robust to reward variance
- **Configurable Gradient Flow**: Prevents circular dependencies

## 🔬 Research Applications

This architecture is particularly suitable for:

- **Few-shot Learning**: Adaptive reasoning for varying problem complexity
- **Meta-Learning**: Learning to learn across different task distributions
- **Reasoning Tasks**: Problems requiring multiple steps of inference
- **Resource-Constrained Inference**: Trading accuracy for computational efficiency
- **Curriculum Learning**: Gradually increasing reasoning complexity

## 📖 Technical Details

### ACT Policy Learning
The model learns a policy π(a|z_H) where:
- a ∈ {0, 1} (continue/stop)
- Reward = -task_loss - λ × cycles_used
- Policy loss = -log π(a|z_H) × advantage

### Hierarchical Processing
- **Low-level**: T GRU steps per cycle (fast, detailed)
- **High-level**: 1 GRU step per cycle (slow, abstract)
- **Cross-connections**: Top-down control, bottom-up aggregation

### Meta-Learning Objective
```
L_total = L_task + L_policy + L_baseline
where:
  L_policy = -𝔼[log π(a|s) × A(s,a)]
  L_baseline = SmoothL1(V(s), R)
  A(s,a) = R - V(s)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Work

- [Adaptive Computation Time (Graves, 2016)](https://arxiv.org/abs/1603.08983)
- [Hierarchical Attention Networks](https://www.aclweb.org/anthology/N16-1174/)
- [Meta-Learning with Differentiable Closed-form Solvers](https://openreview.net/forum?id=HyxnZh0ct7)

## 📞 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Note**: This is an experimental architecture developed for research purposes. The implementation focuses on clarity and modularity rather than maximum optimization.