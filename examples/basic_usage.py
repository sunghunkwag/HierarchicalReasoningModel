#!/usr/bin/env python3
"""
Basic Usage Example for Hierarchical Reasoning Model

This script demonstrates how to:
1. Create and configure the HRM model
2. Run inference on sample data
3. Train the model with synthetic data
4. Monitor training metrics
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import sys
import os

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from hierarchical_reasoning_model import HierarchicalReasoningModel, HRMConfig


def create_synthetic_dataset(num_samples=1000, input_dim=1024, complexity_levels=3):
    """
    Create synthetic dataset with varying complexity levels.
    Easy samples should require fewer reasoning cycles, hard samples more.
    """
    # Generate random inputs
    X = torch.randn(num_samples, input_dim)
    
    # Create targets with different complexity levels
    # Easy: simple linear combination
    # Medium: quadratic features
    # Hard: complex interactions
    
    easy_mask = torch.randperm(num_samples)[:num_samples//3]
    medium_mask = torch.randperm(num_samples)[num_samples//3:2*num_samples//3]
    hard_mask = torch.randperm(num_samples)[2*num_samples//3:]
    
    Y = torch.zeros(num_samples, 1)
    
    # Easy targets: linear
    Y[easy_mask] = X[easy_mask, :10].sum(dim=1, keepdim=True) * 0.1
    
    # Medium targets: quadratic
    Y[medium_mask] = (X[medium_mask, :50].pow(2).sum(dim=1, keepdim=True) - 
                      X[medium_mask, 50:100].sum(dim=1, keepdim=True)) * 0.01
    
    # Hard targets: complex interactions
    Y[hard_mask] = (torch.sin(X[hard_mask, :20].sum(dim=1)) * 
                    torch.cos(X[hard_mask, 20:40].sum(dim=1)) + 
                    X[hard_mask, 40:60].prod(dim=1, keepdim=True).clamp(-1, 1)).unsqueeze(1)
    
    return TensorDataset(X, Y)


def inference_example():
    """
    Demonstrate basic inference with the HRM model.
    """
    print("=== Inference Example ===")
    
    # Configure model
    config = HRMConfig(
        hidden_low=128,
        hidden_high=256,
        T=3,                    # 3 low-level steps per cycle
        max_cycles=5,           # Up to 5 reasoning cycles
        use_act=True,           # Enable adaptive computation
        compute_cost_weight=0.01
    )
    
    # Create model
    model = HierarchicalReasoningModel(config)
    model.eval()
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Sample input
    batch_size = 8
    input_dim = 1024
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    with torch.no_grad():
        output, cycles_used = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Cycles used per sample: {cycles_used.tolist()}")
    print(f"Average cycles: {cycles_used.float().mean():.2f}")
    
    return model, config


def training_example(model=None, config=None, num_epochs=10):
    """
    Demonstrate training with synthetic data.
    """
    print("\n=== Training Example ===")
    
    if model is None:
        config = HRMConfig(
            hidden_low=128,
            hidden_high=256,
            T=3,
            max_cycles=5,
            use_act=True,
            compute_cost_weight=0.02  # Slightly higher penalty for training
        )
        model = HierarchicalReasoningModel(config)
    
    # Create synthetic dataset
    dataset = create_synthetic_dataset(num_samples=1000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training setup
    criterion = nn.MSELoss()
    model.train()
    
    # Track metrics
    metrics_history = {
        'loss': [],
        'task_loss': [],
        'policy_loss': [],
        'baseline_loss': [],
        'cycles_mean': []
    }
    
    print(f"Training for {num_epochs} epochs on {len(dataset)} samples...")
    
    for epoch in range(num_epochs):
        epoch_metrics = {key: [] for key in metrics_history.keys()}
        
        for batch_idx, (x, y) in enumerate(dataloader):
            # Training step
            metrics = model.training_step(x, y, criterion)
            
            # Collect metrics
            for key in epoch_metrics.keys():
                epoch_metrics[key].append(metrics[key])
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}: "
                      f"Loss={metrics['loss']:.4f}, Cycles={metrics['cycles_mean']:.2f}")
        
        # Average metrics for epoch
        for key in metrics_history.keys():
            metrics_history[key].append(np.mean(epoch_metrics[key]))
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Total Loss: {metrics_history['loss'][-1]:.4f}")
        print(f"  Task Loss: {metrics_history['task_loss'][-1]:.4f}")
        print(f"  Policy Loss: {metrics_history['policy_loss'][-1]:.4f}")
        print(f"  Avg Cycles: {metrics_history['cycles_mean'][-1]:.2f}")
    
    return model, metrics_history


def evaluate_adaptive_computation(model, config):
    """
    Evaluate how the model adapts computation to problem difficulty.
    """
    print("\n=== Adaptive Computation Evaluation ===")
    
    model.eval()
    
    # Create samples with known difficulty levels
    easy_samples = torch.randn(20, 1024) * 0.5  # Low variance -> easier
    hard_samples = torch.randn(20, 1024) * 2.0  # High variance -> harder
    
    with torch.no_grad():
        easy_output, easy_cycles = model(easy_samples)
        hard_output, hard_cycles = model(hard_samples)
    
    print(f"Easy samples - Average cycles: {easy_cycles.float().mean():.2f} (std: {easy_cycles.float().std():.2f})")
    print(f"Hard samples - Average cycles: {hard_cycles.float().mean():.2f} (std: {hard_cycles.float().std():.2f})")
    
    if easy_cycles.float().mean() < hard_cycles.float().mean():
        print("✅ Model successfully adapts computation to difficulty!")
    else:
        print("⚠️  Model may need more training to learn adaptive computation.")
    
    return {
        'easy_cycles': easy_cycles.float().numpy(),
        'hard_cycles': hard_cycles.float().numpy()
    }


def plot_training_metrics(metrics_history):
    """
    Plot training metrics over time.
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('HRM Training Metrics')
        
        # Loss components
        axes[0, 0].plot(metrics_history['loss'], label='Total Loss')
        axes[0, 0].plot(metrics_history['task_loss'], label='Task Loss')
        axes[0, 0].plot(metrics_history['policy_loss'], label='Policy Loss')
        axes[0, 0].set_title('Loss Components')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Baseline loss
        axes[0, 1].plot(metrics_history['baseline_loss'])
        axes[0, 1].set_title('Baseline Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Average cycles
        axes[1, 0].plot(metrics_history['cycles_mean'])
        axes[1, 0].set_title('Average Reasoning Cycles')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Cycles')
        axes[1, 0].grid(True)
        
        # Learning efficiency (task loss vs cycles)
        axes[1, 1].scatter(metrics_history['cycles_mean'], metrics_history['task_loss'], alpha=0.7)
        axes[1, 1].set_title('Efficiency: Task Loss vs Cycles')
        axes[1, 1].set_xlabel('Average Cycles')
        axes[1, 1].set_ylabel('Task Loss')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('hrm_training_metrics.png', dpi=150, bbox_inches='tight')
        print("📊 Training metrics plot saved as 'hrm_training_metrics.png'")
        
    except ImportError:
        print("📊 Matplotlib not available - skipping plots")


def main():
    """
    Main example script.
    """
    print("🧠 Hierarchical Reasoning Model - Basic Usage Example\n")
    
    # 1. Inference example
    model, config = inference_example()
    
    # 2. Training example
    model, metrics_history = training_example(model, config, num_epochs=5)
    
    # 3. Evaluate adaptive computation
    adaptation_results = evaluate_adaptive_computation(model, config)
    
    # 4. Plot results
    plot_training_metrics(metrics_history)
    
    print("\n✅ Example completed successfully!")
    print("\n📝 Key takeaways:")
    print("   - HRM adapts computation time based on problem complexity")
    print("   - Meta-RL learns to balance accuracy vs computational cost")
    print("   - Batch-wise ACT allows efficient training with variable cycles")
    print("   - Monitor policy loss to ensure ACT learning is working")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()
