#!/usr/bin/env python3
"""
Advanced Training Example for Hierarchical Reasoning Model

This script demonstrates:
1. Model checkpointing and loading
2. Learning rate scheduling
3. Early stopping
4. Comprehensive metrics tracking
5. Model evaluation utilities
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import sys
import os

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from hierarchical_reasoning_model import HierarchicalReasoningModel, HRMConfig
from utils import (
    save_checkpoint, 
    load_checkpoint,
    print_model_summary,
    evaluate_model,
    EarlyStopping,
    MetricsTracker,
    get_learning_rate
)


def create_synthetic_dataset(num_samples=2000, input_dim=1024):
    """
    Create synthetic dataset with varying complexity levels.
    """
    X = torch.randn(num_samples, input_dim)
    
    easy_mask = torch.randperm(num_samples)[:num_samples//3]
    medium_mask = torch.randperm(num_samples)[num_samples//3:2*num_samples//3]
    hard_mask = torch.randperm(num_samples)[2*num_samples//3:]
    
    Y = torch.zeros(num_samples, 1)
    
    # Easy targets: linear
    Y[easy_mask] = X[easy_mask, :10].sum(dim=1, keepdim=True) * 0.1
    
    # Medium targets: quadratic
    Y[medium_mask] = (X[medium_mask, :50].pow(2).sum(dim=1, keepdim=True) - 
                      X[medium_mask, 50:100].sum(dim=1, keepdim=True)) * 0.01
    
    # Hard targets: complex interactions (FIXED)
    Y[hard_mask] = (torch.sin(X[hard_mask, :20].sum(dim=1, keepdim=True)) * 
                    torch.cos(X[hard_mask, 20:40].sum(dim=1, keepdim=True)) + 
                    X[hard_mask, 40:60].prod(dim=1).clamp(-1, 1).unsqueeze(1))
    
    return TensorDataset(X, Y)


def train_with_validation():
    """
    Train model with validation set and advanced features.
    """
    print("🧠 Advanced Training Example for HRM\n")
    
    # Configuration
    config = HRMConfig(
        hidden_low=256,
        hidden_high=512,
        T=3,
        max_cycles=8,
        use_act=True,
        compute_cost_weight=0.02
    )
    
    # Create model
    model = HierarchicalReasoningModel(config)
    print_model_summary(model)
    
    # Create dataset and split into train/val
    full_dataset = create_synthetic_dataset(num_samples=2000)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Dataset: {train_size} train, {val_size} validation samples\n")
    
    # Training setup
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    # Note: model has internal optimizer, so we'll track LR separately
    # In production, you'd want to modify the model to expose optimizer
    
    # Early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    
    # Metrics tracking
    tracker = MetricsTracker()
    
    # Training loop
    num_epochs = 20
    best_val_loss = float('inf')
    
    print(f"Training for {num_epochs} epochs...\n")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_metrics = {
            'loss': [],
            'task_loss': [],
            'policy_loss': [],
            'cycles_mean': []
        }
        
        for batch_idx, (x, y) in enumerate(train_loader):
            metrics = model.training_step(x, y, criterion)
            
            for key in epoch_train_metrics.keys():
                epoch_train_metrics[key].append(metrics[key])
        
        # Average training metrics
        avg_train_metrics = {
            key: np.mean(values) 
            for key, values in epoch_train_metrics.items()
        }
        
        # Validation phase
        model.eval()
        val_metrics = evaluate_model(model, val_loader, criterion, config.device)
        
        # Track metrics
        tracker.update(
            epoch=epoch,
            train_loss=avg_train_metrics['loss'],
            train_task_loss=avg_train_metrics['task_loss'],
            train_policy_loss=avg_train_metrics['policy_loss'],
            train_cycles=avg_train_metrics['cycles_mean'],
            val_loss=val_metrics['loss'],
            val_cycles=val_metrics['avg_cycles']
        )
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {avg_train_metrics['loss']:.4f}, "
              f"Task: {avg_train_metrics['task_loss']:.4f}, "
              f"Policy: {avg_train_metrics['policy_loss']:.4f}, "
              f"Cycles: {avg_train_metrics['cycles_mean']:.2f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Cycles: {val_metrics['avg_cycles']:.2f} "
              f"(range: {val_metrics['min_cycles']}-{val_metrics['max_cycles']})")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model=model,
                epoch=epoch,
                metrics={
                    'train_loss': avg_train_metrics['loss'],
                    'val_loss': val_metrics['loss']
                },
                save_path='checkpoints/best_model.pth',
                config=config
            )
            print(f"  ✅ New best model saved!")
        
        # Early stopping check
        if early_stopping(val_metrics['loss']):
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            break
        
        print()
    
    # Save final metrics
    tracker.save('checkpoints/training_metrics.json')
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 70)
    
    return model, tracker


def demonstrate_checkpoint_loading():
    """
    Demonstrate loading a saved checkpoint.
    """
    print("\n\n🔄 Demonstrating Checkpoint Loading\n")
    
    checkpoint_path = 'checkpoints/best_model.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("   Run training first to create a checkpoint.")
        return
    
    # Load checkpoint metadata
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Recreate config from checkpoint
    config_dict = checkpoint.get('config', {})
    config = HRMConfig(**config_dict)
    
    # Create new model
    model = HierarchicalReasoningModel(config)
    
    # Load checkpoint
    load_checkpoint(model, checkpoint_path, device=config.device)
    
    # Test loaded model
    model.eval()
    x_test = torch.randn(8, 1024)
    
    with torch.no_grad():
        output, cycles = model(x_test)
    
    print(f"\n✅ Loaded model inference successful")
    print(f"   Output shape: {output.shape}")
    print(f"   Cycles used: {cycles.tolist()}")


def main():
    """
    Main function to run advanced training example.
    """
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Train model with validation
    model, tracker = train_with_validation()
    
    # Demonstrate checkpoint loading
    demonstrate_checkpoint_loading()
    
    print("\n📝 Key Features Demonstrated:")
    print("   ✅ Train/validation split")
    print("   ✅ Model checkpointing")
    print("   ✅ Early stopping")
    print("   ✅ Metrics tracking and saving")
    print("   ✅ Model evaluation utilities")
    print("   ✅ Checkpoint loading and inference")


if __name__ == "__main__":
    main()

