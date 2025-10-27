#!/usr/bin/env python3
"""
Utility functions for Hierarchical Reasoning Model

This module provides helper functions for:
- Model saving and loading
- Checkpoint management
- Evaluation utilities
- Visualization helpers
"""

import os
import torch
from typing import Dict, Optional, Any
from pathlib import Path
import json


def save_checkpoint(
    model,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    save_path: str = "checkpoint.pth",
    config: Optional[Any] = None
) -> None:
    """
    Save model checkpoint with training state.
    
    Args:
        model: The HierarchicalReasoningModel instance
        optimizer: Optional optimizer state
        epoch: Current epoch number
        metrics: Optional training metrics
        save_path: Path to save checkpoint
        config: Optional model configuration
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metrics': metrics or {},
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if config is not None:
        # Convert config to dict if it's a dataclass
        if hasattr(config, '__dataclass_fields__'):
            config_dict = {
                field: getattr(config, field) 
                for field in config.__dataclass_fields__
            }
        else:
            config_dict = vars(config)
        checkpoint['config'] = config_dict
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    torch.save(checkpoint, save_path)
    print(f"✅ Checkpoint saved to {save_path}")


def load_checkpoint(
    model,
    load_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Load model checkpoint and restore training state.
    
    Args:
        model: The HierarchicalReasoningModel instance
        load_path: Path to checkpoint file
        optimizer: Optional optimizer to restore state
        device: Device to load model to
    
    Returns:
        Dictionary containing checkpoint metadata
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_path}")
    
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✅ Checkpoint loaded from {load_path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
    
    if 'metrics' in checkpoint and checkpoint['metrics']:
        print(f"   Metrics: {checkpoint['metrics']}")
    
    return checkpoint


def save_model_weights(model, save_path: str) -> None:
    """
    Save only model weights (lighter than full checkpoint).
    
    Args:
        model: The HierarchicalReasoningModel instance
        save_path: Path to save weights
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model weights saved to {save_path}")


def load_model_weights(model, load_path: str, device: str = "cpu") -> None:
    """
    Load only model weights.
    
    Args:
        model: The HierarchicalReasoningModel instance
        load_path: Path to weights file
        device: Device to load model to
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Weights file not found: {load_path}")
    
    state_dict = torch.load(load_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    print(f"✅ Model weights loaded from {load_path}")


def count_parameters(model) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: The model to analyze
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def print_model_summary(model) -> None:
    """
    Print a summary of the model architecture and parameters.
    
    Args:
        model: The model to summarize
    """
    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    
    param_counts = count_parameters(model)
    
    print(f"Total parameters:       {param_counts['total']:,}")
    print(f"Trainable parameters:   {param_counts['trainable']:,}")
    print(f"Non-trainable params:   {param_counts['non_trainable']:,}")
    
    print("\nModule breakdown:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:20s}: {module_params:,} parameters")
    
    print("=" * 70 + "\n")


def evaluate_model(
    model,
    dataloader,
    criterion,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to run evaluation on
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_cycles = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            output, cycles = model(x)
            loss = criterion(output, y)
            
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_cycles.extend(cycles.cpu().tolist())
    
    avg_loss = total_loss / total_samples
    avg_cycles = sum(all_cycles) / len(all_cycles)
    
    return {
        'loss': avg_loss,
        'avg_cycles': avg_cycles,
        'min_cycles': min(all_cycles),
        'max_cycles': max(all_cycles)
    }


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: The optimizer
    
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


class MetricsTracker:
    """
    Track and log training metrics over time.
    """
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, **kwargs):
        """Add new metric values."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_latest(self, key: str) -> Optional[float]:
        """Get the most recent value for a metric."""
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1]
        return None
    
    def get_average(self, key: str, last_n: Optional[int] = None) -> Optional[float]:
        """Get average of a metric over last N steps."""
        if key not in self.metrics or not self.metrics[key]:
            return None
        
        values = self.metrics[key]
        if last_n is not None:
            values = values[-last_n:]
        
        return sum(values) / len(values)
    
    def save(self, path: str):
        """Save metrics to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✅ Metrics saved to {path}")
    
    def load(self, path: str):
        """Load metrics from JSON file."""
        with open(path, 'r') as f:
            self.metrics = json.load(f)
        print(f"✅ Metrics loaded from {path}")

