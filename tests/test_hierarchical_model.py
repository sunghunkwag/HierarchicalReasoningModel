#!/usr/bin/env python3
"""
Unit tests for Hierarchical Reasoning Model

Run with: python -m pytest tests/test_hierarchical_model.py -v
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hierarchical_reasoning_model import (
    HierarchicalReasoningModel, 
    HRMConfig,
    InputNetwork,
    LowLevelModule,
    HighLevelModule,
    OutputNetwork,
    QNetwork,
    MetaRLController
)


class TestHRMConfig:
    """Test configuration class"""
    
    def test_valid_config(self):
        """Test creating valid configuration"""
        config = HRMConfig(
            hidden_low=128,
            hidden_high=256,
            T=3,
            max_cycles=5
        )
        assert config.hidden_low == 128
        assert config.hidden_high == 256
        assert config.T == 3
        assert config.max_cycles == 5
    
    def test_default_values(self):
        """Test default configuration values"""
        config = HRMConfig(
            hidden_low=128,
            hidden_high=256,
            T=3,
            max_cycles=5
        )
        assert config.use_act == True
        assert config.detach_low_to_high == True
        assert config.detach_high_to_low == False
        assert config.compute_cost_weight == 0.0


class TestComponentModules:
    """Test individual component modules"""
    
    def test_input_network(self):
        """Test InputNetwork module"""
        config = HRMConfig(hidden_low=128, hidden_high=256, T=3, max_cycles=5)
        input_net = InputNetwork(config)
        
        x = torch.randn(8, 1024)
        output = input_net(x)
        
        assert output.shape == (8, 128)
    
    def test_low_level_module(self):
        """Test LowLevelModule"""
        config = HRMConfig(hidden_low=128, hidden_high=256, T=3, max_cycles=5)
        low_level = LowLevelModule(config)
        
        z_L = torch.randn(8, 128)
        z_H = torch.randn(8, 256)
        x_tilde = torch.randn(8, 128)
        
        output = low_level(z_L, z_H, x_tilde)
        
        assert output.shape == (8, 128)
    
    def test_high_level_module(self):
        """Test HighLevelModule"""
        config = HRMConfig(hidden_low=128, hidden_high=256, T=3, max_cycles=5)
        high_level = HighLevelModule(config)
        
        z_H = torch.randn(8, 256)
        z_L_summary = torch.randn(8, 128)
        
        output = high_level(z_H, z_L_summary)
        
        assert output.shape == (8, 256)
    
    def test_output_network(self):
        """Test OutputNetwork"""
        config = HRMConfig(hidden_low=128, hidden_high=256, T=3, max_cycles=5)
        output_net = OutputNetwork(config)
        
        z_H = torch.randn(8, 256)
        output = output_net(z_H)
        
        assert output.shape == (8, 1)
    
    def test_q_network(self):
        """Test QNetwork for ACT"""
        config = HRMConfig(hidden_low=128, hidden_high=256, T=3, max_cycles=5)
        q_net = QNetwork(config)
        
        z_H = torch.randn(8, 256)
        logits = q_net(z_H)
        
        assert logits.shape == (8, 2)


class TestMetaRLController:
    """Test Meta-RL controller"""
    
    def test_reward_computation(self):
        """Test reward computation"""
        config = HRMConfig(
            hidden_low=128, 
            hidden_high=256, 
            T=3, 
            max_cycles=5,
            compute_cost_weight=0.01
        )
        controller = MetaRLController(config)
        
        loss_vec = torch.randn(8)
        cycles = torch.randint(1, 6, (8,))
        
        reward = controller.compute_reward(loss_vec, cycles, config)
        
        assert reward.shape == (8,)
        # Reward should be negative (negative loss minus cost)
        assert reward.mean() < 0
    
    def test_policy_loss(self):
        """Test policy loss computation"""
        config = HRMConfig(hidden_low=128, hidden_high=256, T=3, max_cycles=5)
        controller = MetaRLController(config)
        
        logits = torch.randn(8, 2)
        action = torch.randint(0, 2, (8,))
        advantage = torch.randn(8)
        
        loss = controller.policy_loss(logits, action, advantage)
        
        assert isinstance(loss.item(), float)
        assert not torch.isnan(loss)


class TestHierarchicalReasoningModel:
    """Test main model"""
    
    def test_model_creation(self):
        """Test model can be created"""
        config = HRMConfig(hidden_low=128, hidden_high=256, T=3, max_cycles=5)
        model = HierarchicalReasoningModel(config)
        
        assert model is not None
        assert hasattr(model, 'input_network')
        assert hasattr(model, 'low_level_module')
        assert hasattr(model, 'high_level_module')
        assert hasattr(model, 'output_network')
        assert hasattr(model, 'q_network')
    
    def test_forward_pass(self):
        """Test forward pass"""
        config = HRMConfig(hidden_low=128, hidden_high=256, T=3, max_cycles=5)
        model = HierarchicalReasoningModel(config)
        model.eval()
        
        x = torch.randn(8, 1024)
        
        with torch.no_grad():
            output, cycles = model(x)
        
        assert output.shape == (8, 1)
        assert cycles.shape == (8,)
        assert (cycles >= 1).all()
        assert (cycles <= 5).all()
    
    def test_training_step(self):
        """Test training step"""
        config = HRMConfig(
            hidden_low=128, 
            hidden_high=256, 
            T=3, 
            max_cycles=5,
            use_act=True
        )
        model = HierarchicalReasoningModel(config)
        model.train()
        
        x = torch.randn(16, 1024)
        y = torch.randn(16, 1)
        criterion = nn.MSELoss()
        
        metrics = model.training_step(x, y, criterion)
        
        assert 'loss' in metrics
        assert 'task_loss' in metrics
        assert 'policy_loss' in metrics
        assert 'baseline_loss' in metrics
        assert 'cycles_mean' in metrics
        
        assert metrics['loss'] > 0
        assert not torch.isnan(torch.tensor(metrics['loss']))
    
    def test_batch_sizes(self):
        """Test different batch sizes"""
        config = HRMConfig(hidden_low=64, hidden_high=128, T=2, max_cycles=5)
        model = HierarchicalReasoningModel(config)
        model.eval()
        
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 1024)
            with torch.no_grad():
                output, cycles = model(x)
            
            assert output.shape == (batch_size, 1)
            assert cycles.shape == (batch_size,)
    
    def test_act_enabled_vs_disabled(self):
        """Test ACT enabled vs disabled"""
        # ACT enabled
        config_act = HRMConfig(
            hidden_low=64, 
            hidden_high=128, 
            T=2, 
            max_cycles=8,
            use_act=True
        )
        model_act = HierarchicalReasoningModel(config_act)
        model_act.eval()
        
        x = torch.randn(10, 1024)
        with torch.no_grad():
            _, cycles_act = model_act(x)
        
        # ACT disabled
        config_no_act = HRMConfig(
            hidden_low=64, 
            hidden_high=128, 
            T=2, 
            max_cycles=8,
            use_act=False
        )
        model_no_act = HierarchicalReasoningModel(config_no_act)
        model_no_act.eval()
        
        with torch.no_grad():
            _, cycles_no_act = model_no_act(x)
        
        # Without ACT, all samples should use max_cycles
        assert (cycles_no_act == 8).all()
    
    def test_gradient_flow(self):
        """Test gradient flow with different detachment settings"""
        config = HRMConfig(
            hidden_low=64,
            hidden_high=128,
            T=2,
            max_cycles=3,
            detach_low_to_high=True,
            detach_high_to_low=False
        )
        model = HierarchicalReasoningModel(config)
        model.train()
        
        x = torch.randn(4, 1024, requires_grad=True)
        y = torch.randn(4, 1)
        criterion = nn.MSELoss()
        
        metrics = model.training_step(x, y, criterion)
        
        assert not torch.isnan(torch.tensor(metrics['loss']))
    
    def test_fixed_cycles(self):
        """Test forward pass with fixed number of cycles"""
        config = HRMConfig(hidden_low=64, hidden_high=128, T=2, max_cycles=10)
        model = HierarchicalReasoningModel(config)
        model.eval()
        
        x = torch.randn(8, 1024)
        
        for num_cycles in [1, 3, 5]:
            with torch.no_grad():
                output, cycles = model(x, num_cycles=num_cycles)
            
            assert (cycles == num_cycles).all()


class TestDatasetCreation:
    """Test dataset creation from examples"""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset creation doesn't crash"""
        from torch.utils.data import TensorDataset
        
        num_samples = 100
        input_dim = 1024
        
        X = torch.randn(num_samples, input_dim)
        
        easy_mask = torch.randperm(num_samples)[:num_samples//3]
        medium_mask = torch.randperm(num_samples)[num_samples//3:2*num_samples//3]
        hard_mask = torch.randperm(num_samples)[2*num_samples//3:]
        
        Y = torch.zeros(num_samples, 1)
        
        # Easy targets
        Y[easy_mask] = X[easy_mask, :10].sum(dim=1, keepdim=True) * 0.1
        
        # Medium targets
        Y[medium_mask] = (X[medium_mask, :50].pow(2).sum(dim=1, keepdim=True) - 
                          X[medium_mask, 50:100].sum(dim=1, keepdim=True)) * 0.01
        
        # Hard targets - FIXED VERSION
        Y[hard_mask] = (torch.sin(X[hard_mask, :20].sum(dim=1, keepdim=True)) * 
                        torch.cos(X[hard_mask, 20:40].sum(dim=1, keepdim=True)) + 
                        X[hard_mask, 40:60].prod(dim=1).clamp(-1, 1).unsqueeze(1))
        
        dataset = TensorDataset(X, Y)
        
        assert len(dataset) == num_samples
        assert dataset[0][0].shape == (input_dim,)
        assert dataset[0][1].shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

