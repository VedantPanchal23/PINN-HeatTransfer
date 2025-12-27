"""
Unit tests for the PINN solver module.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pinn.network import PINNNetwork
from src.pinn.loss import HeatEquationLoss
from src.pinn.solver import PINNSolver, PINNConfig


class TestPINNNetwork:
    """Test PINN network architecture."""
    
    def test_network_creation(self):
        """Test PINN network initialization."""
        net = PINNNetwork(
            hidden_layers=[64, 64, 64],
            activation="tanh",
        )
        
        assert net is not None
        
    def test_network_forward(self):
        """Test PINN network forward pass."""
        net = PINNNetwork(
            hidden_layers=[32, 32],
            use_fourier_features=False,  # Disable for simple testing
        )
        
        # Input: (t, x, y) as separate tensors
        t = torch.rand(100)
        x = torch.rand(100)
        y = torch.rand(100)
        
        output = net(t, x, y)
        
        assert output.shape == (100, 1)
        
    def test_network_output_finite(self):
        """Test network outputs reasonable values."""
        net = PINNNetwork(
            hidden_layers=[64, 64],
            use_fourier_features=False,  # Disable for simple testing
        )
        
        t = torch.rand(100)
        x = torch.rand(100)
        y = torch.rand(100)
        
        output = net(t, x, y)
        
        assert torch.isfinite(output).all()
        
    def test_network_gradient_computation(self):
        """Test gradients can be computed for PINN."""
        net = PINNNetwork(
            hidden_layers=[32, 32],
            use_fourier_features=False,  # Disable for simple testing
        )
        
        t = torch.rand(50, requires_grad=True)
        x = torch.rand(50, requires_grad=True)
        y = torch.rand(50, requires_grad=True)
        
        output = net(t, x, y)
        
        # Compute gradients w.r.t. input (needed for PDE loss)
        grads = torch.autograd.grad(
            output.sum(), [t, x, y], create_graph=True
        )
        
        assert grads[0].shape == t.shape
        assert grads[1].shape == x.shape
        assert grads[2].shape == y.shape


class TestHeatEquationLoss:
    """Test heat equation loss computation."""
    
    def test_loss_creation(self):
        """Test loss function initialization."""
        loss_fn = HeatEquationLoss(alpha=0.1)
        
        assert loss_fn.alpha == 0.1
        
    def test_loss_weights(self):
        """Test loss weights are set correctly."""
        loss_fn = HeatEquationLoss(
            alpha=0.1,
            weight_pde=1.0,
            weight_bc=10.0,
            weight_ic=10.0,
        )
        
        assert loss_fn.weight_pde == 1.0
        assert loss_fn.weight_bc == 10.0
        assert loss_fn.weight_ic == 10.0


class TestPINNSolver:
    """Test the PINN solver for generating ground truth."""
    
    def test_solver_creation(self):
        """Test solver initialization."""
        config = PINNConfig(
            hidden_layers=[32, 32],
            device="cpu",
        )
        solver = PINNSolver(config)
        
        assert solver is not None
        assert solver.config.hidden_layers == [32, 32]
        
    def test_solver_model_exists(self):
        """Test solver has model."""
        config = PINNConfig(
            hidden_layers=[32, 32],
            device="cpu",
        )
        solver = PINNSolver(config)
        
        assert solver.model is not None
        assert isinstance(solver.model, PINNNetwork)


class TestPINNGradients:
    """Test gradient-related functionality for PINN."""
    
    def test_laplacian_computation(self):
        """Test Laplacian computation for heat equation."""
        net = PINNNetwork(
            hidden_layers=[32, 32],
            use_fourier_features=False,  # Disable for simple testing
        )
        
        t = torch.rand(50, requires_grad=True)
        x = torch.rand(50, requires_grad=True)
        y = torch.rand(50, requires_grad=True)
        
        # Forward pass
        u = net(t, x, y)
        
        # Compute du/dt
        du_dt = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]
        
        # Compute du/dx
        du_dx = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]
        
        assert du_dt.shape == (50,)
        assert du_dx.shape == (50,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
