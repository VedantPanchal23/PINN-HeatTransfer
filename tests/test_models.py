"""
Unit tests for neural operator models.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.fno import FNO2d, SpectralConv2d, FNOBlock


class TestSpectralConv2d:
    """Test spectral convolution layer."""
    
    def test_spectral_conv_creation(self):
        """Test spectral conv initialization."""
        layer = SpectralConv2d(
            in_channels=16,
            out_channels=16,
            modes1=8,
            modes2=8,
        )
        
        assert layer.in_channels == 16
        assert layer.out_channels == 16
        assert layer.modes1 == 8
        assert layer.modes2 == 8
        
    def test_spectral_conv_forward(self):
        """Test spectral conv forward pass."""
        layer = SpectralConv2d(16, 16, 8, 8)
        x = torch.randn(2, 16, 32, 32)
        
        y = layer(x)
        
        assert y.shape == (2, 16, 32, 32)
        
    def test_spectral_conv_different_channels(self):
        """Test with different in/out channels."""
        layer = SpectralConv2d(16, 32, 8, 8)
        x = torch.randn(2, 16, 32, 32)
        
        y = layer(x)
        
        assert y.shape == (2, 32, 32, 32)


class TestFNOBlock:
    """Test FNO block."""
    
    def test_fno_block_creation(self):
        """Test FNO block initialization."""
        block = FNOBlock(width=32, modes1=8, modes2=8)
        
        assert block is not None
        
    def test_fno_block_forward(self):
        """Test FNO block forward pass."""
        block = FNOBlock(width=32, modes1=8, modes2=8)
        x = torch.randn(2, 32, 32, 32)
        
        y = block(x)
        
        assert y.shape == x.shape


class TestFNO2d:
    """Test the main FNO2d model."""
    
    def test_fno_creation(self):
        """Test FNO2d initialization with actual API."""
        model = FNO2d(
            modes=8,
            width=32,
            num_layers=4,
            geometry_dim=512,
            physics_dim=4,
        )
        
        assert model is not None
        
    def test_fno_parameter_count(self):
        """Test FNO has reasonable parameter count."""
        model = FNO2d(
            modes=8,
            width=32,
            num_layers=4,
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        
        assert num_params > 0
        assert 1000 < num_params < 100_000_000


class TestModelDevice:
    """Test model device handling."""
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_fno_cuda(self):
        """Test FNO on CUDA."""
        model = FNO2d(modes=8, width=32, num_layers=2).cuda()
        
        assert next(model.parameters()).device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
