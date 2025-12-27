"""
Unit tests for the encoder module.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoder.encoder import GeometryEncoder, PrecomputedEncoder


class TestGeometryEncoder:
    """Test geometry encoder."""
    
    def test_encoder_creation_resnet18(self):
        """Test encoder creation with ResNet18."""
        encoder = GeometryEncoder(
            backbone="resnet18",
            pretrained=False,
            embedding_dim=512,
        )
        
        assert encoder is not None
        assert encoder.embedding_dim == 512
        
    def test_encoder_forward(self):
        """Test encoder forward pass."""
        encoder = GeometryEncoder(
            backbone="resnet18",
            pretrained=False,
            embedding_dim=256,
        )
        
        # Geometry mask input
        x = torch.rand(4, 1, 128, 128)
        embedding = encoder(x)
        
        assert embedding.shape == (4, 256)
        
    def test_encoder_different_resolutions(self):
        """Test encoder with different input resolutions."""
        encoder = GeometryEncoder(
            backbone="resnet18",
            pretrained=False,
            embedding_dim=512,
        )
        
        for res in [64, 128, 256]:
            x = torch.rand(2, 1, res, res)
            embedding = encoder(x)
            assert embedding.shape == (2, 512)
            
    def test_encoder_freeze(self):
        """Test freezing backbone works."""
        encoder = GeometryEncoder(
            backbone="resnet18",
            pretrained=False,
            embedding_dim=512,
            freeze_backbone=True,
        )
        
        # Check backbone parameters are frozen
        for param in encoder.backbone.parameters():
            assert not param.requires_grad
            
    def test_encoder_unfreeze(self):
        """Test unfreezing backbone works."""
        encoder = GeometryEncoder(
            backbone="resnet18",
            pretrained=False,
            embedding_dim=512,
            freeze_backbone=False,
        )
        
        # At least some parameters should require gradients
        has_trainable = any(
            p.requires_grad for p in encoder.backbone.parameters()
        )
        assert has_trainable
        
    def test_encoder_deterministic(self):
        """Test encoder is deterministic in eval mode."""
        encoder = GeometryEncoder(
            backbone="resnet18",
            pretrained=False,
            embedding_dim=512,
        )
        encoder.eval()
        
        x = torch.rand(2, 1, 64, 64)
        
        with torch.no_grad():
            emb1 = encoder(x)
            emb2 = encoder(x)
            
        assert torch.allclose(emb1, emb2)
        
    def test_encoder_output_finite(self):
        """Test encoder output is finite."""
        encoder = GeometryEncoder(
            backbone="resnet18",
            pretrained=False,
            embedding_dim=512,
        )
        encoder.eval()
        
        x = torch.rand(4, 1, 128, 128)
        
        with torch.no_grad():
            embedding = encoder(x)
            
        assert torch.isfinite(embedding).all()


class TestPrecomputedEncoder:
    """Test precomputed encoder for cached embeddings."""
    
    def test_precomputed_encoder_creation(self):
        """Test precomputed encoder initialization."""
        encoder = PrecomputedEncoder()
        
        assert encoder is not None
        
    def test_precomputed_encoder_with_embeddings(self):
        """Test precomputed encoder with pre-set embeddings."""
        encoder = PrecomputedEncoder()
        
        # Manually set embeddings
        encoder.embeddings = {
            "geom_0": np.random.randn(512).astype(np.float32),
            "geom_1": np.random.randn(512).astype(np.float32),
        }
        
        assert len(encoder.embeddings) == 2
        
    def test_precomputed_encoder_get(self):
        """Test embedding retrieval."""
        encoder = PrecomputedEncoder()
        encoder.embeddings = {
            "geom_a": np.random.randn(256).astype(np.float32),
        }
        
        emb = encoder.get_embedding("geom_a")
        
        assert emb.shape == (256,)
        
    def test_precomputed_encoder_missing_key(self):
        """Test error handling for missing keys."""
        encoder = PrecomputedEncoder()
        encoder.embeddings = {}
        
        with pytest.raises(KeyError):
            encoder.get_embedding("nonexistent")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
