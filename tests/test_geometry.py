"""
Unit tests for the geometry module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry.shapes import (
    create_circle,
    create_rectangle,
    create_ellipse,
    create_polygon,
    create_l_shape,
    create_t_shape,
    create_complex_shape,
)
from src.geometry.generator import GeometryGenerator, GeometryConfig


class TestShapes:
    """Test individual shape creation functions."""
    
    def test_create_circle(self):
        """Test circle creation."""
        mask = create_circle(resolution=64, center=(0.5, 0.5), radius=0.3)
        
        assert mask.shape == (64, 64)
        assert mask.dtype == np.float32 or mask.dtype == np.float64
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0
        assert mask.sum() > 0
        
    def test_create_circle_different_positions(self):
        """Test circle at different positions."""
        mask_center = create_circle(64, center=(0.5, 0.5), radius=0.2)
        mask_corner = create_circle(64, center=(0.25, 0.25), radius=0.15)
        
        assert mask_center.sum() > 0
        assert mask_corner.sum() > 0
        assert not np.allclose(mask_center, mask_corner)
        
    def test_create_rectangle(self):
        """Test rectangle creation."""
        mask = create_rectangle(
            resolution=64,
            corner=(0.2, 0.2),
            size=(0.4, 0.3),
        )
        
        assert mask.shape == (64, 64)
        assert mask.sum() > 0
        
    def test_create_ellipse(self):
        """Test ellipse creation."""
        mask = create_ellipse(
            resolution=64,
            center=(0.5, 0.5),
            axes=(0.3, 0.2),
        )
        
        assert mask.shape == (64, 64)
        assert mask.sum() > 0
        
    def test_create_polygon(self):
        """Test polygon creation."""
        vertices = [(0.3, 0.3), (0.7, 0.3), (0.5, 0.7)]
        mask = create_polygon(resolution=64, vertices=vertices)
        
        assert mask.shape == (64, 64)
        assert mask.sum() > 0
        
    def test_create_l_shape(self):
        """Test L-shape creation."""
        mask = create_l_shape(resolution=64)
        
        assert mask.shape == (64, 64)
        assert mask.sum() > 0
        
    def test_create_t_shape(self):
        """Test T-shape creation."""
        mask = create_t_shape(resolution=64)
        
        assert mask.shape == (64, 64)
        assert mask.sum() > 0
        
    def test_create_complex_shape(self):
        """Test complex shape creation."""
        rng = np.random.default_rng(42)
        mask = create_complex_shape(resolution=64, rng=rng)
        
        assert mask.shape == (64, 64)
        assert mask.sum() > 0
        
    def test_resolution_scaling(self):
        """Test shapes at different resolutions."""
        for res in [32, 64, 128]:
            mask = create_circle(resolution=res, center=(0.5, 0.5), radius=0.3)
            assert mask.shape == (res, res)


class TestGeometryGenerator:
    """Test the geometry generator."""
    
    def test_generator_creation(self):
        """Test generator initialization."""
        config = GeometryConfig(resolution=64)
        gen = GeometryGenerator(config, seed=42)
        
        assert gen.config.resolution == 64
        assert gen.seed == 42
        
    def test_generate_samples(self):
        """Test sample generation."""
        config = GeometryConfig(resolution=64, shapes=["circle", "rectangle"])
        gen = GeometryGenerator(config, seed=42)
        
        samples = gen.generate(num_samples=5)
        
        assert len(samples) == 5
        for sample in samples:
            assert "mask" in sample
            assert "shape_type" in sample
            assert "hash" in sample
            assert sample["mask"].shape == (64, 64)
            
    def test_reproducibility(self):
        """Test same seed gives same results."""
        config = GeometryConfig(resolution=64)
        
        gen1 = GeometryGenerator(config, seed=42)
        samples1 = gen1.generate(num_samples=3)
        
        gen2 = GeometryGenerator(config, seed=42)
        samples2 = gen2.generate(num_samples=3)
        
        for s1, s2 in zip(samples1, samples2):
            assert np.allclose(s1["mask"], s2["mask"])
            assert s1["hash"] == s2["hash"]
            
    def test_different_seeds(self):
        """Test different seeds give different geometries."""
        config = GeometryConfig(resolution=64, shapes=["circle"])
        
        gen1 = GeometryGenerator(config, seed=42)
        samples1 = gen1.generate(num_samples=3)
        
        gen2 = GeometryGenerator(config, seed=123)
        samples2 = gen2.generate(num_samples=3)
        
        all_same = all(
            np.allclose(s1["mask"], s2["mask"]) 
            for s1, s2 in zip(samples1, samples2)
        )
        assert not all_same
        
    def test_unique_hashes(self):
        """Test generated geometries have unique hashes."""
        config = GeometryConfig(resolution=64)
        gen = GeometryGenerator(config, seed=42)
        
        samples = gen.generate(num_samples=20)
        hashes = [s["hash"] for s in samples]
        
        assert len(set(hashes)) == len(hashes)


class TestNoGeometryLeakage:
    """Test no geometry leakage between splits."""
    
    def test_different_seeds_no_overlap(self):
        """Test different seeds produce non-overlapping geometries."""
        config = GeometryConfig(resolution=64, shapes=["circle", "rectangle"])
        
        gen_train = GeometryGenerator(config, seed=42)
        gen_val = GeometryGenerator(config, seed=1234)
        gen_test = GeometryGenerator(config, seed=5678)
        
        train_samples = gen_train.generate(num_samples=10)
        val_samples = gen_val.generate(num_samples=5)
        test_samples = gen_test.generate(num_samples=5)
        
        train_hashes = set(s["hash"] for s in train_samples)
        val_hashes = set(s["hash"] for s in val_samples)
        test_hashes = set(s["hash"] for s in test_samples)
        
        assert len(train_hashes & val_hashes) == 0
        assert len(train_hashes & test_hashes) == 0
        assert len(val_hashes & test_hashes) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
