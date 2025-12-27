"""
Geometry Generator - Creates diverse 2D geometry datasets.
Ensures no geometry leakage between train/val/test splits.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import hashlib
import json

from .shapes import (
    create_circle,
    create_rectangle,
    create_ellipse,
    create_polygon,
    create_l_shape,
    create_t_shape,
    create_complex_shape,
)


@dataclass
class GeometryConfig:
    """Configuration for geometry generation."""
    resolution: int = 128
    shapes: List[str] = None
    
    def __post_init__(self):
        if self.shapes is None:
            self.shapes = [
                "circle", "rectangle", "ellipse", 
                "polygon", "l_shape", "t_shape", "complex"
            ]


class GeometryGenerator:
    """
    Generates diverse 2D geometry masks for thermal simulation training.
    
    Key features:
    - No geometry leakage between splits (separate seeds)
    - Reproducible generation
    - Diverse shape types
    - Pre-computes geometry embeddings
    """
    
    SHAPE_FUNCTIONS: Dict[str, Callable] = {
        "circle": create_circle,
        "rectangle": create_rectangle,
        "ellipse": create_ellipse,
        "polygon": create_polygon,
        "l_shape": create_l_shape,
        "t_shape": create_t_shape,
        "complex": create_complex_shape,
    }
    
    def __init__(
        self,
        config: GeometryConfig,
        seed: int = 42,
    ):
        """
        Initialize the geometry generator.
        
        Args:
            config: Geometry configuration
            seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Validate shapes
        for shape in config.shapes:
            if shape not in self.SHAPE_FUNCTIONS:
                raise ValueError(f"Unknown shape type: {shape}")
    
    def generate(
        self,
        num_samples: int,
        shape_weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        """
        Generate a batch of geometry samples.
        
        Args:
            num_samples: Number of geometries to generate
            shape_weights: Optional weights for shape selection
            
        Returns:
            List of dictionaries containing:
                - 'mask': Binary geometry mask (H, W)
                - 'shape_type': Name of the shape
                - 'hash': Unique hash for the geometry
                - 'params': Generation parameters
        """
        if shape_weights is None:
            shape_weights = {s: 1.0 for s in self.config.shapes}
        
        # Normalize weights
        total = sum(shape_weights.values())
        probs = [shape_weights.get(s, 0) / total for s in self.config.shapes]
        
        samples = []
        
        for i in range(num_samples):
            # Select shape type
            shape_type = self.rng.choice(self.config.shapes, p=probs)
            
            # Generate shape
            shape_func = self.SHAPE_FUNCTIONS[shape_type]
            mask = shape_func(self.config.resolution, rng=self.rng)
            
            # Compute unique hash for this geometry
            geo_hash = self._compute_hash(mask)
            
            samples.append({
                'mask': mask,
                'shape_type': shape_type,
                'hash': geo_hash,
                'index': i,
            })
        
        return samples
    
    def _compute_hash(self, mask: np.ndarray) -> str:
        """Compute a unique hash for a geometry mask."""
        # Quantize to reduce sensitivity to floating point
        quantized = (mask > 0.5).astype(np.uint8)
        return hashlib.md5(quantized.tobytes()).hexdigest()[:16]
    
    @staticmethod
    def create_splits(
        train_seed: int,
        val_seed: int,
        test_seed: int,
        num_train: int,
        num_val: int,
        num_test: int,
        config: GeometryConfig,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Create train/val/test splits with no geometry leakage.
        
        Uses completely different seeds for each split to ensure
        no geometry can appear in multiple splits.
        
        Args:
            train_seed: Seed for training set
            val_seed: Seed for validation set  
            test_seed: Seed for test set
            num_train: Number of training samples
            num_val: Number of validation samples
            num_test: Number of test samples
            config: Geometry configuration
            
        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        train_gen = GeometryGenerator(config, seed=train_seed)
        val_gen = GeometryGenerator(config, seed=val_seed)
        test_gen = GeometryGenerator(config, seed=test_seed)
        
        train_samples = train_gen.generate(num_train)
        val_samples = val_gen.generate(num_val)
        test_samples = test_gen.generate(num_test)
        
        # Verify no leakage (for debugging)
        train_hashes = set(s['hash'] for s in train_samples)
        val_hashes = set(s['hash'] for s in val_samples)
        test_hashes = set(s['hash'] for s in test_samples)
        
        assert len(train_hashes & val_hashes) == 0, "Geometry leakage: train-val"
        assert len(train_hashes & test_hashes) == 0, "Geometry leakage: train-test"
        assert len(val_hashes & test_hashes) == 0, "Geometry leakage: val-test"
        
        return train_samples, val_samples, test_samples
    
    def save_samples(
        self,
        samples: List[Dict],
        output_dir: Path,
        prefix: str = "geometry",
    ) -> None:
        """
        Save generated samples to disk.
        
        Args:
            samples: List of geometry samples
            output_dir: Directory to save to
            prefix: Filename prefix
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save masks as numpy arrays
        masks = np.stack([s['mask'] for s in samples])
        np.save(output_dir / f"{prefix}_masks.npy", masks)
        
        # Save metadata
        metadata = [
            {'shape_type': s['shape_type'], 'hash': s['hash'], 'index': s['index']}
            for s in samples
        ]
        with open(output_dir / f"{prefix}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @staticmethod
    def load_samples(input_dir: Path, prefix: str = "geometry") -> List[Dict]:
        """
        Load samples from disk.
        
        Args:
            input_dir: Directory containing saved samples
            prefix: Filename prefix
            
        Returns:
            List of geometry samples
        """
        input_dir = Path(input_dir)
        
        masks = np.load(input_dir / f"{prefix}_masks.npy")
        
        with open(input_dir / f"{prefix}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        samples = []
        for i, meta in enumerate(metadata):
            samples.append({
                'mask': masks[i],
                **meta
            })
        
        return samples
