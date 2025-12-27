"""
Dataset classes for thermal surrogate model training.

Key design principles:
1. Geometry embeddings are pre-computed and loaded (no re-encoding)
2. Train/val/test splits have completely separate geometries
3. Efficient HDF5 storage for large datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import json


class ThermalDataset(Dataset):
    """
    Dataset for thermal surrogate model training.
    
    Each sample contains:
    - Pre-computed geometry embedding
    - Physics parameters (alpha, source_x, source_y, intensity)
    - Ground truth temperature field T(t, x, y)
    - Geometry mask (for visualization/masking)
    """
    
    def __init__(
        self,
        data_path: Path,
        embeddings_path: Path,
        transform: Optional[callable] = None,
        load_masks: bool = False,
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to HDF5 file with temperature fields
            embeddings_path: Path to pre-computed embeddings
            transform: Optional data transform
            load_masks: Whether to load geometry masks
        """
        self.data_path = Path(data_path)
        self.embeddings_path = Path(embeddings_path)
        self.transform = transform
        self.load_masks = load_masks
        
        # Load metadata
        with h5py.File(self.data_path, 'r') as f:
            self.num_samples = f['temperature_fields'].shape[0]
            self.time_steps = f['temperature_fields'].shape[1]
            self.resolution = f['temperature_fields'].shape[2]
            
            # Load geometry hashes for embedding lookup
            self.geometry_hashes = [
                h.decode() for h in f['geometry_hashes'][:]
            ]
            
            # Load physics parameters
            self.physics_params = f['physics_params'][:]
        
        # Load pre-computed embeddings
        self._load_embeddings()
        
        print(f"Loaded dataset with {self.num_samples} samples")
        print(f"  Time steps: {self.time_steps}")
        print(f"  Resolution: {self.resolution}x{self.resolution}")
    
    def _load_embeddings(self):
        """Load pre-computed geometry embeddings."""
        with h5py.File(self.embeddings_path, 'r') as f:
            hashes = [h.decode() for h in f['hashes'][:]]
            embeddings = f['embeddings'][:]
            
            self.embedding_lookup = {
                h: embeddings[i] for i, h in enumerate(hashes)
            }
        
        # Verify all required embeddings exist
        for h in self.geometry_hashes:
            if h not in self.embedding_lookup:
                raise ValueError(f"Missing embedding for geometry: {h}")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - 'embedding': Geometry embedding [embedding_dim]
                - 'physics': Physics parameters [physics_dim]
                - 'temperature': Temperature field [T, H, W]
                - 'mask': Geometry mask [H, W] (optional)
        """
        # Load temperature field from HDF5
        with h5py.File(self.data_path, 'r') as f:
            temperature = f['temperature_fields'][idx]
            
            if self.load_masks:
                mask = f['geometry_masks'][idx]
        
        # Get pre-computed embedding
        geo_hash = self.geometry_hashes[idx]
        embedding = self.embedding_lookup[geo_hash]
        
        # Get physics parameters
        physics = self.physics_params[idx]
        
        # Create sample dictionary
        sample = {
            'embedding': torch.tensor(embedding, dtype=torch.float32),
            'physics': torch.tensor(physics, dtype=torch.float32),
            'temperature': torch.tensor(temperature, dtype=torch.float32),
            'geometry_hash': geo_hash,
        }
        
        if self.load_masks:
            sample['mask'] = torch.tensor(mask, dtype=torch.float32)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class ThermalDatasetInMemory(Dataset):
    """
    In-memory dataset for faster training on smaller datasets.
    
    Loads all data into RAM/GPU memory for maximum throughput.
    """
    
    def __init__(
        self,
        data_path: Path,
        embeddings_path: Path,
        device: str = "cpu",
    ):
        """
        Initialize in-memory dataset.
        
        Args:
            data_path: Path to HDF5 data file
            embeddings_path: Path to embeddings file
            device: Device to store tensors on
        """
        self.device = torch.device(device)
        
        # Load all data into memory
        print(f"Loading dataset into memory...")
        
        with h5py.File(data_path, 'r') as f:
            self.temperature_fields = torch.tensor(
                f['temperature_fields'][:], 
                dtype=torch.float32,
                device=self.device
            )
            self.physics_params = torch.tensor(
                f['physics_params'][:],
                dtype=torch.float32,
                device=self.device
            )
            self.geometry_hashes = [h.decode() for h in f['geometry_hashes'][:]]
        
        # Load embeddings
        with h5py.File(embeddings_path, 'r') as f:
            hashes = [h.decode() for h in f['hashes'][:]]
            embeddings_np = f['embeddings'][:]
            embedding_lookup = {h: embeddings_np[i] for i, h in enumerate(hashes)}
        
        # Create embedding tensor in order
        embeddings = np.stack([
            embedding_lookup[h] for h in self.geometry_hashes
        ])
        self.embeddings = torch.tensor(
            embeddings, 
            dtype=torch.float32,
            device=self.device
        )
        
        self.num_samples = len(self.temperature_fields)
        print(f"Loaded {self.num_samples} samples into memory")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'embedding': self.embeddings[idx],
            'physics': self.physics_params[idx],
            'temperature': self.temperature_fields[idx],
        }


def create_dataloaders(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    embeddings_path: Path,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    in_memory: bool = False,
    device: str = "cpu",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Path to test data
        embeddings_path: Path to embeddings (shared across splits)
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        in_memory: Whether to load data into memory
        device: Device for in-memory loading
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    if in_memory:
        train_dataset = ThermalDatasetInMemory(train_path, embeddings_path, device)
        val_dataset = ThermalDatasetInMemory(val_path, embeddings_path, device)
        test_dataset = ThermalDatasetInMemory(test_path, embeddings_path, device)
        
        # In-memory datasets don't need workers
        num_workers = 0
        pin_memory = False
    else:
        train_dataset = ThermalDataset(train_path, embeddings_path)
        val_dataset = ThermalDataset(val_path, embeddings_path)
        test_dataset = ThermalDataset(test_path, embeddings_path)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader, test_loader


class DataGenerator:
    """
    Generates training data by solving heat equation with PINN for each geometry.
    """
    
    def __init__(
        self,
        output_dir: Path,
        pinn_config: dict,
        geometry_config: dict,
    ):
        """
        Initialize data generator.
        
        Args:
            output_dir: Directory to save generated data
            pinn_config: Configuration for PINN solver
            geometry_config: Configuration for geometry generation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.pinn_config = pinn_config
        self.geometry_config = geometry_config
    
    def generate(
        self,
        geometry_samples: List[Dict],
        split_name: str = "train",
        num_physics_variations: int = 5,
    ) -> Path:
        """
        Generate temperature field data for given geometries.
        
        Args:
            geometry_samples: List of geometry samples with masks
            split_name: Name of the split (train/val/test)
            num_physics_variations: Number of physics parameter variations per geometry
            
        Returns:
            Path to generated HDF5 file
        """
        from ..pinn.solver import PINNSolver, PINNConfig
        
        num_geometries = len(geometry_samples)
        num_samples = num_geometries * num_physics_variations
        
        # Initialize arrays
        resolution = self.geometry_config.get('resolution', 128)
        time_steps = self.pinn_config.get('time', {}).get('num_steps', 50)
        
        temperature_fields = np.zeros(
            (num_samples, time_steps, resolution, resolution),
            dtype=np.float32
        )
        physics_params = np.zeros((num_samples, 4), dtype=np.float32)
        geometry_hashes = []
        geometry_masks = np.zeros(
            (num_samples, resolution, resolution),
            dtype=np.float32
        )
        
        # Generate data
        sample_idx = 0
        
        for geo_sample in geometry_samples:
            mask = geo_sample['mask']
            geo_hash = geo_sample['hash']
            
            # Find valid positions for heat source (inside geometry)
            inside_coords = np.where(mask > 0.5)
            
            for var_idx in range(num_physics_variations):
                # Sample random physics parameters
                alpha = np.random.uniform(0.01, 0.5)
                
                # Random source position within geometry
                pos_idx = np.random.randint(len(inside_coords[0]))
                source_y = inside_coords[0][pos_idx] / (resolution - 1)
                source_x = inside_coords[1][pos_idx] / (resolution - 1)
                intensity = np.random.uniform(1.0, 10.0)
                
                # Create PINN solver
                pinn_cfg = PINNConfig(
                    thermal_diffusivity=alpha,
                    max_iterations=self.pinn_config.get('max_iterations', 5000),
                    **{k: v for k, v in self.pinn_config.items() 
                       if k not in ['max_iterations', 'thermal_diffusivity']}
                )
                
                solver = PINNSolver(pinn_cfg)
                
                # Create heat source
                heat_source = solver.create_heat_source(
                    source_x, source_y, intensity
                )
                
                # Train PINN
                solver.train(
                    geometry_mask=mask,
                    heat_source=heat_source,
                    verbose=False,
                )
                
                # Generate temperature field
                temp_field = solver.generate_temperature_field(
                    num_time_steps=time_steps,
                    spatial_resolution=resolution,
                    geometry_mask=mask,
                )
                
                # Store results
                temperature_fields[sample_idx] = temp_field
                physics_params[sample_idx] = [alpha, source_x, source_y, intensity]
                geometry_hashes.append(geo_hash)
                geometry_masks[sample_idx] = mask
                
                sample_idx += 1
                
                if sample_idx % 10 == 0:
                    print(f"Generated {sample_idx}/{num_samples} samples")
        
        # Save to HDF5
        output_path = self.output_dir / f"{split_name}.h5"
        
        with h5py.File(output_path, 'w') as f:
            f.create_dataset(
                'temperature_fields', 
                data=temperature_fields,
                compression='gzip'
            )
            f.create_dataset(
                'physics_params',
                data=physics_params
            )
            f.create_dataset(
                'geometry_hashes',
                data=np.array(geometry_hashes, dtype='S16')
            )
            f.create_dataset(
                'geometry_masks',
                data=geometry_masks,
                compression='gzip'
            )
        
        print(f"Saved {num_samples} samples to {output_path}")
        return output_path
