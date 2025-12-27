"""
Dataset Generation Script

Generates training data by:
1. Creating diverse geometry masks
2. Solving heat equation with PINN for each geometry
3. Pre-computing geometry embeddings
4. Storing everything in efficient HDF5 format
"""

import argparse
from pathlib import Path
import yaml
import numpy as np
import h5py
from tqdm import tqdm
import torch
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry import GeometryGenerator, GeometryConfig
from src.pinn import PINNSolver, PINNConfig
from src.encoder import GeometryEncoder, PrecomputedEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Generate thermal simulation dataset")
    parser.add_argument(
        "--config", 
        type=Path, 
        default=Path("configs/dataset.yaml"),
        help="Path to dataset config"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory"
    )
    parser.add_argument(
        "--num-physics-variations",
        type=int,
        default=5,
        help="Number of physics parameter variations per geometry"
    )
    parser.add_argument(
        "--skip-pinn",
        action="store_true",
        help="Skip PINN solving (for testing geometry/embedding generation)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_cfg = config['dataset']
    pinn_cfg = config.get('pinn', {})
    
    # Create geometry configuration
    geo_config = GeometryConfig(
        resolution=dataset_cfg['geometry']['resolution'],
        shapes=dataset_cfg['geometry']['shapes'],
    )
    
    # Generate geometries for each split
    print("=" * 60)
    print("GENERATING GEOMETRIES")
    print("=" * 60)
    
    train_samples, val_samples, test_samples = GeometryGenerator.create_splits(
        train_seed=dataset_cfg['geometry']['seed_train'],
        val_seed=dataset_cfg['geometry']['seed_val'],
        test_seed=dataset_cfg['geometry']['seed_test'],
        num_train=dataset_cfg['geometry']['num_train'],
        num_val=dataset_cfg['geometry']['num_val'],
        num_test=dataset_cfg['geometry']['num_test'],
        config=geo_config,
    )
    
    print(f"Generated {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test geometries")
    
    # Save geometry masks
    for split_name, samples in [
        ('train', train_samples),
        ('val', val_samples),
        ('test', test_samples)
    ]:
        gen = GeometryGenerator(geo_config)
        gen.save_samples(samples, output_dir / "geometries", prefix=split_name)
    
    # Pre-compute geometry embeddings
    print("\n" + "=" * 60)
    print("PRE-COMPUTING GEOMETRY EMBEDDINGS")
    print("=" * 60)
    
    encoder = GeometryEncoder(
        backbone="resnet18",
        pretrained=True,
        embedding_dim=512,
        freeze_backbone=True,
    )
    
    precomputed = PrecomputedEncoder(
        encoder=encoder,
        cache_path=output_dir / "embeddings.h5",
    )
    
    # Combine all samples for embedding
    all_samples = train_samples + val_samples + test_samples
    precomputed.precompute(
        all_samples,
        batch_size=32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save=True,
    )
    
    if args.skip_pinn:
        print("\nSkipping PINN solving (--skip-pinn flag set)")
        print("Done!")
        return
    
    # Generate temperature fields using PINN
    print("\n" + "=" * 60)
    print("GENERATING TEMPERATURE FIELDS WITH PINN")
    print("=" * 60)
    
    physics_cfg = dataset_cfg['physics']
    time_cfg = physics_cfg['time']
    
    for split_name, samples in [
        ('train', train_samples),
        ('val', val_samples),
        ('test', test_samples)
    ]:
        print(f"\n--- Processing {split_name} split ---")
        
        num_geometries = len(samples)
        num_variations = args.num_physics_variations
        num_samples = num_geometries * num_variations
        
        time_steps = time_cfg['num_steps']
        resolution = dataset_cfg['geometry']['resolution']
        
        # Initialize storage arrays
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
        
        sample_idx = 0
        
        for geo_sample in tqdm(samples, desc=f"Processing {split_name}"):
            mask = geo_sample['mask']
            geo_hash = geo_sample['hash']
            
            # Find valid positions for heat source
            inside_coords = np.where(mask > 0.5)
            
            if len(inside_coords[0]) == 0:
                print(f"Warning: Empty geometry {geo_hash}, skipping")
                continue
            
            for var_idx in range(num_variations):
                # Sample random physics parameters
                alpha = np.random.uniform(
                    physics_cfg['thermal_diffusivity']['min'],
                    physics_cfg['thermal_diffusivity']['max']
                )
                
                # Random source position within geometry
                pos_idx = np.random.randint(len(inside_coords[0]))
                source_y = inside_coords[0][pos_idx] / (resolution - 1)
                source_x = inside_coords[1][pos_idx] / (resolution - 1)
                intensity = np.random.uniform(
                    physics_cfg['heat_source']['intensity']['min'],
                    physics_cfg['heat_source']['intensity']['max']
                )
                
                # Create PINN solver
                pinn_config = PINNConfig(
                    thermal_diffusivity=alpha,
                    hidden_layers=pinn_cfg.get('architecture', {}).get('hidden_layers', [128, 128, 128, 128]),
                    activation=pinn_cfg.get('architecture', {}).get('activation', 'tanh'),
                    max_iterations=pinn_cfg.get('training', {}).get('max_iterations', 5000),
                    num_collocation=pinn_cfg.get('training', {}).get('num_collocation', 10000),
                    num_boundary=pinn_cfg.get('training', {}).get('num_boundary', 2000),
                    num_initial=pinn_cfg.get('training', {}).get('num_initial', 2000),
                    weight_pde=pinn_cfg.get('training', {}).get('weights', {}).get('pde', 1.0),
                    weight_bc=pinn_cfg.get('training', {}).get('weights', {}).get('boundary', 10.0),
                    weight_ic=pinn_cfg.get('training', {}).get('weights', {}).get('initial', 10.0),
                    t_min=time_cfg['t_min'],
                    t_max=time_cfg['t_max'],
                    device=pinn_cfg.get('device', 'cuda'),
                )
                
                solver = PINNSolver(pinn_config)
                
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
        
        # Trim arrays to actual size (in case some were skipped)
        temperature_fields = temperature_fields[:sample_idx]
        physics_params = physics_params[:sample_idx]
        geometry_masks = geometry_masks[:sample_idx]
        
        # Save to HDF5
        output_path = output_dir / f"{split_name}.h5"
        
        print(f"Saving {sample_idx} samples to {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            f.create_dataset(
                'temperature_fields',
                data=temperature_fields,
                compression='gzip',
                chunks=(1, time_steps, resolution, resolution),
            )
            f.create_dataset('physics_params', data=physics_params)
            f.create_dataset(
                'geometry_hashes',
                data=np.array(geometry_hashes, dtype='S16')
            )
            f.create_dataset(
                'geometry_masks',
                data=geometry_masks,
                compression='gzip',
            )
    
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
