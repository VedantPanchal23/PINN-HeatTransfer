"""
Inference Script for Thermal Surrogate Model

Runs inference on new geometry images and generates temperature field predictions.
"""

import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
from PIL import Image
import json
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import ThermalPredictor
from src.visualization import plot_temperature_field, create_animation, compare_predictions
from src.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run thermal surrogate inference")
    parser.add_argument(
        "--geometry",
        type=Path,
        required=True,
        help="Path to geometry image (PNG)"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--encoder",
        type=Path,
        default=None,
        help="Path to geometry encoder checkpoint"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/inference.yaml"),
        help="Path to inference config"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inference"),
        help="Output directory"
    )
    
    # Physics parameters
    parser.add_argument("--alpha", type=float, default=0.1, help="Thermal diffusivity")
    parser.add_argument("--source-x", type=float, default=0.5, help="Heat source X position")
    parser.add_argument("--source-y", type=float, default=0.5, help="Heat source Y position")
    parser.add_argument("--intensity", type=float, default=5.0, help="Heat source intensity")
    
    # Output options
    parser.add_argument("--resolution", type=int, default=128, help="Output resolution")
    parser.add_argument("--save-animation", action="store_true", help="Save animation GIF")
    parser.add_argument("--benchmark", action="store_true", help="Run inference benchmark")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    if args.config.exists():
        config = load_config(args.config)
    else:
        config = {}
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("THERMAL SURROGATE INFERENCE")
    print("=" * 60)
    
    # Initialize predictor
    print(f"\nLoading model from {args.checkpoint}")
    
    predictor = ThermalPredictor(
        model_path=args.checkpoint,
        encoder_path=args.encoder,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compile_model=True,
    )
    
    # Load geometry
    print(f"Loading geometry from {args.geometry}")
    
    geometry_img = Image.open(args.geometry).convert('L')
    geometry_mask = np.array(geometry_img).astype(np.float32) / 255.0
    
    # Resize if needed
    if geometry_mask.shape != (args.resolution, args.resolution):
        import cv2
        geometry_mask = cv2.resize(
            geometry_mask, 
            (args.resolution, args.resolution),
            interpolation=cv2.INTER_NEAREST
        )
    
    # Encode geometry
    print("Encoding geometry...")
    embedding = predictor.encode_geometry(geometry_mask, args.resolution)
    
    # Print physics parameters
    print(f"\nPhysics parameters:")
    print(f"  Thermal diffusivity (α): {args.alpha}")
    print(f"  Heat source position: ({args.source_x}, {args.source_y})")
    print(f"  Heat source intensity: {args.intensity}")
    
    # Run inference
    print("\nRunning inference...")
    
    import time
    start = time.perf_counter()
    
    temperature_field = predictor.predict(
        geometry_embedding=embedding,
        thermal_diffusivity=args.alpha,
        source_x=args.source_x,
        source_y=args.source_y,
        source_intensity=args.intensity,
        resolution=args.resolution,
    )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    inference_time = (time.perf_counter() - start) * 1000
    
    print(f"Inference completed in {inference_time:.2f} ms")
    print(f"Output shape: {temperature_field.shape}")
    
    # Apply geometry mask
    temperature_field = temperature_field * geometry_mask[np.newaxis, :, :]
    
    # Save results
    print(f"\nSaving results to {args.output_dir}")
    
    # Save temperature field
    np.save(args.output_dir / "temperature_field.npy", temperature_field)
    
    # Save metadata
    metadata = {
        'geometry': str(args.geometry),
        'physics': {
            'thermal_diffusivity': args.alpha,
            'source_x': args.source_x,
            'source_y': args.source_y,
            'intensity': args.intensity,
        },
        'inference_time_ms': inference_time,
        'output_shape': list(temperature_field.shape),
    }
    
    with open(args.output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Plot initial, middle, and final timesteps
    num_time_steps = temperature_field.shape[0]
    
    for t_idx, name in [
        (0, 'initial'),
        (num_time_steps // 2, 'middle'),
        (-1, 'final')
    ]:
        fig = plot_temperature_field(
            temperature_field,
            time_idx=t_idx,
            geometry_mask=geometry_mask,
            title=f"Temperature ({name})",
            save_path=args.output_dir / f"temperature_{name}.png",
            show=False,
        )
    
    # Create animation if requested
    if args.save_animation:
        print("Creating animation...")
        create_animation(
            temperature_field,
            geometry_mask=geometry_mask,
            fps=10,
            save_path=args.output_dir / "animation.gif",
        )
    
    # Run benchmark if requested
    if args.benchmark:
        print("\nRunning benchmark...")
        benchmark_results = predictor.benchmark(
            num_warmup=10,
            num_runs=100,
            batch_size=1,
            resolution=args.resolution,
        )
        
        print(f"\nBenchmark Results:")
        print(f"  Mean: {benchmark_results['mean_ms']:.2f} ms")
        print(f"  Std:  {benchmark_results['std_ms']:.2f} ms")
        print(f"  Min:  {benchmark_results['min_ms']:.2f} ms")
        print(f"  Max:  {benchmark_results['max_ms']:.2f} ms")
        print(f"  Throughput: {benchmark_results['throughput_samples_per_sec']:.1f} samples/sec")
        
        with open(args.output_dir / "benchmark.json", 'w') as f:
            json.dump(benchmark_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
