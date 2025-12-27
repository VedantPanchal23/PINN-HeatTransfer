"""
Evaluation Script for Thermal Surrogate Model

Comprehensive evaluation including:
- Metrics computation (MSE, Relative L2, Max Error)
- Visualization generation
- PINN vs Surrogate speed comparison
"""

import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
import json
import sys
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import FNO2d, DeepONet, UNetFNO
from src.models.fno import ConditionalFNO
from src.training import ThermalDataset, create_dataloaders
from src.pinn import PINNSolver, PINNConfig
from src.encoder import GeometryEncoder
from src.utils import (
    compute_mse,
    compute_relative_l2,
    compute_max_error,
    evaluate_model,
    compare_inference_speed,
    summarize_metrics,
    load_config,
)
from src.visualization import (
    plot_temperature_field,
    compare_predictions,
    create_animation,
    plot_training_history,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate thermal surrogate model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Output directory"
    )
    parser.add_argument(
        "--compare-pinn",
        action="store_true",
        help="Compare with PINN solver speed"
    )
    parser.add_argument(
        "--num-visualizations",
        type=int,
        default=10,
        help="Number of samples to visualize"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: str) -> torch.nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # Infer model type from state dict
    if 'spectral_convs.0.weights1' in state_dict:
        model = ConditionalFNO()
    elif 'fno_blocks.0.spectral_conv.weights1' in state_dict:
        model = FNO2d()
    elif 'branch.network.0.weight' in state_dict:
        model = DeepONet()
    elif 'down1.spectral.weights1' in state_dict:
        model = UNetFNO()
    else:
        model = FNO2d()
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def main():
    args = parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("THERMAL SURROGATE MODEL EVALUATION")
    print("=" * 60)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Load test data
    print(f"\nLoading test data from {args.data_dir}")
    
    _, _, test_loader = create_dataloaders(
        train_path=args.data_dir / "train.h5",
        val_path=args.data_dir / "val.h5",
        test_path=args.data_dir / "test.h5",
        embeddings_path=args.data_dir / "embeddings.h5",
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )
    
    print(f"Test samples: {len(test_loader) * args.batch_size}")
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("COMPUTING METRICS")
    print("=" * 60)
    
    metrics = evaluate_model(
        model, 
        test_loader, 
        device=device,
        return_predictions=True,
    )
    
    print(summarize_metrics(metrics))
    
    # Save metrics
    metrics_to_save = {k: v for k, v in metrics.items() 
                       if k not in ['predictions', 'targets']}
    
    with open(args.output_dir / "metrics.json", 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    vis_dir = args.output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    predictions = metrics['predictions']
    targets = metrics['targets']
    
    num_vis = min(args.num_visualizations, len(predictions))
    
    for i in tqdm(range(num_vis), desc="Creating visualizations"):
        pred = predictions[i]
        gt = targets[i]
        
        # Comparison plot at final timestep
        fig = compare_predictions(
            ground_truth=gt,
            prediction=pred,
            time_idx=-1,
            save_path=vis_dir / f"comparison_{i:03d}.png",
            show=False,
        )
        
        # Animation
        create_animation(
            pred,
            fps=10,
            save_path=vis_dir / f"animation_pred_{i:03d}.gif",
        )
    
    # Per-timestep metrics
    print("\nComputing per-timestep metrics...")
    
    from src.utils.metrics import per_timestep_metrics
    
    # Average over samples
    avg_pred = predictions.mean(axis=0)
    avg_gt = targets.mean(axis=0)
    
    timestep_metrics = per_timestep_metrics(avg_pred, avg_gt)
    
    # Plot metrics over time
    import matplotlib.pyplot as plt
    
    num_t = len(timestep_metrics['mse_per_t'])
    time_vals = np.linspace(0, 1, num_t)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(time_vals, timestep_metrics['mse_per_t'])
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('MSE over Time')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(time_vals, timestep_metrics['rel_l2_per_t'])
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Relative L2')
    axes[1].set_title('Relative L2 Error over Time')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(time_vals, timestep_metrics['max_error_per_t'])
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Max Error')
    axes[2].set_title('Max Error over Time')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.output_dir / "metrics_over_time.png", dpi=150)
    plt.close()
    
    # Speed comparison with PINN
    if args.compare_pinn:
        print("\n" + "=" * 60)
        print("PINN vs SURROGATE SPEED COMPARISON")
        print("=" * 60)
        
        # Get a sample geometry
        import h5py
        
        with h5py.File(args.data_dir / "test.h5", 'r') as f:
            sample_mask = f['geometry_masks'][0]
            sample_physics = f['physics_params'][0]
            sample_hash = f['geometry_hashes'][0].decode()
        
        # Load embedding
        with h5py.File(args.data_dir / "embeddings.h5", 'r') as f:
            hashes = [h.decode() for h in f['hashes'][:]]
            embeddings = f['embeddings'][:]
            idx = hashes.index(sample_hash)
            sample_embedding = torch.tensor(
                embeddings[idx], dtype=torch.float32
            ).unsqueeze(0)
        
        # Create PINN solver
        pinn_config = PINNConfig(
            thermal_diffusivity=float(sample_physics[0]),
            max_iterations=5000,
        )
        pinn_solver = PINNSolver(pinn_config)
        
        # Physics params dict
        physics_dict = {
            'thermal_diffusivity': float(sample_physics[0]),
            'source_x': float(sample_physics[1]),
            'source_y': float(sample_physics[2]),
            'source_intensity': float(sample_physics[3]),
        }
        
        # Compare speeds
        speed_results = compare_inference_speed(
            surrogate_model=model,
            pinn_solver=pinn_solver,
            geometry_mask=sample_mask,
            physics_params=physics_dict,
            geometry_embedding=sample_embedding,
            num_runs=10,
            device=device,
        )
        
        with open(args.output_dir / "speed_comparison.json", 'w') as f:
            json.dump(speed_results, f, indent=2)
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("GENERATING REPORT")
    print("=" * 60)
    
    report = []
    report.append("# Thermal Surrogate Model Evaluation Report\n")
    report.append(f"**Checkpoint:** {args.checkpoint}\n")
    report.append(f"**Test Samples:** {metrics['num_samples']}\n")
    report.append("\n## Metrics\n")
    report.append(f"- **MSE:** {metrics['mse']:.6e}")
    report.append(f"- **Relative L2 Error:** {metrics['relative_l2']:.4f}")
    report.append(f"- **Max Error:** {metrics['max_error']:.4f}")
    
    if args.compare_pinn:
        report.append("\n## Speed Comparison\n")
        report.append(f"- **Surrogate:** {speed_results['surrogate']['mean_ms']:.2f} ms")
        report.append(f"- **PINN:** {speed_results['pinn']['mean_ms']:.2f} ms")
        report.append(f"- **Speedup:** {speed_results['speedup']:.1f}x")
    
    with open(args.output_dir / "report.md", 'w') as f:
        f.write('\n'.join(report))
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
