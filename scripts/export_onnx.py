"""
ONNX Export Script

Exports the trained thermal surrogate model to ONNX format for deployment.
"""

import argparse
from pathlib import Path
import torch
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import FNO2d, DeepONet, UNetFNO
from src.models.fno import ConditionalFNO
from src.inference.onnx_export import export_to_onnx, optimize_onnx, quantize_onnx


def parse_args():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output ONNX path (default: same dir as checkpoint)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="Spatial resolution"
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=50,
        help="Number of time steps"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply ONNX optimization"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply dynamic quantization"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference benchmark after export"
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path) -> torch.nn.Module:
    """Load model from checkpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # Infer model type
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
    
    print("=" * 60)
    print("ONNX EXPORT")
    print("=" * 60)
    
    # Determine output path
    if args.output is None:
        output_path = args.checkpoint.parent / f"{args.checkpoint.stem}.onnx"
    else:
        output_path = args.output
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Export to ONNX
    print(f"\nExporting to {output_path}")
    
    onnx_path = export_to_onnx(
        model=model,
        output_path=output_path,
        embedding_dim=512,
        physics_dim=4,
        time_steps=args.time_steps,
        resolution=args.resolution,
        dynamic_batch=True,
    )
    
    # Optimize if requested
    if args.optimize:
        print("\nOptimizing ONNX model...")
        optimized_path = optimize_onnx(onnx_path)
        print(f"Optimized model: {optimized_path}")
    
    # Quantize if requested
    if args.quantize:
        print("\nQuantizing ONNX model...")
        source_path = optimized_path if args.optimize else onnx_path
        quantized_path = quantize_onnx(source_path)
        print(f"Quantized model: {quantized_path}")
    
    # Benchmark if requested
    if args.benchmark:
        print("\nRunning inference benchmark...")
        
        from src.inference import ThermalPredictor
        from src.inference.predictor import ONNXPredictor
        import numpy as np
        
        # PyTorch benchmark
        print("\nPyTorch inference:")
        predictor = ThermalPredictor(
            model_path=args.checkpoint,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compile_model=False,
        )
        pytorch_results = predictor.benchmark(
            num_warmup=10,
            num_runs=100,
            resolution=args.resolution,
        )
        print(f"  Mean: {pytorch_results['mean_ms']:.2f} ms")
        print(f"  Std:  {pytorch_results['std_ms']:.2f} ms")
        
        # ONNX benchmark
        print("\nONNX Runtime inference:")
        onnx_predictor = ONNXPredictor(
            onnx_path=onnx_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        onnx_results = onnx_predictor.benchmark(
            num_warmup=10,
            num_runs=100,
        )
        print(f"  Mean: {onnx_results['mean_ms']:.2f} ms")
        print(f"  Std:  {onnx_results['std_ms']:.2f} ms")
        
        # Speedup
        speedup = pytorch_results['mean_ms'] / onnx_results['mean_ms']
        print(f"\nONNX speedup: {speedup:.2f}x")
    
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"ONNX model: {onnx_path}")


if __name__ == "__main__":
    main()
