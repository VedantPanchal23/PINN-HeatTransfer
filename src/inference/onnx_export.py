"""
ONNX Export utilities for model deployment.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
import numpy as np


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    embedding_dim: int = 512,
    physics_dim: int = 4,
    time_steps: int = 50,
    resolution: int = 128,
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """
    Export model to ONNX format.
    
    Args:
        model: Trained PyTorch model
        output_path: Path to save ONNX model
        embedding_dim: Geometry embedding dimension
        physics_dim: Physics parameters dimension
        time_steps: Number of output time steps
        resolution: Spatial resolution
        opset_version: ONNX opset version
        dynamic_batch: Whether to use dynamic batch size
        
    Returns:
        Path to exported ONNX model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    
    # Create dummy inputs
    batch_size = 1
    dummy_embedding = torch.randn(batch_size, embedding_dim, device=device)
    dummy_physics = torch.randn(batch_size, physics_dim, device=device)
    
    # Define input/output names
    input_names = ['geometry_embedding', 'physics_params']
    output_names = ['temperature_field']
    
    # Dynamic axes for variable batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'geometry_embedding': {0: 'batch_size'},
            'physics_params': {0: 'batch_size'},
            'temperature_field': {0: 'batch_size'},
        }
    
    # Export
    print(f"Exporting model to {output_path}...")
    
    # Wrap model for consistent interface
    class ONNXWrapper(nn.Module):
        def __init__(self, model, resolution):
            super().__init__()
            self.model = model
            self.resolution = resolution
        
        def forward(self, embedding, physics):
            return self.model(embedding, physics, resolution=self.resolution)
    
    wrapper = ONNXWrapper(model, resolution)
    
    torch.onnx.export(
        wrapper,
        (dummy_embedding, dummy_physics),
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    
    print(f"Model exported to {output_path}")
    
    # Verify export
    verify_onnx_export(output_path, model, embedding_dim, physics_dim, resolution)
    
    return output_path


def verify_onnx_export(
    onnx_path: Path,
    pytorch_model: nn.Module,
    embedding_dim: int,
    physics_dim: int,
    resolution: int,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """
    Verify ONNX export matches PyTorch output.
    
    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model
        embedding_dim: Embedding dimension
        physics_dim: Physics dimension
        resolution: Spatial resolution
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        True if verification passes
    """
    import onnx
    import onnxruntime as ort
    
    # Load and check ONNX model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure verified")
    
    # Create test inputs
    device = next(pytorch_model.parameters()).device
    test_embedding = torch.randn(1, embedding_dim, device=device)
    test_physics = torch.randn(1, physics_dim, device=device)
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(
            test_embedding, test_physics, resolution=resolution
        ).cpu().numpy()
    
    # ONNX Runtime inference
    session = ort.InferenceSession(
        str(onnx_path),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    onnx_output = session.run(
        None,
        {
            'geometry_embedding': test_embedding.cpu().numpy(),
            'physics_params': test_physics.cpu().numpy(),
        }
    )[0]
    
    # Compare outputs
    np.testing.assert_allclose(
        pytorch_output, onnx_output,
        rtol=rtol, atol=atol,
        err_msg="ONNX output does not match PyTorch output"
    )
    
    print("ONNX export verification passed!")
    return True


def optimize_onnx(
    onnx_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Optimize ONNX model for inference.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Path for optimized model
        
    Returns:
        Path to optimized model
    """
    import onnx
    from onnxruntime.transformers import optimizer
    
    if output_path is None:
        output_path = onnx_path.parent / f"{onnx_path.stem}_optimized.onnx"
    
    # Load model
    model = onnx.load(str(onnx_path))
    
    # Optimize
    optimized_model = optimizer.optimize_model(
        str(onnx_path),
        model_type='bert',  # Use generic optimization
        num_heads=0,
        hidden_size=0,
    )
    
    optimized_model.save_model_to_file(str(output_path))
    
    print(f"Optimized model saved to {output_path}")
    return output_path


def quantize_onnx(
    onnx_path: Path,
    output_path: Optional[Path] = None,
    quantization_mode: str = "dynamic",
) -> Path:
    """
    Quantize ONNX model for faster inference.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Path for quantized model
        quantization_mode: 'dynamic' or 'static'
        
    Returns:
        Path to quantized model
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    if output_path is None:
        output_path = onnx_path.parent / f"{onnx_path.stem}_quantized.onnx"
    
    if quantization_mode == "dynamic":
        quantize_dynamic(
            str(onnx_path),
            str(output_path),
            weight_type=QuantType.QInt8,
        )
    else:
        raise ValueError(f"Unsupported quantization mode: {quantization_mode}")
    
    print(f"Quantized model saved to {output_path}")
    return output_path
