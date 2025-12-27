"""
Evaluation metrics for thermal surrogate model.

Computes MSE, Relative L2 error, Max error, and inference speed comparisons.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Union
from torch.utils.data import DataLoader
from tqdm import tqdm
import time


def compute_mse(
    prediction: Union[np.ndarray, torch.Tensor],
    ground_truth: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """
    Compute Mean Squared Error.
    
    Args:
        prediction: Predicted values
        ground_truth: Ground truth values
        mask: Optional mask (1 = include, 0 = exclude)
        
    Returns:
        MSE value
    """
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    error = prediction - ground_truth
    
    if mask is not None:
        # Broadcast mask if needed
        if mask.ndim < error.ndim:
            for _ in range(error.ndim - mask.ndim):
                mask = np.expand_dims(mask, 0)
        error = error * mask
        mse = np.sum(error ** 2) / np.sum(mask)
    else:
        mse = np.mean(error ** 2)
    
    return float(mse)


def compute_relative_l2(
    prediction: Union[np.ndarray, torch.Tensor],
    ground_truth: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """
    Compute Relative L2 Error.
    
    rel_l2 = ||pred - gt||_2 / ||gt||_2
    
    Args:
        prediction: Predicted values
        ground_truth: Ground truth values
        mask: Optional mask
        
    Returns:
        Relative L2 error
    """
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    error = prediction - ground_truth
    
    if mask is not None:
        if mask.ndim < error.ndim:
            for _ in range(error.ndim - mask.ndim):
                mask = np.expand_dims(mask, 0)
        error = error * mask
        ground_truth = ground_truth * mask
    
    error_norm = np.linalg.norm(error.flatten())
    gt_norm = np.linalg.norm(ground_truth.flatten())
    
    if gt_norm < 1e-10:
        return float('inf')
    
    return float(error_norm / gt_norm)


def compute_max_error(
    prediction: Union[np.ndarray, torch.Tensor],
    ground_truth: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """
    Compute Maximum Absolute Error.
    
    Args:
        prediction: Predicted values
        ground_truth: Ground truth values
        mask: Optional mask
        
    Returns:
        Maximum absolute error
    """
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    error = np.abs(prediction - ground_truth)
    
    if mask is not None:
        if mask.ndim < error.ndim:
            for _ in range(error.ndim - mask.ndim):
                mask = np.expand_dims(mask, 0)
        error = error * mask
    
    return float(np.max(error))


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    return_predictions: bool = False,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Neural operator model
        dataloader: DataLoader with test data
        device: Device for inference
        return_predictions: Whether to return all predictions
        
    Returns:
        Dictionary with evaluation metrics
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    total_mse = 0.0
    total_rel_l2 = 0.0
    total_max_error = 0.0
    num_samples = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        embeddings = batch['embedding'].to(device)
        physics = batch['physics'].to(device)
        targets = batch['temperature'].to(device)
        
        predictions = model(embeddings, physics)
        
        batch_size = embeddings.shape[0]
        
        for i in range(batch_size):
            pred = predictions[i].cpu().numpy()
            gt = targets[i].cpu().numpy()
            
            total_mse += compute_mse(pred, gt)
            total_rel_l2 += compute_relative_l2(pred, gt)
            total_max_error = max(total_max_error, compute_max_error(pred, gt))
            num_samples += 1
            
            if return_predictions:
                all_predictions.append(pred)
                all_targets.append(gt)
    
    metrics = {
        'mse': total_mse / num_samples,
        'relative_l2': total_rel_l2 / num_samples,
        'max_error': total_max_error,
        'num_samples': num_samples,
    }
    
    if return_predictions:
        metrics['predictions'] = np.stack(all_predictions)
        metrics['targets'] = np.stack(all_targets)
    
    return metrics


def compare_inference_speed(
    surrogate_model: nn.Module,
    pinn_solver,  # PINNSolver instance
    geometry_mask: np.ndarray,
    physics_params: Dict[str, float],
    geometry_embedding: torch.Tensor,
    num_runs: int = 10,
    device: str = "cuda",
) -> Dict[str, Dict[str, float]]:
    """
    Compare inference speed between surrogate model and PINN solver.
    
    Args:
        surrogate_model: Trained neural operator
        pinn_solver: PINN solver instance
        geometry_mask: Geometry mask for PINN
        physics_params: Physics parameters dict
        geometry_embedding: Pre-computed geometry embedding
        num_runs: Number of runs for timing
        device: Device for surrogate inference
        
    Returns:
        Dictionary with timing comparison
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    surrogate_model.to(device)
    surrogate_model.eval()
    
    # Prepare surrogate inputs
    physics_tensor = torch.tensor([
        [physics_params['thermal_diffusivity'],
         physics_params['source_x'],
         physics_params['source_y'],
         physics_params['source_intensity']]
    ], device=device, dtype=torch.float32)
    
    if geometry_embedding.device != device:
        geometry_embedding = geometry_embedding.to(device)
    
    # Warmup surrogate
    print("Warming up surrogate...")
    with torch.no_grad():
        for _ in range(5):
            _ = surrogate_model(geometry_embedding, physics_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Time surrogate inference
    surrogate_times = []
    print("Timing surrogate...")
    
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = surrogate_model(geometry_embedding, physics_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            surrogate_times.append((end - start) * 1000)
    
    # Time PINN solver (single run due to long duration)
    print("Timing PINN solver...")
    
    heat_source = pinn_solver.create_heat_source(
        physics_params['source_x'],
        physics_params['source_y'],
        physics_params['source_intensity'],
    )
    
    pinn_times = []
    for _ in range(min(3, num_runs)):  # Fewer runs for PINN
        start = time.perf_counter()
        pinn_solver.train(
            geometry_mask=geometry_mask,
            heat_source=heat_source,
            verbose=False,
        )
        _ = pinn_solver.generate_temperature_field(
            geometry_mask=geometry_mask
        )
        end = time.perf_counter()
        pinn_times.append((end - start) * 1000)
    
    surrogate_times = np.array(surrogate_times)
    pinn_times = np.array(pinn_times)
    
    speedup = np.mean(pinn_times) / np.mean(surrogate_times)
    
    results = {
        'surrogate': {
            'mean_ms': float(np.mean(surrogate_times)),
            'std_ms': float(np.std(surrogate_times)),
            'min_ms': float(np.min(surrogate_times)),
            'max_ms': float(np.max(surrogate_times)),
        },
        'pinn': {
            'mean_ms': float(np.mean(pinn_times)),
            'std_ms': float(np.std(pinn_times)),
            'min_ms': float(np.min(pinn_times)),
            'max_ms': float(np.max(pinn_times)),
        },
        'speedup': float(speedup),
    }
    
    print(f"\n=== Inference Speed Comparison ===")
    print(f"Surrogate: {results['surrogate']['mean_ms']:.2f} ± {results['surrogate']['std_ms']:.2f} ms")
    print(f"PINN:      {results['pinn']['mean_ms']:.2f} ± {results['pinn']['std_ms']:.2f} ms")
    print(f"Speedup:   {speedup:.1f}x faster")
    
    return results


def summarize_metrics(metrics: Dict[str, float]) -> str:
    """
    Create a formatted summary string of evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Formatted string
    """
    lines = [
        "=" * 50,
        "EVALUATION METRICS",
        "=" * 50,
        f"Mean Squared Error (MSE):     {metrics['mse']:.6e}",
        f"Relative L2 Error:            {metrics['relative_l2']:.4f}",
        f"Maximum Absolute Error:       {metrics['max_error']:.4f}",
        f"Number of Test Samples:       {metrics['num_samples']}",
        "=" * 50,
    ]
    
    return "\n".join(lines)


def per_timestep_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute metrics for each timestep.
    
    Args:
        prediction: [T, H, W] or [B, T, H, W]
        ground_truth: Same shape as prediction
        
    Returns:
        Dictionary with per-timestep metrics
    """
    # Handle batch dimension
    if prediction.ndim == 4:
        # Average over batch
        prediction = prediction.mean(axis=0)
        ground_truth = ground_truth.mean(axis=0)
    
    num_timesteps = prediction.shape[0]
    
    mse = np.zeros(num_timesteps)
    rel_l2 = np.zeros(num_timesteps)
    max_err = np.zeros(num_timesteps)
    
    for t in range(num_timesteps):
        mse[t] = compute_mse(prediction[t], ground_truth[t])
        rel_l2[t] = compute_relative_l2(prediction[t], ground_truth[t])
        max_err[t] = compute_max_error(prediction[t], ground_truth[t])
    
    return {
        'mse_per_t': mse,
        'rel_l2_per_t': rel_l2,
        'max_error_per_t': max_err,
    }
