"""
Visualization utilities for thermal fields.

Provides heatmap plots, animations, and comparison visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from pathlib import Path
from typing import Optional, Tuple, List, Union
import imageio


def plot_temperature_field(
    temperature: np.ndarray,
    time_idx: int = -1,
    geometry_mask: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    colormap: str = "hot",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a single temperature field snapshot.
    
    Args:
        temperature: Temperature field [T, H, W] or [H, W]
        time_idx: Time index to plot (if 3D array)
        geometry_mask: Optional geometry mask for overlay
        title: Plot title
        colormap: Matplotlib colormap
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display figure
        
    Returns:
        Matplotlib figure
    """
    # Handle different input shapes
    if temperature.ndim == 3:
        field = temperature[time_idx]
    else:
        field = temperature
    
    # Apply mask if provided
    if geometry_mask is not None:
        field = field * geometry_mask
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(
        field,
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        origin='lower',
        extent=[0, 1, 0, 1],
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Temperature')
    
    # Add geometry boundary if mask provided
    if geometry_mask is not None:
        ax.contour(
            geometry_mask,
            levels=[0.5],
            colors='white',
            linewidths=2,
            extent=[0, 1, 0, 1],
        )
    
    # Labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    if title:
        ax.set_title(title)
    elif temperature.ndim == 3:
        ax.set_title(f'Temperature at t = {time_idx / (temperature.shape[0] - 1):.2f}')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def create_animation(
    temperature: np.ndarray,
    geometry_mask: Optional[np.ndarray] = None,
    colormap: str = "hot",
    fps: int = 10,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Path] = None,
    title: str = "Temperature Evolution",
) -> Union[animation.FuncAnimation, Path]:
    """
    Create animated visualization of temperature evolution.
    
    Args:
        temperature: Temperature field [T, H, W]
        geometry_mask: Optional geometry mask
        colormap: Matplotlib colormap
        fps: Frames per second
        figsize: Figure size
        save_path: Path to save animation (GIF or MP4)
        title: Animation title
        
    Returns:
        Animation object or path to saved file
    """
    num_frames = temperature.shape[0]
    
    # Compute global min/max for consistent colormap
    if geometry_mask is not None:
        masked_temp = temperature * geometry_mask[np.newaxis, :, :]
        vmin = masked_temp.min()
        vmax = masked_temp.max()
    else:
        vmin = temperature.min()
        vmax = temperature.max()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Initial frame
    if geometry_mask is not None:
        field = temperature[0] * geometry_mask
    else:
        field = temperature[0]
    
    im = ax.imshow(
        field,
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        origin='lower',
        extent=[0, 1, 0, 1],
    )
    
    cbar = plt.colorbar(im, ax=ax, label='Temperature')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    time_text = ax.set_title(f'{title} (t = 0.00)')
    
    # Add geometry boundary
    if geometry_mask is not None:
        ax.contour(
            geometry_mask,
            levels=[0.5],
            colors='white',
            linewidths=1.5,
            extent=[0, 1, 0, 1],
        )
    
    def update(frame):
        if geometry_mask is not None:
            field = temperature[frame] * geometry_mask
        else:
            field = temperature[frame]
        
        im.set_array(field)
        t = frame / (num_frames - 1)
        time_text.set_text(f'{title} (t = {t:.2f})')
        return [im, time_text]
    
    anim = animation.FuncAnimation(
        fig, update,
        frames=num_frames,
        interval=1000 / fps,
        blit=True,
    )
    
    if save_path:
        save_path = Path(save_path)
        
        if save_path.suffix == '.gif':
            # Save as GIF using imageio
            frames = []
            for i in range(num_frames):
                if geometry_mask is not None:
                    field = temperature[i] * geometry_mask
                else:
                    field = temperature[i]
                
                # Normalize to 0-255
                norm = Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.get_cmap(colormap)
                colored = (cmap(norm(field)) * 255).astype(np.uint8)
                frames.append(colored)
            
            imageio.mimsave(save_path, frames, fps=fps)
        else:
            # Save as video
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, bitrate=1800)
            anim.save(str(save_path), writer=writer)
        
        plt.close(fig)
        return save_path
    
    return anim


def compare_predictions(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    time_idx: int = -1,
    geometry_mask: Optional[np.ndarray] = None,
    colormap: str = "hot",
    error_colormap: str = "coolwarm",
    figsize: Tuple[int, int] = (16, 4),
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Compare ground truth and predicted temperature fields.
    
    Args:
        ground_truth: Ground truth field [T, H, W]
        prediction: Predicted field [T, H, W]
        time_idx: Time index to compare
        geometry_mask: Optional geometry mask
        colormap: Colormap for temperature
        error_colormap: Colormap for error
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    # Extract time slice
    if ground_truth.ndim == 3:
        gt = ground_truth[time_idx]
        pred = prediction[time_idx]
    else:
        gt = ground_truth
        pred = prediction
    
    # Apply mask
    if geometry_mask is not None:
        gt = gt * geometry_mask
        pred = pred * geometry_mask
    
    # Compute error
    error = pred - gt
    
    # Compute metrics
    mse = np.mean(error ** 2)
    rel_l2 = np.linalg.norm(error) / np.linalg.norm(gt)
    max_err = np.max(np.abs(error))
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Common colormap range
    vmin = min(gt.min(), pred.min())
    vmax = max(gt.max(), pred.max())
    
    # Ground truth
    im0 = axes[0].imshow(
        gt, cmap=colormap, vmin=vmin, vmax=vmax,
        origin='lower', extent=[0, 1, 0, 1]
    )
    axes[0].set_title('Ground Truth')
    plt.colorbar(im0, ax=axes[0], label='T')
    
    # Prediction
    im1 = axes[1].imshow(
        pred, cmap=colormap, vmin=vmin, vmax=vmax,
        origin='lower', extent=[0, 1, 0, 1]
    )
    axes[1].set_title('Prediction')
    plt.colorbar(im1, ax=axes[1], label='T')
    
    # Absolute error
    im2 = axes[2].imshow(
        np.abs(error), cmap='hot',
        origin='lower', extent=[0, 1, 0, 1]
    )
    axes[2].set_title('Absolute Error')
    plt.colorbar(im2, ax=axes[2], label='|Error|')
    
    # Signed error
    err_max = np.max(np.abs(error))
    im3 = axes[3].imshow(
        error, cmap=error_colormap, vmin=-err_max, vmax=err_max,
        origin='lower', extent=[0, 1, 0, 1]
    )
    axes[3].set_title('Signed Error')
    plt.colorbar(im3, ax=axes[3], label='Error')
    
    # Add geometry boundaries
    if geometry_mask is not None:
        for ax in axes:
            ax.contour(
                geometry_mask, levels=[0.5],
                colors='cyan', linewidths=1,
                extent=[0, 1, 0, 1]
            )
    
    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    
    # Add metrics text
    fig.suptitle(
        f'MSE: {mse:.2e}  |  Rel. L2: {rel_l2:.4f}  |  Max Error: {max_err:.4f}',
        y=1.02
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_temporal_evolution(
    temperature: np.ndarray,
    points: List[Tuple[int, int]],
    point_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot temperature evolution at specific points over time.
    
    Args:
        temperature: Temperature field [T, H, W]
        points: List of (y, x) pixel coordinates
        point_labels: Optional labels for each point
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    num_time_steps = temperature.shape[0]
    time_vals = np.linspace(0, 1, num_time_steps)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, (y, x) in enumerate(points):
        temp_at_point = temperature[:, y, x]
        label = point_labels[i] if point_labels else f'Point ({x}, {y})'
        ax.plot(time_vals, temp_at_point, label=label, linewidth=2)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature')
    ax.set_title('Temperature Evolution at Selected Points')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_training_history(
    history: List[dict],
    metrics: List[str] = ['train_loss', 'val_loss', 'val_rel_l2'],
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot training history curves.
    
    Args:
        history: List of metric dictionaries from training
        metrics: Metrics to plot
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    epochs = [h['epoch'] for h in history]
    
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
    
    if num_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        values = [h.get(metric, np.nan) for h in history]
        ax.plot(epochs, values, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        
        # Log scale for loss metrics
        if 'loss' in metric.lower():
            ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig
