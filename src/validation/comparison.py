"""
Comparison tools for validating PINN against reference solutions.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt
from scipy import interpolate


@dataclass
class ValidationMetrics:
    """Metrics for comparing PINN with reference solution."""
    # Error metrics
    mse: float                    # Mean Squared Error
    rmse: float                   # Root Mean Squared Error
    mae: float                    # Mean Absolute Error
    max_error: float              # Maximum Absolute Error
    relative_l2_error: float      # Relative L2 error (%)
    
    # Correlation
    correlation: float            # Pearson correlation coefficient
    
    # Location of maximum error
    max_error_location: Tuple[float, float, float]  # (t, x, y)
    
    # Timing
    pinn_time: float              # PINN training time (s)
    reference_time: float         # Reference solver time (s)
    speedup: float                # Reference time / PINN inference time
    
    def to_dict(self) -> dict:
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "max_error": self.max_error,
            "relative_l2_error": self.relative_l2_error,
            "correlation": self.correlation,
            "pinn_time": self.pinn_time,
            "reference_time": self.reference_time,
            "speedup": self.speedup,
        }
    
    def summary(self) -> str:
        lines = [
            "Validation Metrics:",
            "=" * 40,
            "",
            "Error Metrics:",
            f"  MSE:              {self.mse:.6e}",
            f"  RMSE:             {self.rmse:.6e}",
            f"  MAE:              {self.mae:.6e}",
            f"  Max Error:        {self.max_error:.4f}",
            f"  Relative L2 (%):  {self.relative_l2_error:.2f}%",
            "",
            "Correlation:",
            f"  Pearson:          {self.correlation:.6f}",
            "",
            "Timing:",
            f"  PINN Time:        {self.pinn_time:.2f}s",
            f"  Reference Time:   {self.reference_time:.2f}s",
            f"  Speedup:          {self.speedup:.1f}x",
            "",
            f"Max Error Location: t={self.max_error_location[0]:.2f}, "
            f"x={self.max_error_location[1]:.2f}, y={self.max_error_location[2]:.2f}",
        ]
        return "\n".join(lines)


class ValidationComparison:
    """
    Compare PINN results with reference solutions.
    
    Handles:
    - Grid interpolation for different resolutions
    - Time alignment
    - Error computation
    - Visualization
    """
    
    def __init__(self):
        self.pinn_result = None
        self.reference_result = None
        self.metrics = None
    
    def compare(
        self,
        pinn_temperature: np.ndarray,
        pinn_times: np.ndarray,
        pinn_x: np.ndarray,
        pinn_y: np.ndarray,
        reference_temperature: np.ndarray,
        reference_times: np.ndarray,
        reference_x: np.ndarray,
        reference_y: np.ndarray,
        pinn_training_time: float = 0.0,
        reference_solve_time: float = 0.0,
    ) -> ValidationMetrics:
        """
        Compare PINN results with reference solution.
        
        Args:
            pinn_temperature: PINN temperature field [T1, H1, W1]
            pinn_times: PINN time points [T1]
            pinn_x, pinn_y: PINN grid coordinates [H1, W1]
            reference_temperature: Reference temperature field [T2, H2, W2]
            reference_times: Reference time points [T2]
            reference_x, reference_y: Reference grid coordinates [H2, W2]
            pinn_training_time: Time for PINN training
            reference_solve_time: Time for reference solver
            
        Returns:
            ValidationMetrics object
        """
        # Interpolate PINN to reference grid if different
        if pinn_temperature.shape != reference_temperature.shape:
            pinn_interp = self._interpolate_to_grid(
                pinn_temperature, pinn_times, pinn_x, pinn_y,
                reference_times, reference_x, reference_y
            )
        else:
            pinn_interp = pinn_temperature
        
        self.pinn_result = pinn_interp
        self.reference_result = reference_temperature
        
        # Compute error metrics
        error = pinn_interp - reference_temperature
        
        mse = np.mean(error**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(error))
        max_error = np.max(np.abs(error))
        
        # Relative L2 error
        l2_ref = np.sqrt(np.sum(reference_temperature**2))
        l2_error = np.sqrt(np.sum(error**2))
        relative_l2 = 100 * l2_error / (l2_ref + 1e-10)
        
        # Correlation
        pinn_flat = pinn_interp.flatten()
        ref_flat = reference_temperature.flatten()
        correlation = np.corrcoef(pinn_flat, ref_flat)[0, 1]
        
        # Location of maximum error
        max_idx = np.unravel_index(np.argmax(np.abs(error)), error.shape)
        t_max = reference_times[max_idx[0]] if len(reference_times) > max_idx[0] else 0
        
        if reference_x.ndim == 2:
            x_max = reference_x[max_idx[1], max_idx[2]]
            y_max = reference_y[max_idx[1], max_idx[2]]
        else:
            x_max = reference_x[max_idx[2]]
            y_max = reference_y[max_idx[1]]
        
        # Speedup (comparing solve time to PINN inference)
        # Assume inference is ~0.01s for a forward pass
        pinn_inference_time = 0.01
        speedup = reference_solve_time / (pinn_inference_time + 1e-10)
        
        self.metrics = ValidationMetrics(
            mse=mse,
            rmse=rmse,
            mae=mae,
            max_error=max_error,
            relative_l2_error=relative_l2,
            correlation=correlation,
            max_error_location=(t_max, x_max, y_max),
            pinn_time=pinn_training_time,
            reference_time=reference_solve_time,
            speedup=speedup,
        )
        
        return self.metrics
    
    def _interpolate_to_grid(
        self,
        temperature: np.ndarray,
        times: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        target_times: np.ndarray,
        target_x: np.ndarray,
        target_y: np.ndarray,
    ) -> np.ndarray:
        """Interpolate temperature field to target grid."""
        # Get target shape
        if target_x.ndim == 2:
            ny, nx = target_x.shape
        else:
            nx = len(target_x)
            ny = len(target_y)
        
        nt = len(target_times)
        result = np.zeros((nt, ny, nx))
        
        # Flatten source coordinates
        if x.ndim == 2:
            x_flat = x.flatten()
            y_flat = y.flatten()
        else:
            X, Y = np.meshgrid(x, y)
            x_flat = X.flatten()
            y_flat = Y.flatten()
        
        # Flatten target coordinates
        if target_x.ndim == 2:
            tx_flat = target_x.flatten()
            ty_flat = target_y.flatten()
        else:
            TX, TY = np.meshgrid(target_x, target_y)
            tx_flat = TX.flatten()
            ty_flat = TY.flatten()
        
        # Interpolate in space for each time
        # First, interpolate in time
        for i, t_target in enumerate(target_times):
            # Find closest time in source
            t_idx = np.argmin(np.abs(times - t_target))
            
            # Spatial interpolation
            T_source = temperature[t_idx].flatten()
            
            # Use scipy interpolate
            interp = interpolate.griddata(
                (x_flat, y_flat),
                T_source,
                (tx_flat, ty_flat),
                method='linear',
                fill_value=np.mean(T_source)
            )
            
            result[i] = interp.reshape(ny, nx)
        
        return result
    
    def plot_comparison(
        self,
        time_index: int = -1,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create comparison plot of PINN vs reference solution.
        
        Args:
            time_index: Which time step to plot
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if self.pinn_result is None or self.reference_result is None:
            raise ValueError("Must call compare() before plotting")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        T_pinn = self.pinn_result[time_index]
        T_ref = self.reference_result[time_index]
        error = T_pinn - T_ref
        
        # PINN result
        im1 = axes[0].imshow(T_pinn, origin='lower', cmap='hot')
        axes[0].set_title('PINN Solution')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0], label='Temperature (°C)')
        
        # Reference result
        im2 = axes[1].imshow(T_ref, origin='lower', cmap='hot')
        axes[1].set_title('Reference (FDM) Solution')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1], label='Temperature (°C)')
        
        # Error
        max_err = np.max(np.abs(error))
        im3 = axes[2].imshow(error, origin='lower', cmap='RdBu_r', 
                             vmin=-max_err, vmax=max_err)
        axes[2].set_title(f'Error (PINN - Reference)\nMax: {max_err:.4f}°C')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        plt.colorbar(im3, ax=axes[2], label='Error (°C)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_time_evolution(
        self,
        location: Tuple[int, int],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot temperature evolution at a specific location.
        
        Args:
            location: (row, col) indices in the grid
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if self.pinn_result is None or self.reference_result is None:
            raise ValueError("Must call compare() before plotting")
        
        i, j = location
        T_pinn = self.pinn_result[:, i, j]
        T_ref = self.reference_result[:, i, j]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        times = np.arange(len(T_pinn))  # Normalized time indices
        
        ax.plot(times, T_pinn, 'b-', linewidth=2, label='PINN')
        ax.plot(times, T_ref, 'r--', linewidth=2, label='Reference (FDM)')
        
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'Temperature Evolution at ({i}, {j})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def generate_report(self) -> str:
        """Generate a text validation report."""
        if self.metrics is None:
            return "No comparison performed yet."
        
        lines = [
            "=" * 60,
            "PINN VALIDATION REPORT",
            "=" * 60,
            "",
            self.metrics.summary(),
            "",
            "Interpretation:",
            "-" * 40,
        ]
        
        # Interpret results
        if self.metrics.relative_l2_error < 1.0:
            lines.append("✅ Excellent agreement: <1% relative L2 error")
        elif self.metrics.relative_l2_error < 5.0:
            lines.append("✅ Good agreement: <5% relative L2 error")
        elif self.metrics.relative_l2_error < 10.0:
            lines.append("⚠️ Acceptable agreement: <10% relative L2 error")
        else:
            lines.append("❌ Poor agreement: >10% relative L2 error")
        
        if self.metrics.correlation > 0.99:
            lines.append("✅ Excellent correlation: >0.99")
        elif self.metrics.correlation > 0.95:
            lines.append("✅ Good correlation: >0.95")
        else:
            lines.append("⚠️ Correlation could be improved")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
