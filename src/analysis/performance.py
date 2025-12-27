"""
Performance metrics for thermal simulation analysis.

Calculates key thermal performance indicators.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ThermalPerformanceMetrics:
    """Comprehensive thermal performance metrics."""
    # Temperature metrics
    max_temperature: float           # °C
    min_temperature: float           # °C
    mean_temperature: float          # °C
    temperature_uniformity: float    # Standard deviation
    
    # Thermal resistance
    thermal_resistance: float        # °C/W (junction to ambient)
    
    # Time metrics
    time_to_steady_state: float      # seconds
    thermal_time_constant: float     # seconds
    
    # Heat dissipation
    heat_dissipation_rate: float     # W
    cooling_efficiency: float        # % (actual vs ideal)
    
    # Gradients
    max_temperature_gradient: float  # °C/mm
    gradient_location: Tuple[float, float]
    
    def to_dict(self) -> dict:
        return {
            "max_temperature": self.max_temperature,
            "min_temperature": self.min_temperature,
            "mean_temperature": self.mean_temperature,
            "temperature_uniformity": self.temperature_uniformity,
            "thermal_resistance": self.thermal_resistance,
            "time_to_steady_state": self.time_to_steady_state,
            "thermal_time_constant": self.thermal_time_constant,
            "heat_dissipation_rate": self.heat_dissipation_rate,
            "cooling_efficiency": self.cooling_efficiency,
            "max_temperature_gradient": self.max_temperature_gradient,
        }
    
    def summary(self) -> str:
        lines = [
            "Thermal Performance Metrics:",
            "",
            "Temperature:",
            f"  Max: {self.max_temperature:.1f}°C",
            f"  Min: {self.min_temperature:.1f}°C",
            f"  Mean: {self.mean_temperature:.1f}°C",
            f"  Uniformity (σ): {self.temperature_uniformity:.2f}°C",
            "",
            "Thermal Resistance:",
            f"  Rth: {self.thermal_resistance:.3f} °C/W",
            "",
            "Time Response:",
            f"  Time to Steady State: {self.time_to_steady_state:.1f}s",
            f"  Thermal Time Constant: {self.thermal_time_constant:.2f}s",
            "",
            "Heat Dissipation:",
            f"  Rate: {self.heat_dissipation_rate:.2f} W",
            f"  Efficiency: {self.cooling_efficiency:.1f}%",
            "",
            "Gradients:",
            f"  Max Gradient: {self.max_temperature_gradient:.1f} °C/mm",
            f"  Location: ({self.gradient_location[0]:.2f}, {self.gradient_location[1]:.2f})",
        ]
        return "\n".join(lines)


class PerformanceAnalyzer:
    """
    Analyze thermal simulation results for performance metrics.
    """
    
    def __init__(
        self,
        domain_size: Tuple[float, float] = (0.1, 0.1),  # meters
        domain_thickness: float = 0.005,                 # meters
    ):
        """
        Initialize performance analyzer.
        
        Args:
            domain_size: Physical domain size (width, height) in meters
            domain_thickness: Domain thickness for 2D approximation
        """
        self.domain_size = domain_size
        self.domain_thickness = domain_thickness
    
    def analyze(
        self,
        temperature_field: np.ndarray,
        time_points: np.ndarray,
        heat_source_power: float,
        ambient_temperature: float = 25.0,
        heat_transfer_coeff: float = 10.0,
    ) -> ThermalPerformanceMetrics:
        """
        Analyze temperature field for performance metrics.
        
        Args:
            temperature_field: Temperature array [T, H, W]
            time_points: Time array [T] in seconds
            heat_source_power: Total heat source power (W)
            ambient_temperature: Ambient temperature (°C)
            heat_transfer_coeff: Convective heat transfer coefficient
            
        Returns:
            ThermalPerformanceMetrics object
        """
        # Validate inputs
        if temperature_field is None or temperature_field.size == 0:
            return self._default_metrics(ambient_temperature)
        
        # Handle NaN and Inf values
        if not np.isfinite(temperature_field).all():
            temperature_field = np.nan_to_num(temperature_field, nan=ambient_temperature, posinf=1e6, neginf=-1e6)
        
        # Get steady state field (last timestep)
        if temperature_field.ndim == 3:
            T_steady = temperature_field[-1]
        else:
            T_steady = temperature_field
        
        # Temperature metrics
        max_temp = np.max(T_steady)
        min_temp = np.min(T_steady)
        mean_temp = np.mean(T_steady)
        temp_std = np.std(T_steady)
        
        # Thermal resistance
        if heat_source_power > 0:
            delta_T = max_temp - ambient_temperature
            thermal_resistance = delta_T / heat_source_power
        else:
            thermal_resistance = float('inf') if max_temp > ambient_temperature else 0.0
        
        # Time metrics
        time_to_steady = self._calculate_time_to_steady(
            temperature_field, time_points
        )
        time_constant = self._calculate_time_constant(
            temperature_field, time_points
        )
        
        # Heat dissipation
        surface_area = 2 * (
            self.domain_size[0] * self.domain_size[1] +
            self.domain_size[0] * self.domain_thickness +
            self.domain_size[1] * self.domain_thickness
        )
        
        avg_surface_temp = mean_temp  # Simplified
        heat_dissipation = heat_transfer_coeff * surface_area * (
            avg_surface_temp - ambient_temperature
        )
        
        # Cooling efficiency
        if heat_source_power > 0:
            efficiency = min(heat_dissipation / heat_source_power * 100, 100)
        else:
            efficiency = 100.0
        
        # Temperature gradients
        max_gradient, gradient_loc = self._calculate_max_gradient(T_steady)
        
        return ThermalPerformanceMetrics(
            max_temperature=max_temp,
            min_temperature=min_temp,
            mean_temperature=mean_temp,
            temperature_uniformity=temp_std,
            thermal_resistance=thermal_resistance,
            time_to_steady_state=time_to_steady,
            thermal_time_constant=time_constant,
            heat_dissipation_rate=heat_dissipation,
            cooling_efficiency=efficiency,
            max_temperature_gradient=max_gradient,
            gradient_location=gradient_loc,
        )
    
    def _calculate_time_to_steady(
        self,
        temperature_field: np.ndarray,
        time_points: np.ndarray,
        threshold: float = 0.99,
    ) -> float:
        """Calculate time to reach steady state (99% of final value)."""
        if temperature_field.ndim != 3 or len(time_points) < 2:
            return 0.0
        
        # Track maximum temperature over time
        max_temps = temperature_field.max(axis=(1, 2))
        final_temp = max_temps[-1]
        initial_temp = max_temps[0]
        
        # If no temperature change, already at steady state
        if abs(final_temp - initial_temp) < 1e-10:
            return 0.0
        
        target = initial_temp + threshold * (final_temp - initial_temp)
        
        for i, t in enumerate(time_points):
            if max_temps[i] >= target:
                return t
        
        return time_points[-1]
    
    def _calculate_time_constant(
        self,
        temperature_field: np.ndarray,
        time_points: np.ndarray,
    ) -> float:
        """
        Calculate thermal time constant (time to reach 63.2% of change).
        """
        if temperature_field.ndim != 3 or len(time_points) < 2:
            return 0.0
        
        max_temps = temperature_field.max(axis=(1, 2))
        final_temp = max_temps[-1]
        initial_temp = max_temps[0]
        
        # If no temperature change, return 0
        if abs(final_temp - initial_temp) < 1e-10:
            return 0.0
        
        # Time constant is when T = T0 + 0.632 * (T_final - T0)
        target = initial_temp + 0.632 * (final_temp - initial_temp)
        
        for i, t in enumerate(time_points):
            if max_temps[i] >= target:
                return t
        
        return time_points[-1] / 5  # Rough estimate
    
    def _default_metrics(self, ambient_temperature: float = 25.0) -> ThermalPerformanceMetrics:
        """Return default metrics when input is invalid."""
        return ThermalPerformanceMetrics(
            max_temperature=ambient_temperature,
            min_temperature=ambient_temperature,
            mean_temperature=ambient_temperature,
            temperature_uniformity=0.0,
            thermal_resistance=0.0,
            time_to_steady_state=0.0,
            thermal_time_constant=0.0,
            heat_dissipation_rate=0.0,
            cooling_efficiency=100.0,
            max_temperature_gradient=0.0,
            gradient_location=(0.5, 0.5),
        )
    
    def _calculate_max_gradient(
        self,
        temperature_field: np.ndarray,
    ) -> Tuple[float, Tuple[float, float]]:
        """Calculate maximum temperature gradient in °C/mm."""
        h, w = temperature_field.shape
        
        # Handle edge cases
        if h < 2 or w < 2:
            return 0.0, (0.5, 0.5)
        
        # Calculate gradients
        dy = self.domain_size[1] / h * 1000  # mm
        dx = self.domain_size[0] / w * 1000  # mm
        
        grad_y, grad_x = np.gradient(temperature_field, dy, dx)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        max_grad = np.max(gradient_magnitude)
        max_idx = np.unravel_index(np.argmax(gradient_magnitude), gradient_magnitude.shape)
        
        location = (max_idx[1] / w, max_idx[0] / h)  # Normalized (x, y)
        
        return max_grad, location
    
    def compare_designs(
        self,
        metrics_list: list,
        design_names: list,
    ) -> str:
        """Compare performance metrics across multiple designs."""
        if len(metrics_list) != len(design_names):
            raise ValueError("Number of metrics must match number of names")
        
        lines = ["Design Comparison:", ""]
        
        # Header
        header = f"{'Metric':<25}"
        for name in design_names:
            header += f"{name:<15}"
        lines.append(header)
        lines.append("-" * (25 + 15 * len(design_names)))
        
        # Metrics
        metric_names = [
            ("Max Temperature (°C)", "max_temperature"),
            ("Mean Temperature (°C)", "mean_temperature"),
            ("Thermal Resistance (°C/W)", "thermal_resistance"),
            ("Time to Steady (s)", "time_to_steady_state"),
            ("Efficiency (%)", "cooling_efficiency"),
            ("Max Gradient (°C/mm)", "max_temperature_gradient"),
        ]
        
        for display_name, attr_name in metric_names:
            row = f"{display_name:<25}"
            for m in metrics_list:
                value = getattr(m, attr_name)
                row += f"{value:<15.2f}"
            lines.append(row)
        
        return "\n".join(lines)
