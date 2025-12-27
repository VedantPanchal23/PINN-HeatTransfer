"""
Thermal limit analysis for material safety evaluation.

Provides analysis of:
- Maximum safe operating temperature
- Time to reach critical temperature
- Thermal failure risk assessment
- Temperature margin analysis
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

from .mixture import EffectiveProperties


class RiskLevel(Enum):
    """Thermal risk classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ThermalLimitResult:
    """Results of thermal limit analysis."""
    # Temperature limits
    max_operating_temp: float        # °C
    melting_point: float             # °C
    
    # Predicted temperatures
    steady_state_temp: float         # °C
    max_transient_temp: float        # °C
    time_to_steady_state: float      # seconds
    
    # Safety margins
    temp_margin: float               # °C (max_operating - steady_state)
    safety_factor: float             # ratio
    
    # Risk assessment
    risk_level: RiskLevel
    is_safe: bool
    
    # Time to failure (if applicable)
    time_to_max_operating: Optional[float] = None  # seconds
    time_to_melting: Optional[float] = None        # seconds
    
    # Geometry lifetime analysis
    geometry_lifetime: Optional[float] = None      # seconds (safe operating time)
    lifetime_limiting_factor: Optional[str] = None # what limits the lifetime
    max_temperature_capacity: Optional[float] = None  # °C (max temp the geometry can handle)
    thermal_headroom: Optional[float] = None      # % remaining capacity
    
    # Recommendations
    warnings: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.recommendations is None:
            self.recommendations = []
    
    def summary(self) -> str:
        """Get human-readable summary."""
        status = "✅ SAFE" if self.is_safe else "❌ UNSAFE"
        lines = [
            f"Thermal Limit Analysis: {status}",
            f"",
            f"Temperature Limits:",
            f"  Max Operating: {self.max_operating_temp:.1f}°C",
            f"  Melting Point: {self.melting_point:.1f}°C",
            f"  Max Temperature Capacity: {self.max_temperature_capacity:.1f}°C" if self.max_temperature_capacity else "",
            f"",
            f"Predicted Temperatures:",
            f"  Steady State: {self.steady_state_temp:.1f}°C",
            f"  Max Transient: {self.max_transient_temp:.1f}°C",
            f"  Time to Steady State: {self.time_to_steady_state:.1f}s",
            f"",
            f"Safety:",
            f"  Temperature Margin: {self.temp_margin:.1f}°C",
            f"  Safety Factor: {self.safety_factor:.2f}",
            f"  Risk Level: {self.risk_level.value.upper()}",
            f"  Thermal Headroom: {self.thermal_headroom:.1f}%" if self.thermal_headroom is not None else "",
        ]
        
        # Filter out empty strings
        lines = [l for l in lines if l]
        
        if self.time_to_max_operating is not None:
            lines.append(f"  Time to Max Operating: {self.time_to_max_operating:.1f}s")
        
        if self.time_to_melting is not None:
            lines.append(f"  ⚠️ Time to Melting: {self.time_to_melting:.1f}s")
        
        # Geometry lifetime
        lines.append(f"")
        lines.append(f"Geometry Lifetime:")
        if self.geometry_lifetime is not None:
            if self.geometry_lifetime == float('inf'):
                lines.append(f"  ✅ Safe Operating Time: Indefinite (stable)")
            else:
                # Format time nicely
                lifetime = self.geometry_lifetime
                if lifetime < 60:
                    lines.append(f"  ⏱️ Safe Operating Time: {lifetime:.1f} seconds")
                elif lifetime < 3600:
                    lines.append(f"  ⏱️ Safe Operating Time: {lifetime/60:.1f} minutes")
                else:
                    lines.append(f"  ⏱️ Safe Operating Time: {lifetime/3600:.1f} hours")
        else:
            lines.append(f"  Safe Operating Time: Not calculated")
        
        if self.lifetime_limiting_factor:
            lines.append(f"  Limiting Factor: {self.lifetime_limiting_factor}")
        
        if self.warnings:
            lines.append(f"")
            lines.append(f"⚠️ Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        
        if self.recommendations:
            lines.append(f"")
            lines.append(f"💡 Recommendations:")
            for r in self.recommendations:
                lines.append(f"  - {r}")
        
        return "\n".join(lines)


class ThermalLimitAnalyzer:
    """
    Analyze thermal limits and safety margins.
    
    Uses PINN predictions or analytical estimates to determine:
    - Whether the design is thermally safe
    - Time margins before reaching critical temperatures
    - Design recommendations for improving thermal performance
    """
    
    def __init__(
        self,
        safety_margin_factor: float = 0.8,
        critical_temp_fraction: float = 0.95,
    ):
        """
        Initialize thermal limit analyzer.
        
        Args:
            safety_margin_factor: Fraction of max operating temp considered safe
            critical_temp_fraction: Fraction of max temp considered critical risk
        """
        self.safety_margin_factor = safety_margin_factor
        self.critical_temp_fraction = critical_temp_fraction
    
    def analyze(
        self,
        material_props: EffectiveProperties,
        heat_source_power: float,        # W
        domain_area: float,              # m²
        domain_thickness: float,         # m
        ambient_temp: float = 25.0,      # °C
        heat_transfer_coeff: float = 10.0,  # W/(m²·K) natural convection
        initial_temp: Optional[float] = None,  # °C
        predicted_temps: Optional[np.ndarray] = None,  # From PINN
        predicted_times: Optional[np.ndarray] = None,  # From PINN
    ) -> ThermalLimitResult:
        """
        Perform thermal limit analysis.
        
        Args:
            material_props: Effective material properties
            heat_source_power: Total heat source power (W)
            domain_area: Surface area of domain (m²)
            domain_thickness: Thickness for lumped analysis (m)
            ambient_temp: Ambient temperature (°C)
            heat_transfer_coeff: Convective heat transfer coefficient
            initial_temp: Initial temperature (defaults to ambient)
            predicted_temps: Temperature field from PINN (optional)
            predicted_times: Time points from PINN (optional)
            
        Returns:
            ThermalLimitResult with analysis results
        """
        if initial_temp is None:
            initial_temp = ambient_temp
        
        max_op = material_props.max_operating_temp
        melt = material_props.min_melting_point
        
        # Calculate thermal properties
        k = material_props.thermal_conductivity
        rho = material_props.density
        cp = material_props.specific_heat
        alpha = material_props.thermal_diffusivity
        
        # Volume and thermal capacity
        volume = domain_area * domain_thickness
        thermal_capacity = rho * cp * volume  # J/K
        
        # Effective thermal resistance
        R_conv = 1.0 / (heat_transfer_coeff * domain_area)  # K/W
        
        # Estimate steady-state temperature rise
        # Using simple thermal resistance model: dT = Q * R
        steady_state_rise = heat_source_power * R_conv
        steady_state_temp = ambient_temp + steady_state_rise
        
        # Time constant for lumped system
        # tau = rho * V * cp / (h * A) = thermal_capacity * R_conv
        time_constant = thermal_capacity * R_conv
        time_to_steady = 5 * time_constant  # 99.3% of steady state
        
        # Use PINN predictions if available
        if predicted_temps is not None and predicted_times is not None:
            max_transient_temp = float(np.max(predicted_temps))
            # Find time to reach 99% of steady state
            if len(predicted_temps) > 0:
                final_temp = predicted_temps[-1] if isinstance(predicted_temps[-1], (int, float)) else np.max(predicted_temps[-1])
                steady_state_temp = final_temp
        else:
            # Estimate max transient (could be higher than steady state with pulsed sources)
            max_transient_temp = steady_state_temp
        
        # Safety analysis
        temp_margin = max_op - steady_state_temp
        safety_factor = max_op / max(steady_state_temp, 1.0)
        
        # Risk assessment
        if steady_state_temp >= melt:
            risk_level = RiskLevel.CRITICAL
            is_safe = False
        elif steady_state_temp >= max_op:
            risk_level = RiskLevel.CRITICAL
            is_safe = False
        elif steady_state_temp >= max_op * self.critical_temp_fraction:
            risk_level = RiskLevel.HIGH
            is_safe = False
        elif steady_state_temp >= max_op * self.safety_margin_factor:
            risk_level = RiskLevel.MEDIUM
            is_safe = True  # Marginal
        else:
            risk_level = RiskLevel.LOW
            is_safe = True
        
        # Time to reach critical temperatures
        time_to_max_op = self._time_to_temperature(
            target_temp=max_op,
            initial_temp=initial_temp,
            steady_state_temp=steady_state_temp,
            time_constant=time_constant,
        )
        
        time_to_melt = self._time_to_temperature(
            target_temp=melt,
            initial_temp=initial_temp,
            steady_state_temp=steady_state_temp,
            time_constant=time_constant,
        )
        
        # Generate warnings and recommendations
        warnings = []
        recommendations = []
        
        if not is_safe:
            warnings.append(f"Steady-state temperature ({steady_state_temp:.1f}°C) exceeds safe limit ({max_op:.1f}°C)")
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            warnings.append(f"Risk level: {risk_level.value.upper()}")
        
        if temp_margin < 50:
            warnings.append(f"Low temperature margin ({temp_margin:.1f}°C)")
        
        # Recommendations
        if not is_safe:
            if "copper" not in material_props.composition:
                recommendations.append("Consider adding copper to improve thermal conductivity")
            recommendations.append("Increase heat sink surface area")
            recommendations.append("Add active cooling (forced convection)")
            
        if time_to_max_op is not None and time_to_max_op < 60:
            recommendations.append(f"Critical temperature reached in {time_to_max_op:.1f}s - add thermal mass")
        
        if safety_factor < 1.5:
            recommendations.append("Increase safety factor by reducing heat source power or improving cooling")
        
        # Calculate geometry lifetime and max temperature capacity
        max_temperature_capacity = min(max_op, melt * 0.9)  # 90% of melting point or max operating
        thermal_headroom = max(0, (max_temperature_capacity - steady_state_temp) / max_temperature_capacity * 100)
        
        # Geometry lifetime calculation
        # This is the time the geometry can safely operate before reaching critical temperature
        if steady_state_temp < max_op:
            # Geometry is stable - can operate indefinitely at steady state
            geometry_lifetime = float('inf')
            lifetime_limiting_factor = "Stable - steady state within safe limits"
        elif time_to_max_op is not None:
            # Limited by max operating temperature
            geometry_lifetime = time_to_max_op
            lifetime_limiting_factor = f"Max operating temperature ({max_op:.1f}°C)"
        elif time_to_melt is not None:
            # Limited by melting point
            geometry_lifetime = time_to_melt
            lifetime_limiting_factor = f"Melting point ({melt:.1f}°C)"
        else:
            # Cannot determine lifetime
            geometry_lifetime = None
            lifetime_limiting_factor = "Unable to determine - check heat source configuration"
        
        # Add warnings about lifetime
        if geometry_lifetime is not None and geometry_lifetime != float('inf'):
            if geometry_lifetime < 10:
                warnings.append(f"⚠️ CRITICAL: Geometry will fail in {geometry_lifetime:.1f} seconds!")
            elif geometry_lifetime < 60:
                warnings.append(f"Short operating time: Only {geometry_lifetime:.1f} seconds before failure")
            elif geometry_lifetime < 300:
                warnings.append(f"Limited operating time: {geometry_lifetime/60:.1f} minutes before max temperature")
        
        if time_to_melt is not None and time_to_melt < 600:
            warnings.append(f"🔥 MELTING WARNING: Material will start melting in {time_to_melt:.1f} seconds!")
        
        return ThermalLimitResult(
            max_operating_temp=max_op,
            melting_point=melt,
            steady_state_temp=steady_state_temp,
            max_transient_temp=max_transient_temp,
            time_to_steady_state=time_to_steady,
            temp_margin=temp_margin,
            safety_factor=safety_factor,
            risk_level=risk_level,
            is_safe=is_safe,
            time_to_max_operating=time_to_max_op,
            time_to_melting=time_to_melt,
            geometry_lifetime=geometry_lifetime,
            lifetime_limiting_factor=lifetime_limiting_factor,
            max_temperature_capacity=max_temperature_capacity,
            thermal_headroom=thermal_headroom,
            warnings=warnings,
            recommendations=recommendations,
        )
    
    def _time_to_temperature(
        self,
        target_temp: float,
        initial_temp: float,
        steady_state_temp: float,
        time_constant: float,
    ) -> Optional[float]:
        """
        Calculate time to reach a target temperature.
        
        Uses exponential approach model:
        T(t) = T_ss - (T_ss - T_0) * exp(-t/tau)
        
        Solving for t:
        t = -tau * ln((T_ss - T_target) / (T_ss - T_0))
        
        Returns None if target is never reached.
        """
        if steady_state_temp <= target_temp:
            # Target will never be reached
            return None
        
        if initial_temp >= target_temp:
            # Already at or above target
            return 0.0
        
        ratio = (steady_state_temp - target_temp) / (steady_state_temp - initial_temp)
        
        if ratio <= 0:
            return None
        
        return -time_constant * np.log(ratio)
    
    def quick_check(
        self,
        material_props: EffectiveProperties,
        max_expected_temp: float,
    ) -> Tuple[bool, str]:
        """
        Quick safety check without full analysis.
        
        Args:
            material_props: Material properties
            max_expected_temp: Maximum expected temperature
            
        Returns:
            Tuple of (is_safe, message)
        """
        max_op = material_props.max_operating_temp
        
        if max_expected_temp >= material_props.min_melting_point:
            return False, f"CRITICAL: Expected temp {max_expected_temp:.1f}°C exceeds melting point!"
        
        if max_expected_temp >= max_op:
            return False, f"UNSAFE: Expected temp {max_expected_temp:.1f}°C exceeds max operating {max_op:.1f}°C"
        
        margin = max_op - max_expected_temp
        if margin < 50:
            return True, f"MARGINAL: Only {margin:.1f}°C safety margin"
        
        return True, f"SAFE: {margin:.1f}°C safety margin"
