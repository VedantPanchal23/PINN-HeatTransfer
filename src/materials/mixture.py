"""
Material mixture property calculator.

Calculates effective thermal properties for composite materials
using established mixing rules:
- Rule of Mixtures (upper bound)
- Inverse Rule of Mixtures (lower bound)
- Hashin-Shtrikman bounds (more accurate for composites)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

from .database import MaterialDatabase, Material, get_database


@dataclass
class EffectiveProperties:
    """Effective thermal properties for a material mixture."""
    thermal_conductivity: float      # W/(m·K)
    density: float                   # kg/m³
    specific_heat: float             # J/(kg·K)
    thermal_diffusivity: float       # m²/s
    max_operating_temp: float        # °C
    min_melting_point: float         # °C
    
    # Bounds for thermal conductivity
    k_lower_bound: float             # W/(m·K)
    k_upper_bound: float             # W/(m·K)
    
    # Original composition
    composition: Dict[str, float]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "thermal_conductivity": self.thermal_conductivity,
            "density": self.density,
            "specific_heat": self.specific_heat,
            "thermal_diffusivity": self.thermal_diffusivity,
            "max_operating_temp": self.max_operating_temp,
            "min_melting_point": self.min_melting_point,
            "k_bounds": (self.k_lower_bound, self.k_upper_bound),
            "composition": self.composition,
        }
    
    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "Effective Material Properties:",
            f"  Thermal Conductivity: {self.thermal_conductivity:.2f} W/(m·K)",
            f"    (bounds: {self.k_lower_bound:.2f} - {self.k_upper_bound:.2f})",
            f"  Density: {self.density:.2f} kg/m³",
            f"  Specific Heat: {self.specific_heat:.2f} J/(kg·K)",
            f"  Thermal Diffusivity: {self.thermal_diffusivity:.2e} m²/s",
            f"  Max Operating Temp: {self.max_operating_temp:.1f} °C",
            f"  Min Melting Point: {self.min_melting_point:.1f} °C",
            f"  Composition: {self.composition}",
        ]
        return "\n".join(lines)


class MixtureCalculator:
    """
    Calculate effective thermal properties for material mixtures.
    
    Supports multiple mixing models:
    - Volume-weighted average (rule of mixtures)
    - Mass-weighted average
    - Hashin-Shtrikman bounds
    
    Usage:
        calc = MixtureCalculator()
        props = calc.calculate({"aluminum": 0.7, "copper": 0.3})
    """
    
    def __init__(self, database: Optional[MaterialDatabase] = None):
        """
        Initialize mixture calculator.
        
        Args:
            database: Material database to use (uses global if not provided)
        """
        self.db = database or get_database()
    
    def calculate(
        self,
        composition: Dict[str, float],
        method: str = "hashin_shtrikman",
    ) -> EffectiveProperties:
        """
        Calculate effective properties for a material mixture.
        
        Args:
            composition: Dictionary mapping material names to volume fractions
                        Example: {"aluminum": 0.7, "copper": 0.3}
            method: Mixing method - "arithmetic", "harmonic", "hashin_shtrikman"
            
        Returns:
            EffectiveProperties object with calculated values
            
        Raises:
            ValueError: If composition doesn't sum to 1 or material not found
        """
        # Validate composition
        total = sum(composition.values())
        if not np.isclose(total, 1.0, atol=0.01):
            raise ValueError(
                f"Volume fractions must sum to 1.0, got {total:.3f}. "
                f"Composition: {composition}"
            )
        
        # Normalize to exactly 1.0
        composition = {k: v / total for k, v in composition.items()}
        
        # Get materials
        materials: Dict[str, Tuple[Material, float]] = {}
        for name, fraction in composition.items():
            material = self.db.get_or_raise(name)
            materials[name] = (material, fraction)
        
        # Calculate properties
        k_eff = self._calculate_thermal_conductivity(materials, method)
        k_lower, k_upper = self._calculate_conductivity_bounds(materials)
        rho_eff = self._calculate_density(materials)
        cp_eff = self._calculate_specific_heat(materials)
        alpha_eff = k_eff / (rho_eff * cp_eff)
        max_temp = self._calculate_max_operating_temp(materials)
        min_melt = self._calculate_min_melting_point(materials)
        
        return EffectiveProperties(
            thermal_conductivity=k_eff,
            density=rho_eff,
            specific_heat=cp_eff,
            thermal_diffusivity=alpha_eff,
            max_operating_temp=max_temp,
            min_melting_point=min_melt,
            k_lower_bound=k_lower,
            k_upper_bound=k_upper,
            composition=composition,
        )
    
    def _calculate_thermal_conductivity(
        self,
        materials: Dict[str, Tuple[Material, float]],
        method: str,
    ) -> float:
        """Calculate effective thermal conductivity."""
        if method == "arithmetic":
            # Rule of mixtures (upper bound, parallel model)
            return sum(
                mat.thermal_conductivity * frac
                for mat, frac in materials.values()
            )
        
        elif method == "harmonic":
            # Inverse rule of mixtures (lower bound, series model)
            inv_sum = sum(
                frac / mat.thermal_conductivity
                for mat, frac in materials.values()
            )
            return 1.0 / inv_sum
        
        elif method == "hashin_shtrikman":
            # Hashin-Shtrikman bounds - use average of bounds
            k_lower, k_upper = self._calculate_conductivity_bounds(materials)
            return (k_lower + k_upper) / 2
        
        else:
            raise ValueError(f"Unknown mixing method: {method}")
    
    def _calculate_conductivity_bounds(
        self,
        materials: Dict[str, Tuple[Material, float]],
    ) -> Tuple[float, float]:
        """
        Calculate Hashin-Shtrikman bounds for thermal conductivity.
        
        These are the tightest possible bounds for isotropic composites
        given only the volume fractions and constituent conductivities.
        """
        # Get conductivities and fractions
        k_values = [mat.thermal_conductivity for mat, _ in materials.values()]
        fractions = [frac for _, frac in materials.values()]
        
        k_min = min(k_values)
        k_max = max(k_values)
        
        # For binary mixture, use exact HS bounds
        if len(materials) == 2:
            items = list(materials.values())
            k1, f1 = items[0][0].thermal_conductivity, items[0][1]
            k2, f2 = items[1][0].thermal_conductivity, items[1][1]
            
            # If conductivities are nearly equal, use arithmetic mean
            if abs(k1 - k2) < 1e-10:
                k_avg = k1 * f1 + k2 * f2
                return k_avg, k_avg
            
            # Lower bound (matrix is lower conductivity material)
            if k1 < k2:
                km, ki = k1, k2
                fm, fi = f1, f2
            else:
                km, ki = k2, k1
                fm, fi = f2, f1
            
            # Avoid division by zero when km is very small
            if km < 1e-10:
                k_lower = km
            else:
                k_lower = km + fi / (1/(ki - km) + fm/(3*km))
            
            # Upper bound (matrix is higher conductivity material)
            if k1 > k2:
                km, ki = k1, k2
                fm, fi = f1, f2
            else:
                km, ki = k2, k1
                fm, fi = f2, f1
            
            # Avoid division by zero when km is very small
            if km < 1e-10:
                k_upper = ki * fi
            else:
                k_upper = km + fi / (1/(ki - km) + fm/(3*km))
            
            return min(k_lower, k_upper), max(k_lower, k_upper)
        
        # For multi-component, use simple bounds
        k_lower = 1.0 / sum(f / k for k, f in zip(k_values, fractions))
        k_upper = sum(k * f for k, f in zip(k_values, fractions))
        
        return k_lower, k_upper
    
    def _calculate_density(
        self,
        materials: Dict[str, Tuple[Material, float]],
    ) -> float:
        """Calculate effective density (volume-weighted average)."""
        return sum(
            mat.density * frac
            for mat, frac in materials.values()
        )
    
    def _calculate_specific_heat(
        self,
        materials: Dict[str, Tuple[Material, float]],
    ) -> float:
        """
        Calculate effective specific heat.
        
        For mixtures, use mass-weighted average of specific heats.
        """
        # First calculate mass fractions from volume fractions
        total_mass = sum(
            mat.density * frac
            for mat, frac in materials.values()
        )
        
        mass_fractions = {
            name: (mat.density * frac) / total_mass
            for name, (mat, frac) in materials.items()
        }
        
        # Mass-weighted specific heat
        return sum(
            mat.specific_heat * mass_fractions[name]
            for name, (mat, _) in materials.items()
        )
    
    def _calculate_max_operating_temp(
        self,
        materials: Dict[str, Tuple[Material, float]],
    ) -> float:
        """
        Calculate maximum operating temperature.
        
        Uses the minimum of all constituent max operating temps
        (weakest link principle).
        """
        return min(mat.max_operating_temp for mat, _ in materials.values())
    
    def _calculate_min_melting_point(
        self,
        materials: Dict[str, Tuple[Material, float]],
    ) -> float:
        """Calculate minimum melting point of constituents."""
        return min(mat.melting_point for mat, _ in materials.values())
    
    def quick_lookup(
        self,
        *materials_with_fractions: str,
    ) -> EffectiveProperties:
        """
        Quick lookup with string input.
        
        Usage:
            props = calc.quick_lookup("70% aluminum", "30% copper")
            
        Args:
            materials_with_fractions: Strings like "70% aluminum"
            
        Returns:
            EffectiveProperties object
        """
        composition = {}
        
        for item in materials_with_fractions:
            item = item.strip().lower()
            # Parse "70% aluminum" or "aluminum 70%"
            parts = item.replace("%", "").split()
            
            if len(parts) == 2:
                if parts[0].replace(".", "").isdigit():
                    fraction = float(parts[0]) / 100
                    material = parts[1]
                else:
                    material = parts[0]
                    fraction = float(parts[1]) / 100
            else:
                raise ValueError(f"Cannot parse '{item}'. Use format '70% aluminum'")
            
            composition[material] = fraction
        
        return self.calculate(composition)
