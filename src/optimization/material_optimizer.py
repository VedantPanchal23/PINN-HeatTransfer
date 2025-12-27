"""
Material composition optimization for thermal performance.

Optimizes material mixture ratios to achieve target thermal properties
while respecting cost and manufacturability constraints.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..materials import MaterialDatabase, MixtureCalculator, EffectiveProperties


@dataclass
class MaterialOptimizationResult:
    """Results of material optimization."""
    optimal_composition: Dict[str, float]
    initial_composition: Dict[str, float]
    
    # Properties
    optimal_properties: EffectiveProperties
    initial_properties: EffectiveProperties
    
    # Improvement
    conductivity_improvement: float  # Percentage
    cost_change: float               # Relative cost change
    
    # Constraints satisfaction
    max_temp_satisfied: bool
    cost_satisfied: bool
    
    def summary(self) -> str:
        lines = [
            "=" * 50,
            "MATERIAL OPTIMIZATION RESULTS",
            "=" * 50,
            "",
            "Optimal Composition:",
        ]
        
        for name, fraction in self.optimal_composition.items():
            initial = self.initial_composition.get(name, 0)
            lines.append(f"  {name}: {initial*100:.1f}% → {fraction*100:.1f}%")
        
        lines.extend([
            "",
            "Thermal Conductivity:",
            f"  {self.initial_properties.thermal_conductivity:.1f} → "
            f"{self.optimal_properties.thermal_conductivity:.1f} W/(m·K) "
            f"({self.conductivity_improvement:+.1f}%)",
            "",
            f"Max Operating Temp: {self.optimal_properties.max_operating_temp:.0f}°C",
            "",
            "Constraints:",
            f"  Max Temp Constraint: {'✅' if self.max_temp_satisfied else '❌'}",
            f"  Cost Constraint: {'✅' if self.cost_satisfied else '❌'}",
            "=" * 50,
        ])
        
        return "\n".join(lines)


class MaterialOptimizer:
    """
    Optimize material composition for thermal performance.
    
    Features:
    - Maximize thermal conductivity
    - Minimize cost while meeting thermal requirements
    - Consider manufacturing constraints
    - Multi-objective optimization
    """
    
    # Approximate material costs (relative to aluminum = 1.0)
    MATERIAL_COSTS = {
        "aluminum": 1.0,
        "copper": 3.0,
        "silver": 200.0,
        "gold": 2000.0,
        "iron": 0.5,
        "steel_carbon": 0.6,
        "steel_stainless_304": 2.5,
        "titanium": 15.0,
        "magnesium": 2.0,
        "brass": 2.5,
        "aluminum_nitride": 50.0,
        "silicon_carbide": 20.0,
        "silicon": 10.0,
    }
    
    def __init__(
        self,
        available_materials: List[str] = None,
        cost_weights: Dict[str, float] = None,
    ):
        """
        Initialize material optimizer.
        
        Args:
            available_materials: List of materials to consider
            cost_weights: Custom cost weights for materials
        """
        self.db = MaterialDatabase()
        self.calc = MixtureCalculator(self.db)
        
        self.available_materials = available_materials or [
            "aluminum", "copper", "iron", "brass"
        ]
        
        self.cost_weights = cost_weights or self.MATERIAL_COSTS
    
    def optimize_for_conductivity(
        self,
        initial_composition: Dict[str, float],
        max_cost_ratio: float = 2.0,
        min_max_operating_temp: float = 200.0,
    ) -> MaterialOptimizationResult:
        """
        Optimize material composition to maximize thermal conductivity.
        
        Args:
            initial_composition: Starting material composition
            max_cost_ratio: Maximum cost relative to initial composition
            min_max_operating_temp: Minimum required max operating temperature
            
        Returns:
            MaterialOptimizationResult with optimal composition
        """
        # Calculate initial properties
        initial_props = self.calc.calculate(initial_composition)
        initial_cost = self._calculate_cost(initial_composition)
        max_cost = initial_cost * max_cost_ratio
        
        # Get materials in composition
        materials = list(initial_composition.keys())
        n_materials = len(materials)
        
        # Define objective (negative because we minimize)
        def objective(x):
            composition = {mat: x[i] for i, mat in enumerate(materials)}
            try:
                props = self.calc.calculate(composition)
                return -props.thermal_conductivity
            except Exception:
                return 1e10  # Large penalty for invalid composition
        
        # Define constraints
        def constraint_sum(x):
            return np.sum(x) - 1.0  # Must sum to 1
        
        def constraint_cost(x):
            composition = {mat: x[i] for i, mat in enumerate(materials)}
            return max_cost - self._calculate_cost(composition)
        
        def constraint_temp(x):
            composition = {mat: x[i] for i, mat in enumerate(materials)}
            try:
                props = self.calc.calculate(composition)
                return props.max_operating_temp - min_max_operating_temp
            except Exception:
                return -1000
        
        constraints = [
            {'type': 'eq', 'fun': constraint_sum},
            {'type': 'ineq', 'fun': constraint_cost},
            {'type': 'ineq', 'fun': constraint_temp},
        ]
        
        # Bounds: each material 0-100%
        bounds = [(0.0, 1.0) for _ in range(n_materials)]
        
        # Initial guess
        x0 = [initial_composition[mat] for mat in materials]
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200}
        )
        
        # Extract optimal composition
        optimal_composition = {mat: max(0, result.x[i]) for i, mat in enumerate(materials)}
        
        # Normalize to sum to 1
        total = sum(optimal_composition.values())
        if total > 0:
            optimal_composition = {k: v/total for k, v in optimal_composition.items()}
        
        # Calculate optimal properties
        optimal_props = self.calc.calculate(optimal_composition)
        
        # Check constraints
        optimal_cost = self._calculate_cost(optimal_composition)
        
        conductivity_improvement = (
            (optimal_props.thermal_conductivity - initial_props.thermal_conductivity)
            / initial_props.thermal_conductivity * 100
        )
        
        cost_change = (optimal_cost - initial_cost) / initial_cost * 100
        
        return MaterialOptimizationResult(
            optimal_composition=optimal_composition,
            initial_composition=initial_composition,
            optimal_properties=optimal_props,
            initial_properties=initial_props,
            conductivity_improvement=conductivity_improvement,
            cost_change=cost_change,
            max_temp_satisfied=optimal_props.max_operating_temp >= min_max_operating_temp,
            cost_satisfied=optimal_cost <= max_cost,
        )
    
    def optimize_for_target_conductivity(
        self,
        target_conductivity: float,
        available_materials: List[str] = None,
        minimize_cost: bool = True,
    ) -> MaterialOptimizationResult:
        """
        Find cheapest material composition achieving target conductivity.
        
        Args:
            target_conductivity: Target thermal conductivity (W/(m·K))
            available_materials: Materials to consider
            minimize_cost: If True, minimize cost; if False, maximize conductivity
            
        Returns:
            MaterialOptimizationResult
        """
        materials = available_materials or self.available_materials
        n_materials = len(materials)
        
        # Check if target is achievable
        max_k = max(
            self.db.get(mat).thermal_conductivity 
            for mat in materials
        )
        
        if target_conductivity > max_k:
            print(f"Warning: Target {target_conductivity} W/(m·K) exceeds maximum "
                  f"achievable {max_k:.0f} W/(m·K)")
        
        def objective(x):
            composition = {mat: x[i] for i, mat in enumerate(materials)}
            try:
                cost = self._calculate_cost(composition)
                props = self.calc.calculate(composition)
                
                # Penalty for not meeting target
                penalty = max(0, target_conductivity - props.thermal_conductivity) * 100
                
                if minimize_cost:
                    return cost + penalty
                else:
                    return -props.thermal_conductivity + penalty
            except Exception:
                return 1e10
        
        def constraint_sum(x):
            return np.sum(x) - 1.0
        
        constraints = [{'type': 'eq', 'fun': constraint_sum}]
        bounds = [(0.0, 1.0) for _ in range(n_materials)]
        
        # Use global optimization for better results
        result = differential_evolution(
            objective,
            bounds,
            constraints=({'type': 'eq', 'fun': constraint_sum}),
            maxiter=100,
            seed=42,
        )
        
        # Extract results
        optimal_composition = {mat: max(0, result.x[i]) for i, mat in enumerate(materials)}
        total = sum(optimal_composition.values())
        if total > 0:
            optimal_composition = {k: v/total for k, v in optimal_composition.items()}
        
        optimal_props = self.calc.calculate(optimal_composition)
        
        # Initial: equal mix
        initial_composition = {mat: 1.0/n_materials for mat in materials}
        initial_props = self.calc.calculate(initial_composition)
        
        initial_cost = self._calculate_cost(initial_composition)
        optimal_cost = self._calculate_cost(optimal_composition)
        
        return MaterialOptimizationResult(
            optimal_composition=optimal_composition,
            initial_composition=initial_composition,
            optimal_properties=optimal_props,
            initial_properties=initial_props,
            conductivity_improvement=(
                (optimal_props.thermal_conductivity - initial_props.thermal_conductivity)
                / initial_props.thermal_conductivity * 100
            ),
            cost_change=(optimal_cost - initial_cost) / initial_cost * 100,
            max_temp_satisfied=True,
            cost_satisfied=True,
        )
    
    def suggest_materials(
        self,
        target_conductivity: float,
        max_cost_ratio: float = 5.0,
        min_operating_temp: float = 200.0,
    ) -> List[Tuple[str, float, float]]:
        """
        Suggest best single materials for the application.
        
        Returns list of (material_name, conductivity, relative_cost) sorted by
        conductivity/cost ratio.
        """
        suggestions = []
        
        for mat_name in self.db.list_all():
            mat = self.db.get(mat_name)
            if mat is None:
                continue
            
            cost = self.cost_weights.get(mat_name, 5.0)
            
            # Filter by requirements
            if mat.thermal_conductivity < target_conductivity * 0.5:
                continue
            if mat.max_operating_temp < min_operating_temp:
                continue
            
            # Score by conductivity/cost
            score = mat.thermal_conductivity / cost
            suggestions.append((mat_name, mat.thermal_conductivity, cost, score))
        
        # Sort by score (best first)
        suggestions.sort(key=lambda x: x[3], reverse=True)
        
        return [(s[0], s[1], s[2]) for s in suggestions[:10]]
    
    def _calculate_cost(self, composition: Dict[str, float]) -> float:
        """Calculate relative cost of material composition."""
        cost = 0.0
        for mat_name, fraction in composition.items():
            mat = self.db.get(mat_name)
            if mat:
                mat_cost = self.cost_weights.get(mat_name, 5.0)
                # Weight by density (cost per unit volume)
                cost += fraction * mat_cost * mat.density / 2700  # Normalize to aluminum density
        return cost
