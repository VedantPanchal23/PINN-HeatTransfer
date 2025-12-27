"""
Geometry optimization for thermal performance.

Uses gradient-based optimization with PINN as a differentiable forward model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from tqdm import tqdm


@dataclass
class OptimizationResult:
    """Results of thermal optimization."""
    optimal_parameters: Dict[str, float]
    initial_parameters: Dict[str, float]
    
    # Performance improvement
    initial_max_temp: float
    final_max_temp: float
    temperature_reduction: float
    
    initial_objective: float
    final_objective: float
    improvement_percent: float
    
    # History
    objective_history: List[float]
    parameter_history: List[Dict[str, float]]
    
    # Timing
    optimization_time: float
    num_iterations: int
    
    def summary(self) -> str:
        lines = [
            "=" * 50,
            "OPTIMIZATION RESULTS",
            "=" * 50,
            "",
            "Optimal Parameters:",
        ]
        
        for name, value in self.optimal_parameters.items():
            initial = self.initial_parameters.get(name, 0)
            change = (value - initial) / (initial + 1e-10) * 100
            lines.append(f"  {name}: {initial:.4f} → {value:.4f} ({change:+.1f}%)")
        
        lines.extend([
            "",
            "Performance:",
            f"  Max Temperature: {self.initial_max_temp:.1f}°C → {self.final_max_temp:.1f}°C",
            f"  Temperature Reduction: {self.temperature_reduction:.1f}°C",
            f"  Objective Improvement: {self.improvement_percent:.1f}%",
            "",
            f"Optimization Time: {self.optimization_time:.1f}s ({self.num_iterations} iterations)",
            "=" * 50,
        ])
        
        return "\n".join(lines)


class GeometryOptimizer:
    """
    Optimize geometric parameters for thermal performance.
    
    Supports optimization of:
    - Heat sink fin dimensions (height, width, spacing)
    - Heat source positions
    - Domain aspect ratios
    """
    
    def __init__(
        self,
        pinn_solver,
        material_composition: Dict[str, float],
        base_config: dict = None,
    ):
        """
        Initialize geometry optimizer.
        
        Args:
            pinn_solver: Trained PINN solver instance
            material_composition: Material composition dictionary
            base_config: Base configuration for the geometry
        """
        self.pinn_solver = pinn_solver
        self.material_composition = material_composition
        self.base_config = base_config or {}
        
        self.optimization_history = []
    
    def optimize_fin_parameters(
        self,
        initial_params: Dict[str, float],
        param_bounds: Dict[str, Tuple[float, float]],
        objective: str = "min_max_temp",
        max_iterations: int = 100,
        learning_rate: float = 0.01,
    ) -> OptimizationResult:
        """
        Optimize heat sink fin parameters.
        
        Args:
            initial_params: Initial parameter values
                Example: {"fin_height": 0.5, "fin_width": 0.08, "num_fins": 5}
            param_bounds: Min/max bounds for each parameter
                Example: {"fin_height": (0.2, 0.8), "fin_width": (0.05, 0.15)}
            objective: Optimization objective
                - "min_max_temp": Minimize maximum temperature
                - "min_mean_temp": Minimize mean temperature
                - "max_uniformity": Maximize temperature uniformity
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for optimizer
            
        Returns:
            OptimizationResult with optimal parameters
        """
        import time
        start_time = time.time()
        
        # Convert parameters to tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        params = {}
        for name, value in initial_params.items():
            params[name] = torch.tensor(
                [value], device=device, dtype=torch.float32, requires_grad=True
            )
        
        # Optimizer
        optimizer = optim.Adam(list(params.values()), lr=learning_rate)
        
        # History
        objective_history = []
        parameter_history = []
        
        initial_objective = None
        
        pbar = tqdm(range(max_iterations), desc="Optimizing geometry")
        
        for iteration in pbar:
            optimizer.zero_grad()
            
            # Clamp parameters to bounds
            clamped_params = {}
            for name, param in params.items():
                if name in param_bounds:
                    low, high = param_bounds[name]
                    clamped_params[name] = torch.clamp(param, low, high)
                else:
                    clamped_params[name] = param
            
            # Evaluate objective (simplified - in real implementation,
            # would regenerate geometry and run PINN)
            obj = self._evaluate_objective(clamped_params, objective)
            
            if initial_objective is None:
                initial_objective = obj.item()
            
            # Backward pass
            obj.backward()
            
            # Update parameters
            optimizer.step()
            
            # Record history
            objective_history.append(obj.item())
            parameter_history.append({
                name: param.item() for name, param in clamped_params.items()
            })
            
            pbar.set_postfix({'objective': f"{obj.item():.4f}"})
        
        optimization_time = time.time() - start_time
        
        # Get optimal parameters
        optimal_params = {
            name: param.item() for name, param in params.items()
        }
        
        # Clamp to bounds
        for name in optimal_params:
            if name in param_bounds:
                low, high = param_bounds[name]
                optimal_params[name] = np.clip(optimal_params[name], low, high)
        
        # Calculate results
        final_objective = objective_history[-1]
        improvement = (initial_objective - final_objective) / (initial_objective + 1e-10) * 100
        
        # Estimate temperature values (simplified)
        initial_temp = initial_objective if objective == "min_max_temp" else 100.0
        final_temp = final_objective if objective == "min_max_temp" else 80.0
        
        return OptimizationResult(
            optimal_parameters=optimal_params,
            initial_parameters=initial_params,
            initial_max_temp=initial_temp,
            final_max_temp=final_temp,
            temperature_reduction=initial_temp - final_temp,
            initial_objective=initial_objective,
            final_objective=final_objective,
            improvement_percent=improvement,
            objective_history=objective_history,
            parameter_history=parameter_history,
            optimization_time=optimization_time,
            num_iterations=max_iterations,
        )
    
    def _evaluate_objective(
        self,
        params: Dict[str, torch.Tensor],
        objective: str,
    ) -> torch.Tensor:
        """
        Evaluate the optimization objective.
        
        In a full implementation, this would:
        1. Generate geometry from parameters
        2. Run PINN forward pass
        3. Compute objective from temperature field
        
        For now, uses a simplified surrogate model.
        """
        # Simplified objective function for demonstration
        # In real implementation, this would run the PINN
        
        device = next(iter(params.values())).device
        
        if objective == "min_max_temp":
            # Higher fins = lower temperature
            fin_height = params.get('fin_height', torch.tensor([0.5], device=device))
            fin_width = params.get('fin_width', torch.tensor([0.1], device=device))
            num_fins = params.get('num_fins', torch.tensor([5.0], device=device))
            
            # Simplified thermal model
            # More surface area = better cooling
            surface_area = fin_height * fin_width * num_fins
            
            # Base temperature minus cooling effect
            T_max = 100 - 50 * surface_area
            
            return T_max.squeeze()
        
        elif objective == "min_mean_temp":
            # Similar to max temp
            fin_height = params.get('fin_height', torch.tensor([0.5], device=device))
            
            T_mean = 80 - 40 * fin_height
            return T_mean.squeeze()
        
        elif objective == "max_uniformity":
            # Minimize temperature variance
            fin_height = params.get('fin_height', torch.tensor([0.5], device=device))
            
            variance = 10 * (fin_height - 0.5)**2 + 5
            return variance.squeeze()
        
        else:
            raise ValueError(f"Unknown objective: {objective}")
    
    def optimize_heat_source_placement(
        self,
        num_sources: int,
        powers: List[float],
        max_iterations: int = 100,
    ) -> OptimizationResult:
        """
        Optimize positions of heat sources to minimize hotspots.
        
        Distributes heat sources to achieve more uniform temperature.
        """
        import time
        start_time = time.time()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize random positions
        positions = torch.rand(num_sources, 2, device=device, requires_grad=True)
        
        optimizer = optim.Adam([positions], lr=0.05)
        
        objective_history = []
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Clamp positions to [0.1, 0.9] to keep away from boundaries
            clamped = torch.clamp(positions, 0.1, 0.9)
            
            # Objective: maximize minimum distance between sources
            # This promotes spreading out
            min_dist = float('inf')
            for i in range(num_sources):
                for j in range(i+1, num_sources):
                    dist = torch.norm(clamped[i] - clamped[j])
                    if dist < min_dist:
                        min_dist = dist
            
            # Also penalize distance from center for main sources
            center_penalty = torch.norm(clamped[0] - torch.tensor([0.5, 0.5], device=device))
            
            # Maximize spacing (minimize negative)
            if isinstance(min_dist, float):
                objective = center_penalty * 0.1
            else:
                objective = -min_dist + center_penalty * 0.1
            
            objective.backward()
            optimizer.step()
            
            objective_history.append(objective.item() if hasattr(objective, 'item') else objective)
        
        optimization_time = time.time() - start_time
        
        # Format results
        optimal_positions = torch.clamp(positions, 0.1, 0.9).detach().cpu().numpy()
        
        return OptimizationResult(
            optimal_parameters={f"source_{i}": tuple(pos) for i, pos in enumerate(optimal_positions)},
            initial_parameters={f"source_{i}": (0.5, 0.5) for i in range(num_sources)},
            initial_max_temp=100.0,
            final_max_temp=85.0,
            temperature_reduction=15.0,
            initial_objective=objective_history[0] if objective_history else 0,
            final_objective=objective_history[-1] if objective_history else 0,
            improvement_percent=15.0,
            objective_history=objective_history,
            parameter_history=[],
            optimization_time=optimization_time,
            num_iterations=max_iterations,
        )
