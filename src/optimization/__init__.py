"""
Optimization module for thermal design.

Uses PINN as a differentiable surrogate for:
- Geometry optimization
- Material composition optimization
- Heat source placement optimization
"""

from .geometry_optimizer import GeometryOptimizer, OptimizationResult
from .material_optimizer import MaterialOptimizer

__all__ = [
    "GeometryOptimizer",
    "MaterialOptimizer",
    "OptimizationResult",
]
