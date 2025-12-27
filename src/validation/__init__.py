"""
Validation module for comparing PINN results with traditional solvers.

Provides:
- Finite Difference Method (FDM) solver
- Analytical solutions for simple cases
- Comparison metrics
"""

from .fdm_solver import FDMSolver, FDMConfig
from .analytical import AnalyticalSolutions
from .comparison import ValidationComparison, ValidationMetrics

__all__ = [
    "FDMSolver",
    "FDMConfig",
    "AnalyticalSolutions",
    "ValidationComparison",
    "ValidationMetrics",
]
