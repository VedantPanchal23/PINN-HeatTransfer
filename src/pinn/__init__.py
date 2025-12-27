# PINN Solver Module
from .solver import PINNSolver, PINNConfig
from .enhanced_solver import EnhancedPINNSolver, EnhancedPINNConfig, SimulationResult
from .network import PINNNetwork, MaterialAwarePINN, PINNNetworkWithGeometry, FourierFeatures
from .loss import HeatEquationLoss, GeometryMaskedLoss
from .boundary_conditions import (
    BoundaryCondition,
    BoundaryConditionSet,
    BoundaryConditionLoss,
    BoundaryLocation,
    BCType,
    create_heated_plate_bc,
    create_heat_sink_bc,
    create_insulated_box_bc,
)

__all__ = [
    # Solvers
    "PINNSolver",
    "PINNConfig",
    "EnhancedPINNSolver",
    "EnhancedPINNConfig",
    "SimulationResult",
    # Networks
    "PINNNetwork",
    "MaterialAwarePINN",
    "PINNNetworkWithGeometry",
    "FourierFeatures",
    # Loss functions
    "HeatEquationLoss",
    "GeometryMaskedLoss",
    # Boundary conditions
    "BoundaryCondition",
    "BoundaryConditionSet",
    "BoundaryConditionLoss",
    "BoundaryLocation",
    "BCType",
    "create_heated_plate_bc",
    "create_heat_sink_bc",
    "create_insulated_box_bc",
]
