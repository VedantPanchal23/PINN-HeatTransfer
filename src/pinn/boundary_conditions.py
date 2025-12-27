"""
Boundary conditions for PINN thermal simulations.

Implements various boundary condition types:
- Dirichlet (fixed temperature)
- Neumann (fixed heat flux)
- Robin/Convective (heat transfer coefficient)
- Mixed (different BCs on different boundaries)
- Periodic
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Union
from enum import Enum


class BCType(Enum):
    """Boundary condition types."""
    DIRICHLET = "dirichlet"      # T = T_specified
    NEUMANN = "neumann"          # -k * dT/dn = q
    ROBIN = "robin"              # -k * dT/dn = h * (T - T_inf)
    ADIABATIC = "adiabatic"      # dT/dn = 0 (special case of Neumann with q=0)
    PERIODIC = "periodic"        # T(x=0) = T(x=L)


class BoundaryLocation(Enum):
    """Predefined boundary locations for rectangular domains."""
    LEFT = "left"       # x = 0
    RIGHT = "right"     # x = 1
    BOTTOM = "bottom"   # y = 0
    TOP = "top"         # y = 1
    ALL = "all"         # All boundaries


@dataclass
class BoundaryCondition:
    """
    Definition of a single boundary condition.
    
    Attributes:
        bc_type: Type of boundary condition
        location: Boundary location
        value: BC value (temperature, flux, or tuple for Robin)
        thermal_conductivity: Material thermal conductivity (for flux BCs)
    """
    bc_type: BCType
    location: BoundaryLocation
    value: Union[float, Tuple[float, float], Callable] = 0.0
    thermal_conductivity: float = 1.0
    
    # For Robin BC
    heat_transfer_coeff: float = 10.0    # W/(m²·K)
    ambient_temp: float = 25.0            # °C
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "type": self.bc_type.value,
            "location": self.location.value,
            "value": self.value if not callable(self.value) else "function",
            "k": self.thermal_conductivity,
            "h": self.heat_transfer_coeff,
            "T_inf": self.ambient_temp,
        }


@dataclass
class BoundaryConditionSet:
    """
    Collection of boundary conditions for a simulation.
    
    Provides methods to:
    - Define BCs for each boundary
    - Generate BC points for training
    - Compute BC losses
    """
    conditions: Dict[BoundaryLocation, BoundaryCondition] = field(default_factory=dict)
    domain_bounds: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # (x_min, y_min, x_max, y_max)
    
    def set_dirichlet(
        self,
        location: BoundaryLocation,
        temperature: Union[float, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ) -> "BoundaryConditionSet":
        """
        Set Dirichlet (fixed temperature) boundary condition.
        
        Args:
            location: Boundary location
            temperature: Fixed temperature value or function T(x, y)
        """
        self.conditions[location] = BoundaryCondition(
            bc_type=BCType.DIRICHLET,
            location=location,
            value=temperature,
        )
        return self
    
    def set_neumann(
        self,
        location: BoundaryLocation,
        heat_flux: float,
        thermal_conductivity: float = 1.0,
    ) -> "BoundaryConditionSet":
        """
        Set Neumann (fixed heat flux) boundary condition.
        
        Args:
            location: Boundary location
            heat_flux: Heat flux value (W/m²), positive = into domain
            thermal_conductivity: Material thermal conductivity
        """
        self.conditions[location] = BoundaryCondition(
            bc_type=BCType.NEUMANN,
            location=location,
            value=heat_flux,
            thermal_conductivity=thermal_conductivity,
        )
        return self
    
    def set_adiabatic(self, location: BoundaryLocation) -> "BoundaryConditionSet":
        """Set adiabatic (insulated) boundary condition."""
        self.conditions[location] = BoundaryCondition(
            bc_type=BCType.ADIABATIC,
            location=location,
            value=0.0,
        )
        return self
    
    def set_convective(
        self,
        location: BoundaryLocation,
        heat_transfer_coeff: float,
        ambient_temp: float,
        thermal_conductivity: float = 1.0,
    ) -> "BoundaryConditionSet":
        """
        Set Robin (convective) boundary condition.
        
        -k * dT/dn = h * (T - T_inf)
        
        Args:
            location: Boundary location
            heat_transfer_coeff: Convective heat transfer coefficient (W/(m²·K))
            ambient_temp: Ambient temperature (°C)
            thermal_conductivity: Material thermal conductivity
        """
        self.conditions[location] = BoundaryCondition(
            bc_type=BCType.ROBIN,
            location=location,
            value=(heat_transfer_coeff, ambient_temp),
            thermal_conductivity=thermal_conductivity,
            heat_transfer_coeff=heat_transfer_coeff,
            ambient_temp=ambient_temp,
        )
        return self
    
    def set_all_dirichlet(self, temperature: float) -> "BoundaryConditionSet":
        """Set all boundaries to the same fixed temperature."""
        for loc in [BoundaryLocation.LEFT, BoundaryLocation.RIGHT,
                    BoundaryLocation.BOTTOM, BoundaryLocation.TOP]:
            self.set_dirichlet(loc, temperature)
        return self
    
    def set_all_convective(
        self,
        heat_transfer_coeff: float,
        ambient_temp: float,
        thermal_conductivity: float = 1.0,
    ) -> "BoundaryConditionSet":
        """Set all boundaries to convective."""
        for loc in [BoundaryLocation.LEFT, BoundaryLocation.RIGHT,
                    BoundaryLocation.BOTTOM, BoundaryLocation.TOP]:
            self.set_convective(loc, heat_transfer_coeff, ambient_temp, thermal_conductivity)
        return self
    
    def generate_boundary_points(
        self,
        num_points_per_boundary: int,
        t_range: Tuple[float, float] = (0.0, 1.0),
        device: torch.device = torch.device('cpu'),
    ) -> Dict[BoundaryLocation, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Generate boundary collocation points for training.
        
        Returns:
            Dictionary mapping location to (t, x, y) tensors
        """
        x_min, y_min, x_max, y_max = self.domain_bounds
        t_min, t_max = t_range
        
        points = {}
        
        for location in self.conditions.keys():
            t = torch.rand(num_points_per_boundary, device=device) * (t_max - t_min) + t_min
            
            if location == BoundaryLocation.LEFT:
                x = torch.full((num_points_per_boundary,), x_min, device=device)
                y = torch.rand(num_points_per_boundary, device=device) * (y_max - y_min) + y_min
            
            elif location == BoundaryLocation.RIGHT:
                x = torch.full((num_points_per_boundary,), x_max, device=device)
                y = torch.rand(num_points_per_boundary, device=device) * (y_max - y_min) + y_min
            
            elif location == BoundaryLocation.BOTTOM:
                x = torch.rand(num_points_per_boundary, device=device) * (x_max - x_min) + x_min
                y = torch.full((num_points_per_boundary,), y_min, device=device)
            
            elif location == BoundaryLocation.TOP:
                x = torch.rand(num_points_per_boundary, device=device) * (x_max - x_min) + x_min
                y = torch.full((num_points_per_boundary,), y_max, device=device)
            
            else:
                continue
            
            points[location] = (t, x, y)
        
        return points
    
    def get_normal_vector(self, location: BoundaryLocation) -> Tuple[float, float]:
        """Get outward normal vector for a boundary."""
        normals = {
            BoundaryLocation.LEFT: (-1.0, 0.0),
            BoundaryLocation.RIGHT: (1.0, 0.0),
            BoundaryLocation.BOTTOM: (0.0, -1.0),
            BoundaryLocation.TOP: (0.0, 1.0),
        }
        return normals.get(location, (0.0, 0.0))
    
    def summary(self) -> str:
        """Get human-readable summary."""
        lines = ["Boundary Conditions:"]
        
        for loc, bc in self.conditions.items():
            if bc.bc_type == BCType.DIRICHLET:
                val_str = f"T = {bc.value}°C" if not callable(bc.value) else "T = f(x,y)"
            elif bc.bc_type == BCType.NEUMANN:
                val_str = f"q = {bc.value} W/m²"
            elif bc.bc_type == BCType.ADIABATIC:
                val_str = "Insulated (q = 0)"
            elif bc.bc_type == BCType.ROBIN:
                val_str = f"h={bc.heat_transfer_coeff} W/(m²·K), T∞={bc.ambient_temp}°C"
            else:
                val_str = str(bc.value)
            
            lines.append(f"  {loc.value.capitalize()}: {bc.bc_type.value} - {val_str}")
        
        return "\n".join(lines)


class BoundaryConditionLoss(nn.Module):
    """
    Compute boundary condition losses for PINN training.
    
    Handles all BC types with proper gradient computation.
    """
    
    def __init__(
        self,
        bc_set: BoundaryConditionSet,
        weight_dirichlet: float = 10.0,
        weight_neumann: float = 10.0,
        weight_robin: float = 10.0,
    ):
        super().__init__()
        self.bc_set = bc_set
        self.weight_dirichlet = weight_dirichlet
        self.weight_neumann = weight_neumann
        self.weight_robin = weight_robin
    
    def compute_bc_loss(
        self,
        model: nn.Module,
        boundary_points: Dict[BoundaryLocation, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total boundary condition loss.
        
        Args:
            model: PINN network
            boundary_points: Dictionary of (t, x, y) for each boundary
            
        Returns:
            Total BC loss and dictionary of individual losses
        """
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        loss_dict = {}
        
        for location, (t, x, y) in boundary_points.items():
            bc = self.bc_set.conditions.get(location)
            if bc is None:
                continue
            
            if bc.bc_type == BCType.DIRICHLET:
                loss = self._dirichlet_loss(model, t, x, y, bc)
                total_loss = total_loss + self.weight_dirichlet * loss
                loss_dict[f"bc_{location.value}_dirichlet"] = loss.item()
            
            elif bc.bc_type == BCType.NEUMANN:
                loss = self._neumann_loss(model, t, x, y, bc, location)
                total_loss = total_loss + self.weight_neumann * loss
                loss_dict[f"bc_{location.value}_neumann"] = loss.item()
            
            elif bc.bc_type == BCType.ADIABATIC:
                loss = self._neumann_loss(model, t, x, y, bc, location)
                total_loss = total_loss + self.weight_neumann * loss
                loss_dict[f"bc_{location.value}_adiabatic"] = loss.item()
            
            elif bc.bc_type == BCType.ROBIN:
                loss = self._robin_loss(model, t, x, y, bc, location)
                total_loss = total_loss + self.weight_robin * loss
                loss_dict[f"bc_{location.value}_robin"] = loss.item()
        
        return total_loss, loss_dict
    
    def _dirichlet_loss(
        self,
        model: nn.Module,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        bc: BoundaryCondition,
    ) -> torch.Tensor:
        """Compute Dirichlet BC loss: (T - T_bc)² = 0."""
        T_pred = model(t, x, y)
        
        if callable(bc.value):
            T_bc = bc.value(x, y)
        else:
            T_bc = bc.value
        
        return torch.mean((T_pred.squeeze() - T_bc) ** 2)
    
    def _neumann_loss(
        self,
        model: nn.Module,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        bc: BoundaryCondition,
        location: BoundaryLocation,
    ) -> torch.Tensor:
        """Compute Neumann BC loss: -k * dT/dn = q."""
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        
        T = model(t, x, y)
        
        # Compute gradients
        grads = torch.autograd.grad(
            T, [x, y],
            grad_outputs=torch.ones_like(T),
            create_graph=True,
        )
        dT_dx = grads[0]
        dT_dy = grads[1]
        
        # Get normal direction
        nx, ny = self.bc_set.get_normal_vector(location)
        
        # Normal derivative
        dT_dn = nx * dT_dx + ny * dT_dy
        
        # Expected flux
        q = bc.value  # Prescribed flux
        k = bc.thermal_conductivity
        
        # -k * dT/dn = q  =>  dT/dn = -q/k
        expected_dT_dn = -q / k
        
        return torch.mean((dT_dn - expected_dT_dn) ** 2)
    
    def _robin_loss(
        self,
        model: nn.Module,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        bc: BoundaryCondition,
        location: BoundaryLocation,
    ) -> torch.Tensor:
        """Compute Robin BC loss: -k * dT/dn = h * (T - T_inf)."""
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        
        T = model(t, x, y)
        
        # Compute gradients
        grads = torch.autograd.grad(
            T, [x, y],
            grad_outputs=torch.ones_like(T),
            create_graph=True,
        )
        dT_dx = grads[0]
        dT_dy = grads[1]
        
        # Get normal direction
        nx, ny = self.bc_set.get_normal_vector(location)
        
        # Normal derivative
        dT_dn = nx * dT_dx + ny * dT_dy
        
        # Robin condition: -k * dT/dn = h * (T - T_inf)
        k = bc.thermal_conductivity
        h = bc.heat_transfer_coeff
        T_inf = bc.ambient_temp
        
        lhs = -k * dT_dn
        rhs = h * (T.squeeze() - T_inf)
        
        return torch.mean((lhs - rhs) ** 2)


# Convenience functions for common BC configurations
def create_heated_plate_bc(
    bottom_temp: float = 100.0,
    ambient_temp: float = 25.0,
    h_conv: float = 10.0,
    k: float = 50.0,
) -> BoundaryConditionSet:
    """
    Create BCs for a heated plate scenario:
    - Bottom: Fixed hot temperature (Dirichlet)
    - Top, Left, Right: Convective cooling (Robin)
    """
    bc = BoundaryConditionSet()
    bc.set_dirichlet(BoundaryLocation.BOTTOM, bottom_temp)
    bc.set_convective(BoundaryLocation.TOP, h_conv, ambient_temp, k)
    bc.set_convective(BoundaryLocation.LEFT, h_conv, ambient_temp, k)
    bc.set_convective(BoundaryLocation.RIGHT, h_conv, ambient_temp, k)
    return bc


def create_heat_sink_bc(
    base_temp: float = 80.0,
    ambient_temp: float = 25.0,
    h_conv: float = 25.0,
    k: float = 200.0,
) -> BoundaryConditionSet:
    """
    Create BCs for a heat sink scenario:
    - Bottom: Fixed base temperature from heat source
    - Other sides: Convective cooling
    """
    bc = BoundaryConditionSet()
    bc.set_dirichlet(BoundaryLocation.BOTTOM, base_temp)
    bc.set_convective(BoundaryLocation.TOP, h_conv, ambient_temp, k)
    bc.set_convective(BoundaryLocation.LEFT, h_conv, ambient_temp, k)
    bc.set_convective(BoundaryLocation.RIGHT, h_conv, ambient_temp, k)
    return bc


def create_insulated_box_bc(
    left_temp: float = 100.0,
    right_temp: float = 25.0,
) -> BoundaryConditionSet:
    """
    Create BCs for an insulated box with temperature gradient:
    - Left: Hot temperature
    - Right: Cold temperature  
    - Top/Bottom: Adiabatic (insulated)
    """
    bc = BoundaryConditionSet()
    bc.set_dirichlet(BoundaryLocation.LEFT, left_temp)
    bc.set_dirichlet(BoundaryLocation.RIGHT, right_temp)
    bc.set_adiabatic(BoundaryLocation.TOP)
    bc.set_adiabatic(BoundaryLocation.BOTTOM)
    return bc
