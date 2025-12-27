"""
Physics-based loss functions for the PINN solver.

Implements the 2D transient heat equation residual:
    ∂T/∂t - α * (∂²T/∂x² + ∂²T/∂y²) - Q = 0
"""

import torch
import torch.nn as nn
from typing import Tuple, Callable, Optional


class HeatEquationLoss(nn.Module):
    """
    Computes the physics-informed loss for the 2D heat equation.
    
    Loss = w_pde * L_pde + w_bc * L_bc + w_ic * L_ic
    
    Where:
        L_pde: PDE residual loss at collocation points
        L_bc: Boundary condition loss
        L_ic: Initial condition loss
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        weight_pde: float = 1.0,
        weight_bc: float = 10.0,
        weight_ic: float = 10.0,
    ):
        """
        Initialize the loss function.
        
        Args:
            alpha: Thermal diffusivity
            weight_pde: Weight for PDE residual loss
            weight_bc: Weight for boundary condition loss
            weight_ic: Weight for initial condition loss
        """
        super().__init__()
        
        self.alpha = alpha
        self.weight_pde = weight_pde
        self.weight_bc = weight_bc
        self.weight_ic = weight_ic
    
    def pde_residual(
        self,
        model: nn.Module,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        heat_source: Optional[Callable] = None,
        geometry_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the PDE residual using automatic differentiation.
        
        Residual: ∂T/∂t - α * (∂²T/∂x² + ∂²T/∂y²) - Q
        
        Args:
            model: PINN network
            t: Time coordinates [N,]
            x: X coordinates [N,]
            y: Y coordinates [N,]
            heat_source: Optional heat source function Q(t, x, y)
            geometry_embedding: Optional geometry embedding for conditioned PINN
            
        Returns:
            PDE residual [N,]
        """
        # Enable gradients for input coordinates
        t = t.requires_grad_(True)
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        
        # Forward pass
        if geometry_embedding is not None:
            T = model(t, x, y, geometry_embedding)
        else:
            T = model(t, x, y)
        
        # Compute gradients using autograd
        # First derivatives
        grads = torch.autograd.grad(
            T, [t, x, y],
            grad_outputs=torch.ones_like(T),
            create_graph=True,
            retain_graph=True,
        )
        
        dT_dt = grads[0]
        dT_dx = grads[1]
        dT_dy = grads[2]
        
        # Second derivatives
        d2T_dx2 = torch.autograd.grad(
            dT_dx, x,
            grad_outputs=torch.ones_like(dT_dx),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        d2T_dy2 = torch.autograd.grad(
            dT_dy, y,
            grad_outputs=torch.ones_like(dT_dy),
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Laplacian
        laplacian = d2T_dx2 + d2T_dy2
        
        # Heat source term
        if heat_source is not None:
            Q = heat_source(t, x, y)
        else:
            Q = torch.zeros_like(T)
        
        # PDE residual: ∂T/∂t - α * ∇²T - Q = 0
        residual = dT_dt - self.alpha * laplacian - Q
        
        return residual.squeeze()
    
    def boundary_loss(
        self,
        model: nn.Module,
        t_bc: torch.Tensor,
        x_bc: torch.Tensor,
        y_bc: torch.Tensor,
        T_bc: torch.Tensor,
        geometry_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute boundary condition loss.
        
        Args:
            model: PINN network
            t_bc: Time coordinates at boundary [N,]
            x_bc: X coordinates at boundary [N,]
            y_bc: Y coordinates at boundary [N,]
            T_bc: Prescribed temperature at boundary [N,]
            geometry_embedding: Optional geometry embedding
            
        Returns:
            Boundary loss scalar
        """
        if geometry_embedding is not None:
            T_pred = model(t_bc, x_bc, y_bc, geometry_embedding)
        else:
            T_pred = model(t_bc, x_bc, y_bc)
        
        return torch.mean((T_pred.squeeze() - T_bc) ** 2)
    
    def initial_loss(
        self,
        model: nn.Module,
        x_ic: torch.Tensor,
        y_ic: torch.Tensor,
        T_ic: torch.Tensor,
        geometry_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute initial condition loss.
        
        Args:
            model: PINN network
            x_ic: X coordinates at t=0 [N,]
            y_ic: Y coordinates at t=0 [N,]
            T_ic: Initial temperature distribution [N,]
            geometry_embedding: Optional geometry embedding
            
        Returns:
            Initial condition loss scalar
        """
        t_ic = torch.zeros_like(x_ic)
        
        if geometry_embedding is not None:
            T_pred = model(t_ic, x_ic, y_ic, geometry_embedding)
        else:
            T_pred = model(t_ic, x_ic, y_ic)
        
        return torch.mean((T_pred.squeeze() - T_ic) ** 2)
    
    def forward(
        self,
        model: nn.Module,
        collocation_points: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        boundary_points: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        initial_points: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        heat_source: Optional[Callable] = None,
        geometry_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total physics-informed loss.
        
        Args:
            model: PINN network
            collocation_points: (t, x, y) for PDE residual
            boundary_points: (t, x, y, T) for boundary conditions
            initial_points: (x, y, T) for initial condition
            heat_source: Optional heat source function
            geometry_embedding: Optional geometry embedding
            
        Returns:
            Total loss and dictionary of individual losses
        """
        t_col, x_col, y_col = collocation_points
        t_bc, x_bc, y_bc, T_bc = boundary_points
        x_ic, y_ic, T_ic = initial_points
        
        # PDE residual loss
        residual = self.pde_residual(
            model, t_col, x_col, y_col, 
            heat_source, geometry_embedding
        )
        loss_pde = torch.mean(residual ** 2)
        
        # Boundary loss
        loss_bc = self.boundary_loss(
            model, t_bc, x_bc, y_bc, T_bc, 
            geometry_embedding
        )
        
        # Initial condition loss
        loss_ic = self.initial_loss(
            model, x_ic, y_ic, T_ic,
            geometry_embedding
        )
        
        # Total weighted loss
        total_loss = (
            self.weight_pde * loss_pde +
            self.weight_bc * loss_bc +
            self.weight_ic * loss_ic
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'pde': loss_pde.item(),
            'bc': loss_bc.item(),
            'ic': loss_ic.item(),
        }
        
        return total_loss, loss_dict


class GeometryMaskedLoss(HeatEquationLoss):
    """
    Heat equation loss that respects geometry boundaries.
    
    Only computes PDE residual inside the geometry mask,
    and applies boundary conditions at the geometry boundary.
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        weight_pde: float = 1.0,
        weight_bc: float = 10.0,
        weight_ic: float = 10.0,
        weight_geometry: float = 5.0,
    ):
        super().__init__(alpha, weight_pde, weight_bc, weight_ic)
        self.weight_geometry = weight_geometry
    
    def geometry_constraint_loss(
        self,
        model: nn.Module,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        inside_mask: torch.Tensor,
        geometry_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Enforce zero temperature outside geometry.
        
        Args:
            model: PINN network
            t: Time coordinates [N,]
            x: X coordinates [N,]
            y: Y coordinates [N,]
            inside_mask: Boolean mask indicating points inside geometry [N,]
            geometry_embedding: Optional geometry embedding
            
        Returns:
            Geometry constraint loss
        """
        outside_mask = ~inside_mask
        
        if not outside_mask.any():
            return torch.tensor(0.0, device=t.device)
        
        t_out = t[outside_mask]
        x_out = x[outside_mask]
        y_out = y[outside_mask]
        
        if geometry_embedding is not None:
            T_pred = model(t_out, x_out, y_out, geometry_embedding)
        else:
            T_pred = model(t_out, x_out, y_out)
        
        # Temperature should be zero outside geometry
        return torch.mean(T_pred ** 2)
