"""
PINN Solver for 2D transient heat equation.

Solves the heat equation for a given geometry and generates
ground truth temperature fields for training the neural operator.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from tqdm import tqdm
import time

from .network import PINNNetwork
from .loss import HeatEquationLoss


@dataclass
class PINNConfig:
    """Configuration for PINN solver."""
    # Network architecture
    hidden_layers: list = None
    activation: str = "tanh"
    
    # Training
    max_iterations: int = 10000
    learning_rate: float = 0.001
    
    # Collocation points
    num_collocation: int = 10000
    num_boundary: int = 2000
    num_initial: int = 2000
    
    # Physics
    thermal_diffusivity: float = 0.1
    
    # Loss weights
    weight_pde: float = 1.0
    weight_bc: float = 10.0
    weight_ic: float = 10.0
    
    # Domain
    t_min: float = 0.0
    t_max: float = 1.0
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    
    # Device
    device: str = "cuda"
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 128, 128, 128]


class PINNSolver:
    """
    Physics-Informed Neural Network solver for 2D heat equation.
    
    Features:
    - Solves heat equation for arbitrary geometries
    - GPU-accelerated training
    - Supports custom heat sources
    - Generates full spatio-temporal temperature fields
    """
    
    def __init__(self, config: PINNConfig):
        """
        Initialize the PINN solver.
        
        Args:
            config: PINN configuration
        """
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize network
        self.model = PINNNetwork(
            hidden_layers=config.hidden_layers,
            activation=config.activation,
        ).to(self.device)
        
        # Initialize loss function
        self.loss_fn = HeatEquationLoss(
            alpha=config.thermal_diffusivity,
            weight_pde=config.weight_pde,
            weight_bc=config.weight_bc,
            weight_ic=config.weight_ic,
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        
        # Training history
        self.history = []
    
    def _sample_collocation_points(
        self,
        geometry_mask: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample collocation points for PDE residual computation.
        
        Args:
            geometry_mask: Optional binary mask to sample within
            
        Returns:
            (t, x, y) tensors of collocation points
        """
        n = self.config.num_collocation
        
        if geometry_mask is None:
            # Uniform sampling in domain
            t = torch.rand(n, device=self.device) * (
                self.config.t_max - self.config.t_min
            ) + self.config.t_min
            x = torch.rand(n, device=self.device) * (
                self.config.x_max - self.config.x_min
            ) + self.config.x_min
            y = torch.rand(n, device=self.device) * (
                self.config.y_max - self.config.y_min
            ) + self.config.y_min
        else:
            # Sample within geometry
            mask_coords = np.where(geometry_mask > 0.5)
            if len(mask_coords[0]) == 0:
                raise ValueError("Geometry mask is empty")
            
            # Normalize coordinates to [0, 1]
            h, w = geometry_mask.shape
            valid_y = mask_coords[0] / (h - 1)
            valid_x = mask_coords[1] / (w - 1)
            
            # Random sampling from valid points
            idx = np.random.choice(len(valid_x), size=n, replace=True)
            
            t = torch.rand(n, device=self.device) * (
                self.config.t_max - self.config.t_min
            ) + self.config.t_min
            x = torch.tensor(valid_x[idx], dtype=torch.float32, device=self.device)
            y = torch.tensor(valid_y[idx], dtype=torch.float32, device=self.device)
        
        return t, x, y
    
    def _sample_boundary_points(
        self,
        geometry_mask: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample boundary points for boundary condition enforcement.
        
        Args:
            geometry_mask: Optional binary mask to find boundary
            
        Returns:
            (t, x, y, T) tensors for boundary conditions
        """
        n = self.config.num_boundary
        
        if geometry_mask is None:
            # Sample from domain boundary (rectangular)
            n_per_side = n // 4
            
            t = torch.rand(n, device=self.device) * (
                self.config.t_max - self.config.t_min
            ) + self.config.t_min
            
            # Bottom
            x_bottom = torch.rand(n_per_side, device=self.device)
            y_bottom = torch.zeros(n_per_side, device=self.device)
            
            # Top
            x_top = torch.rand(n_per_side, device=self.device)
            y_top = torch.ones(n_per_side, device=self.device)
            
            # Left
            x_left = torch.zeros(n_per_side, device=self.device)
            y_left = torch.rand(n_per_side, device=self.device)
            
            # Right
            x_right = torch.ones(n_per_side, device=self.device)
            y_right = torch.rand(n_per_side, device=self.device)
            
            x = torch.cat([x_bottom, x_top, x_left, x_right])
            y = torch.cat([y_bottom, y_top, y_left, y_right])
            
            # Trim to exact size
            x = x[:n]
            y = y[:n]
            t = t[:len(x)]
        else:
            # Find boundary of geometry mask using gradient
            import cv2
            mask_uint8 = (geometry_mask * 255).astype(np.uint8)
            edges = cv2.Canny(mask_uint8, 100, 200)
            boundary_coords = np.where(edges > 0)
            
            if len(boundary_coords[0]) == 0:
                # Fallback: use erosion to find boundary
                kernel = np.ones((3, 3), np.uint8)
                eroded = cv2.erode(mask_uint8, kernel)
                boundary = mask_uint8 - eroded
                boundary_coords = np.where(boundary > 0)
            
            h, w = geometry_mask.shape
            valid_y = boundary_coords[0] / (h - 1)
            valid_x = boundary_coords[1] / (w - 1)
            
            # Sample from boundary
            idx = np.random.choice(
                len(valid_x), 
                size=min(n, len(valid_x)), 
                replace=len(valid_x) < n
            )
            
            t = torch.rand(len(idx), device=self.device) * (
                self.config.t_max - self.config.t_min
            ) + self.config.t_min
            x = torch.tensor(valid_x[idx], dtype=torch.float32, device=self.device)
            y = torch.tensor(valid_y[idx], dtype=torch.float32, device=self.device)
        
        # Dirichlet BC: T = 0 at boundary
        T_bc = torch.zeros_like(x)
        
        return t, x, y, T_bc
    
    def _sample_initial_points(
        self,
        geometry_mask: Optional[np.ndarray] = None,
        initial_condition: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample points for initial condition at t=0.
        
        Args:
            geometry_mask: Optional binary mask
            initial_condition: Function T_0(x, y) for initial temperature
            
        Returns:
            (x, y, T) tensors for initial condition
        """
        n = self.config.num_initial
        
        if geometry_mask is None:
            x = torch.rand(n, device=self.device)
            y = torch.rand(n, device=self.device)
        else:
            mask_coords = np.where(geometry_mask > 0.5)
            h, w = geometry_mask.shape
            valid_y = mask_coords[0] / (h - 1)
            valid_x = mask_coords[1] / (w - 1)
            
            idx = np.random.choice(len(valid_x), size=n, replace=True)
            x = torch.tensor(valid_x[idx], dtype=torch.float32, device=self.device)
            y = torch.tensor(valid_y[idx], dtype=torch.float32, device=self.device)
        
        # Initial condition
        if initial_condition is not None:
            T_ic = initial_condition(x, y)
        else:
            # Default: zero initial temperature
            T_ic = torch.zeros_like(x)
        
        return x, y, T_ic
    
    def create_heat_source(
        self,
        source_x: float,
        source_y: float,
        intensity: float = 5.0,
        sigma: float = 0.1,
    ) -> Callable:
        """
        Create a Gaussian heat source function.
        
        Args:
            source_x: X coordinate of heat source center
            source_y: Y coordinate of heat source center
            intensity: Heat source intensity
            sigma: Gaussian spread parameter
            
        Returns:
            Heat source function Q(t, x, y)
        """
        def heat_source(t, x, y):
            # Gaussian heat source
            r2 = (x - source_x)**2 + (y - source_y)**2
            return intensity * torch.exp(-r2 / (2 * sigma**2))
        
        return heat_source
    
    def train(
        self,
        geometry_mask: Optional[np.ndarray] = None,
        heat_source: Optional[Callable] = None,
        initial_condition: Optional[Callable] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Train the PINN to solve the heat equation.
        
        Args:
            geometry_mask: Binary mask defining the geometry
            heat_source: Heat source function Q(t, x, y)
            initial_condition: Initial temperature function T_0(x, y)
            verbose: Whether to show progress bar
            
        Returns:
            Dictionary with final training metrics
        """
        self.model.train()
        self.history = []
        
        start_time = time.time()
        
        iterator = tqdm(
            range(self.config.max_iterations),
            desc="Training PINN",
            disable=not verbose,
        )
        
        for iteration in iterator:
            self.optimizer.zero_grad()
            
            # Sample points (resample each iteration for better coverage)
            t_col, x_col, y_col = self._sample_collocation_points(geometry_mask)
            t_bc, x_bc, y_bc, T_bc = self._sample_boundary_points(geometry_mask)
            x_ic, y_ic, T_ic = self._sample_initial_points(
                geometry_mask, initial_condition
            )
            
            # Compute loss
            loss, loss_dict = self.loss_fn(
                self.model,
                (t_col, x_col, y_col),
                (t_bc, x_bc, y_bc, T_bc),
                (x_ic, y_ic, T_ic),
                heat_source,
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            self.history.append(loss_dict)
            
            if verbose and iteration % 100 == 0:
                iterator.set_postfix({
                    'loss': f"{loss_dict['total']:.6f}",
                    'pde': f"{loss_dict['pde']:.6f}",
                })
        
        training_time = time.time() - start_time
        
        return {
            'final_loss': self.history[-1]['total'],
            'training_time': training_time,
            'iterations': self.config.max_iterations,
        }
    
    @torch.no_grad()
    def predict(
        self,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Predict temperature at given coordinates.
        
        Args:
            t: Time values [N,] or scalar
            x: X coordinates [N,]
            y: Y coordinates [N,]
            
        Returns:
            Temperature values [N,]
        """
        self.model.eval()
        
        t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        # Broadcast t if scalar
        if t_tensor.dim() == 0:
            t_tensor = t_tensor.expand_as(x_tensor)
        
        T = self.model(t_tensor, x_tensor, y_tensor)
        
        return T.squeeze().cpu().numpy()
    
    @torch.no_grad()
    def generate_temperature_field(
        self,
        num_time_steps: int = 50,
        spatial_resolution: int = 128,
        geometry_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate full temperature field T(t, x, y).
        
        Args:
            num_time_steps: Number of time steps
            spatial_resolution: Spatial grid resolution
            geometry_mask: Optional mask (zeros outside geometry)
            
        Returns:
            Temperature field [T, H, W]
        """
        self.model.eval()
        
        # Create coordinate grids
        t_vals = np.linspace(
            self.config.t_min, self.config.t_max, num_time_steps
        )
        x_vals = np.linspace(
            self.config.x_min, self.config.x_max, spatial_resolution
        )
        y_vals = np.linspace(
            self.config.y_min, self.config.y_max, spatial_resolution
        )
        
        # Create meshgrid
        X, Y = np.meshgrid(x_vals, y_vals)
        x_flat = X.flatten()
        y_flat = Y.flatten()
        
        # Predict for each time step
        temperature_field = np.zeros(
            (num_time_steps, spatial_resolution, spatial_resolution)
        )
        
        for i, t in enumerate(t_vals):
            t_flat = np.full_like(x_flat, t)
            T = self.predict(t_flat, x_flat, y_flat)
            temperature_field[i] = T.reshape(spatial_resolution, spatial_resolution)
        
        # Apply geometry mask if provided
        if geometry_mask is not None:
            # Resize mask if needed
            if geometry_mask.shape != (spatial_resolution, spatial_resolution):
                import cv2
                geometry_mask = cv2.resize(
                    geometry_mask.astype(np.float32),
                    (spatial_resolution, spatial_resolution),
                    interpolation=cv2.INTER_NEAREST,
                )
            
            temperature_field = temperature_field * geometry_mask[np.newaxis, :, :]
        
        return temperature_field
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
        }, path)
    
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', [])
