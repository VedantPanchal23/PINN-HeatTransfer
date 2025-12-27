"""
Enhanced PINN Solver for 2D transient heat equation.

Integrates:
- Material properties from database
- Multiple boundary condition types
- Heat source configurations
- Thermal limit analysis
- Geometry-aware training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple, Callable, List, Union
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
import time
import json

from .network import PINNNetwork, MaterialAwarePINN, FourierFeatures
from .loss import HeatEquationLoss
from .boundary_conditions import (
    BoundaryConditionSet, 
    BoundaryConditionLoss,
    BoundaryLocation,
    BCType,
)
from ..materials import MaterialDatabase, MixtureCalculator, EffectiveProperties, ThermalLimitAnalyzer
from ..geometry import HeatSourceConfiguration, DomainInfo


@dataclass
class EnhancedPINNConfig:
    """Configuration for enhanced PINN solver."""
    # Network architecture
    hidden_layers: List[int] = field(default_factory=lambda: [128, 128, 128, 128])
    activation: str = "tanh"
    use_fourier_features: bool = True
    num_fourier_frequencies: int = 32
    fourier_scale: float = 1.0
    use_residual: bool = False
    
    # Training
    max_iterations: int = 10000
    learning_rate: float = 0.001
    lr_scheduler: str = "cosine"  # "cosine", "step", "none"
    lr_decay_steps: int = 2000
    lr_decay_rate: float = 0.5
    
    # Collocation points
    num_collocation: int = 10000
    num_boundary: int = 2000
    num_initial: int = 2000
    
    # Loss weights
    weight_pde: float = 1.0
    weight_bc: float = 10.0
    weight_ic: float = 10.0
    
    # Adaptive weighting
    use_adaptive_weights: bool = True
    adaptive_weight_freq: int = 100
    
    # Domain (normalized)
    t_min: float = 0.0
    t_max: float = 1.0
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    
    # Physical time scale (seconds)
    physical_time_max: float = 60.0
    
    # Device
    device: str = "cuda"
    
    # Logging
    log_frequency: int = 100
    save_frequency: int = 1000
    checkpoint_dir: str = "outputs/checkpoints"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "use_fourier_features": self.use_fourier_features,
            "max_iterations": self.max_iterations,
            "learning_rate": self.learning_rate,
            "num_collocation": self.num_collocation,
            "weight_pde": self.weight_pde,
            "weight_bc": self.weight_bc,
            "weight_ic": self.weight_ic,
            "physical_time_max": self.physical_time_max,
        }


@dataclass
class SimulationResult:
    """Results from PINN simulation."""
    # Temperature field
    temperature_field: np.ndarray      # [T, H, W]
    time_points: np.ndarray            # [T]
    x_coords: np.ndarray               # [H, W]
    y_coords: np.ndarray               # [H, W]
    
    # Analysis results
    max_temperature: float
    max_temp_location: Tuple[float, float]
    max_temp_time: float
    steady_state_temperature: float
    time_to_steady_state: float
    
    # Thermal limits
    is_safe: bool
    safety_margin: float
    thermal_limit_info: Optional[dict] = None
    
    # Training info
    training_time: float = 0.0
    final_loss: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary (without large arrays)."""
        return {
            "max_temperature": self.max_temperature,
            "max_temp_location": self.max_temp_location,
            "max_temp_time": self.max_temp_time,
            "steady_state_temperature": self.steady_state_temperature,
            "time_to_steady_state": self.time_to_steady_state,
            "is_safe": self.is_safe,
            "safety_margin": self.safety_margin,
            "training_time": self.training_time,
            "final_loss": self.final_loss,
        }


class EnhancedPINNSolver:
    """
    Enhanced Physics-Informed Neural Network solver for 2D heat equation.
    
    Features:
    - Material property integration
    - Multiple boundary condition types
    - Heat source configurations
    - Automatic thermal limit analysis
    - Adaptive loss weighting
    - Checkpoint saving and loading
    """
    
    def __init__(self, config: EnhancedPINNConfig):
        """
        Initialize the enhanced PINN solver.
        
        Args:
            config: Solver configuration
        """
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize network (will be set up in solve())
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        
        # Components (set during solve())
        self.material_props: Optional[EffectiveProperties] = None
        self.boundary_conditions: Optional[BoundaryConditionSet] = None
        self.heat_sources: Optional[HeatSourceConfiguration] = None
        self.domain_info: Optional[DomainInfo] = None
        
        # Training state
        self.loss_history: List[dict] = []
        self.current_iteration = 0
        
        # Adaptive weights
        self.adaptive_weights = {
            'pde': config.weight_pde,
            'bc': config.weight_bc,
            'ic': config.weight_ic,
        }
    
    def solve(
        self,
        material_composition: Dict[str, float],
        boundary_conditions: BoundaryConditionSet,
        heat_sources: Optional[HeatSourceConfiguration] = None,
        initial_temperature: float = 25.0,
        domain_info: Optional[DomainInfo] = None,
        progress_callback: Optional[Callable[[int, dict], None]] = None,
    ) -> SimulationResult:
        """
        Solve the heat equation using PINN.
        
        Args:
            material_composition: Dictionary of material names to volume fractions
                                 Example: {"aluminum": 0.7, "copper": 0.3}
            boundary_conditions: Boundary condition configuration
            heat_sources: Optional heat source configuration
            initial_temperature: Initial temperature (°C)
            domain_info: Optional geometry domain info
            progress_callback: Optional callback for progress updates
            
        Returns:
            SimulationResult with temperature field and analysis
        """
        start_time = time.time()
        
        # Calculate effective material properties
        calc = MixtureCalculator()
        self.material_props = calc.calculate(material_composition)
        print(f"\n{self.material_props.summary()}\n")
        
        # Store components
        self.boundary_conditions = boundary_conditions
        self.heat_sources = heat_sources
        self.domain_info = domain_info
        
        # Physical parameters
        alpha = self.material_props.thermal_diffusivity
        k = self.material_props.thermal_conductivity
        
        # Time scaling: normalize physical time to [0, 1]
        time_scale = self.config.physical_time_max
        
        # Normalized thermal diffusivity for scaled domain
        # α_normalized = α * time_scale / L²
        L = 0.1  # Assuming 10cm domain
        alpha_normalized = alpha * time_scale / (L ** 2)
        
        # Initialize network
        self.model = PINNNetwork(
            hidden_layers=self.config.hidden_layers,
            activation=self.config.activation,
            use_fourier_features=self.config.use_fourier_features,
            num_fourier_frequencies=self.config.num_fourier_frequencies,
            fourier_scale=self.config.fourier_scale,
            use_residual=self.config.use_residual,
        ).to(self.device)
        
        # Initialize loss function
        self.loss_fn = HeatEquationLoss(
            alpha=alpha_normalized,
            weight_pde=self.config.weight_pde,
            weight_bc=self.config.weight_bc,
            weight_ic=self.config.weight_ic,
        )
        
        # Initialize BC loss handler
        self.bc_loss_handler = BoundaryConditionLoss(
            bc_set=boundary_conditions,
            weight_dirichlet=self.config.weight_bc,
            weight_neumann=self.config.weight_bc,
            weight_robin=self.config.weight_bc,
        )
        
        # Update BCs with material conductivity
        for bc in boundary_conditions.conditions.values():
            bc.thermal_conductivity = k
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )
        
        # Initialize scheduler
        if self.config.lr_scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_iterations,
            )
        elif self.config.lr_scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_decay_steps,
                gamma=self.config.lr_decay_rate,
            )
        else:
            self.scheduler = None
        
        # Heat source function
        heat_source_fn = None
        if heat_sources is not None:
            heat_source_fn = heat_sources.get_source_function()
        
        # Training loop
        print(f"Training PINN on {self.device}...")
        print(f"Material: {material_composition}")
        print(f"Thermal diffusivity: {alpha:.2e} m²/s")
        print(f"Max simulation time: {self.config.physical_time_max} seconds\n")
        
        pbar = tqdm(range(self.config.max_iterations), desc="Training")
        
        for iteration in pbar:
            self.current_iteration = iteration
            
            # Generate collocation points
            t_col, x_col, y_col = self._sample_collocation_points()
            t_bc_points = boundary_conditions.generate_boundary_points(
                self.config.num_boundary // 4,
                t_range=(self.config.t_min, self.config.t_max),
                device=self.device,
            )
            x_ic, y_ic = self._sample_initial_points()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute PDE loss
            residual = self.loss_fn.pde_residual(
                self.model, t_col, x_col, y_col, heat_source_fn
            )
            loss_pde = torch.mean(residual ** 2)
            
            # Compute BC loss
            loss_bc, bc_losses = self.bc_loss_handler.compute_bc_loss(
                self.model, t_bc_points
            )
            
            # Compute IC loss
            t_ic = torch.zeros_like(x_ic)
            T_ic = torch.full_like(x_ic, initial_temperature)
            T_pred_ic = self.model(t_ic, x_ic, y_ic)
            loss_ic = torch.mean((T_pred_ic.squeeze() - T_ic) ** 2)
            
            # Total loss with adaptive weights
            total_loss = (
                self.adaptive_weights['pde'] * loss_pde +
                self.adaptive_weights['bc'] * loss_bc +
                self.adaptive_weights['ic'] * loss_ic
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Adaptive weight update
            if (self.config.use_adaptive_weights and 
                iteration > 0 and 
                iteration % self.config.adaptive_weight_freq == 0):
                self._update_adaptive_weights(loss_pde, loss_bc, loss_ic)
            
            # Logging
            loss_dict = {
                'total': total_loss.item(),
                'pde': loss_pde.item(),
                'bc': loss_bc.item(),
                'ic': loss_ic.item(),
            }
            self.loss_history.append(loss_dict)
            
            if iteration % self.config.log_frequency == 0:
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.2e}",
                    'pde': f"{loss_pde.item():.2e}",
                    'bc': f"{loss_bc.item():.2e}",
                    'ic': f"{loss_ic.item():.2e}",
                })
            
            # Progress callback
            if progress_callback is not None:
                progress_callback(iteration, loss_dict)
            
            # Checkpoint saving
            if (iteration > 0 and 
                iteration % self.config.save_frequency == 0):
                self._save_checkpoint(iteration)
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.1f} seconds")
        
        # Generate temperature field
        result = self._generate_results(
            initial_temperature=initial_temperature,
            training_time=training_time,
        )
        
        return result
    
    def _sample_collocation_points(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample random collocation points in the domain."""
        n = self.config.num_collocation
        
        t = torch.rand(n, device=self.device) * (
            self.config.t_max - self.config.t_min
        ) + self.config.t_min
        
        x = torch.rand(n, device=self.device) * (
            self.config.x_max - self.config.x_min
        ) + self.config.x_min
        
        y = torch.rand(n, device=self.device) * (
            self.config.y_max - self.config.y_min
        ) + self.config.y_min
        
        return t, x, y
    
    def _sample_initial_points(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample points for initial condition."""
        n = self.config.num_initial
        
        x = torch.rand(n, device=self.device) * (
            self.config.x_max - self.config.x_min
        ) + self.config.x_min
        
        y = torch.rand(n, device=self.device) * (
            self.config.y_max - self.config.y_min
        ) + self.config.y_min
        
        return x, y
    
    def _update_adaptive_weights(
        self,
        loss_pde: torch.Tensor,
        loss_bc: torch.Tensor,
        loss_ic: torch.Tensor,
    ) -> None:
        """Update adaptive loss weights based on gradient magnitudes."""
        # Simple approach: balance losses to similar magnitudes
        losses = {
            'pde': loss_pde.item(),
            'bc': loss_bc.item(),
            'ic': loss_ic.item(),
        }
        
        # Compute mean loss (with protection against zero)
        loss_values = list(losses.values())
        mean_loss = np.mean(loss_values)
        
        # Skip update if mean is effectively zero (all losses are zero)
        if mean_loss < 1e-12:
            return
        
        # Adjust weights inversely proportional to loss magnitude
        for key in self.adaptive_weights:
            if losses[key] > 1e-12:  # Avoid division by very small numbers
                ratio = mean_loss / losses[key]
                # Clamp ratio to prevent extreme weight changes
                ratio = np.clip(ratio, 0.1, 10.0)
                # Smooth update with momentum
                self.adaptive_weights[key] = 0.9 * self.adaptive_weights[key] + 0.1 * ratio
    
    def _generate_results(
        self,
        initial_temperature: float,
        training_time: float,
        resolution: int = 64,
        num_time_steps: int = 20,
    ) -> SimulationResult:
        """Generate temperature field and analyze results."""
        self.model.eval()
        
        # Create evaluation grid
        x = np.linspace(self.config.x_min, self.config.x_max, resolution)
        y = np.linspace(self.config.y_min, self.config.y_max, resolution)
        t = np.linspace(self.config.t_min, self.config.t_max, num_time_steps)
        
        X, Y = np.meshgrid(x, y)
        
        # Evaluate temperature field
        temperature_field = np.zeros((num_time_steps, resolution, resolution))
        
        with torch.no_grad():
            for i, t_val in enumerate(t):
                t_tensor = torch.full(
                    (resolution * resolution,), t_val, 
                    device=self.device, dtype=torch.float32
                )
                x_tensor = torch.tensor(
                    X.flatten(), device=self.device, dtype=torch.float32
                )
                y_tensor = torch.tensor(
                    Y.flatten(), device=self.device, dtype=torch.float32
                )
                
                T_pred = self.model(t_tensor, x_tensor, y_tensor)
                temperature_field[i] = T_pred.cpu().numpy().reshape(resolution, resolution)
        
        # Convert normalized time to physical time
        physical_times = t * self.config.physical_time_max
        
        # Analyze results
        max_temp = temperature_field.max()
        max_idx = np.unravel_index(temperature_field.argmax(), temperature_field.shape)
        max_temp_time = physical_times[max_idx[0]]
        # Ensure indices are within bounds for X, Y arrays
        y_idx = min(max_idx[1], resolution - 1)
        x_idx = min(max_idx[2], resolution - 1)
        max_temp_location = (X[y_idx, x_idx], Y[y_idx, x_idx])
        
        # Steady state analysis (last time step)
        steady_state_temp = temperature_field[-1].max()
        
        # Estimate time to steady state (99% of final value)
        # Handle both heating and cooling scenarios
        final_max = temperature_field[-1].max()
        initial_max = temperature_field[0].max()
        temp_change = final_max - initial_max
        time_to_steady = physical_times[-1]  # Default to final time
        
        if abs(temp_change) > 1e-10:
            target = initial_max + 0.99 * temp_change
            for i, t_phys in enumerate(physical_times):
                current_max = temperature_field[i].max()
                if temp_change > 0:  # Heating
                    if current_max >= target:
                        time_to_steady = t_phys
                        break
                else:  # Cooling
                    if current_max <= target:
                        time_to_steady = t_phys
                        break
        else:
            time_to_steady = 0.0  # Already at steady state
        
        # Thermal limit analysis
        analyzer = ThermalLimitAnalyzer()
        
        domain_area = 0.1 * 0.1  # 10cm x 10cm
        domain_thickness = 0.005  # 5mm
        heat_power = 0.0
        if self.heat_sources is not None:
            heat_power = getattr(self.heat_sources, 'total_power', 0.0) or 0.0
        
        limit_result = analyzer.analyze(
            material_props=self.material_props,
            heat_source_power=heat_power,
            domain_area=domain_area,
            domain_thickness=domain_thickness,
            ambient_temp=initial_temperature,
            predicted_temps=temperature_field,
            predicted_times=physical_times,
        )
        
        # Prepare result
        result = SimulationResult(
            temperature_field=temperature_field,
            time_points=physical_times,
            x_coords=X,
            y_coords=Y,
            max_temperature=max_temp,
            max_temp_location=max_temp_location,
            max_temp_time=max_temp_time,
            steady_state_temperature=steady_state_temp,
            time_to_steady_state=time_to_steady,
            is_safe=limit_result.is_safe,
            safety_margin=limit_result.temp_margin,
            thermal_limit_info=limit_result.__dict__,
            training_time=training_time,
            final_loss=self.loss_history[-1]['total'] if self.loss_history else 0.0,
            loss_history=[h['total'] for h in self.loss_history],
        )
        
        return result
    
    def _save_checkpoint(self, iteration: int) -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'loss_history': self.loss_history,
            'adaptive_weights': self.adaptive_weights,
        }
        
        path = checkpoint_dir / f"checkpoint_{iteration}.pt"
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        if self.model is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.current_iteration = checkpoint['iteration']
        self.loss_history = checkpoint['loss_history']
        self.adaptive_weights = checkpoint['adaptive_weights']
    
    def predict(
        self,
        t: Union[float, np.ndarray],
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
    ) -> np.ndarray:
        """
        Predict temperature at given points.
        
        Args:
            t: Time coordinate(s) (normalized 0-1)
            x: X coordinate(s) (normalized 0-1)
            y: Y coordinate(s) (normalized 0-1)
            
        Returns:
            Predicted temperature(s)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call solve() first.")
        
        self.model.eval()
        
        # Convert to tensors
        t = np.atleast_1d(t)
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        
        t_tensor = torch.tensor(t, device=self.device, dtype=torch.float32)
        x_tensor = torch.tensor(x, device=self.device, dtype=torch.float32)
        y_tensor = torch.tensor(y, device=self.device, dtype=torch.float32)
        
        with torch.no_grad():
            T_pred = self.model(t_tensor, x_tensor, y_tensor)
        
        return T_pred.cpu().numpy().squeeze()
