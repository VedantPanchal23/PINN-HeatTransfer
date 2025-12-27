"""
Finite Difference Method (FDM) solver for 2D transient heat equation.

Used as a reference solution for validating PINN results.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, List
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm


@dataclass
class FDMConfig:
    """Configuration for FDM solver."""
    # Grid
    nx: int = 64               # Number of grid points in x
    ny: int = 64               # Number of grid points in y
    
    # Time
    dt: float = None           # Time step (auto if None)
    t_max: float = 60.0        # Maximum simulation time (seconds)
    
    # Domain
    Lx: float = 0.1            # Domain size in x (meters)
    Ly: float = 0.1            # Domain size in y (meters)
    
    # Method
    method: str = "implicit"   # "explicit", "implicit", "crank_nicolson"
    
    # Output
    output_times: int = 20     # Number of time snapshots to save


class FDMSolver:
    """
    2D transient heat equation solver using Finite Difference Method.
    
    Solves: ∂T/∂t = α * (∂²T/∂x² + ∂²T/∂y²) + Q(x, y, t)
    
    Supports:
    - Explicit, implicit, and Crank-Nicolson schemes
    - Dirichlet, Neumann, and mixed boundary conditions
    - Arbitrary heat sources
    """
    
    def __init__(self, config: FDMConfig):
        """
        Initialize FDM solver.
        
        Args:
            config: Solver configuration
        """
        self.config = config
        
        # Grid spacing
        self.dx = config.Lx / (config.nx - 1)
        self.dy = config.Ly / (config.ny - 1)
        
        # Create grid
        self.x = np.linspace(0, config.Lx, config.nx)
        self.y = np.linspace(0, config.Ly, config.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
    
    def solve(
        self,
        alpha: float,
        initial_temperature: float = 25.0,
        boundary_temperatures: dict = None,
        heat_source: Optional[Callable] = None,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the 2D transient heat equation.
        
        Args:
            alpha: Thermal diffusivity (m²/s)
            initial_temperature: Initial temperature (°C)
            boundary_temperatures: Dict with 'left', 'right', 'top', 'bottom' temperatures
            heat_source: Optional function Q(x, y, t) returning heat source term
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (temperature_field [T, ny, nx], time_points [T])
        """
        config = self.config
        
        # Default boundary conditions (Dirichlet)
        if boundary_temperatures is None:
            boundary_temperatures = {
                'left': initial_temperature,
                'right': initial_temperature,
                'top': initial_temperature,
                'bottom': initial_temperature,
            }
        
        # Stability-based time step for explicit method
        dx_min = min(self.dx, self.dy)
        dt_stable = 0.25 * dx_min**2 / alpha  # CFL condition
        
        if config.dt is None:
            if config.method == "explicit":
                self.dt = 0.5 * dt_stable  # Safety factor
            else:
                self.dt = dt_stable * 5  # Implicit methods are more stable
        else:
            self.dt = config.dt
        
        # Number of time steps
        nt = int(config.t_max / self.dt) + 1
        
        # Initialize temperature field
        T = np.full((config.ny, config.nx), initial_temperature, dtype=np.float64)
        
        # Apply boundary conditions
        T = self._apply_boundary_conditions(T, boundary_temperatures)
        
        # Storage for output
        output_interval = max(1, nt // config.output_times)
        T_history = [T.copy()]
        t_history = [0.0]
        
        # Solver coefficients
        rx = alpha * self.dt / self.dx**2
        ry = alpha * self.dt / self.dy**2
        
        # Choose solver method
        if config.method == "explicit":
            solver_step = lambda T, t: self._explicit_step(
                T, rx, ry, heat_source, t
            )
        elif config.method == "implicit":
            A = self._build_implicit_matrix(rx, ry)
            solver_step = lambda T, t: self._implicit_step(
                T, A, rx, ry, heat_source, t
            )
        elif config.method == "crank_nicolson":
            A = self._build_cn_matrix(rx, ry)
            solver_step = lambda T, t: self._crank_nicolson_step(
                T, A, rx, ry, heat_source, t
            )
        else:
            raise ValueError(f"Unknown method: {config.method}")
        
        # Time stepping
        iterator = range(1, nt)
        if show_progress:
            iterator = tqdm(iterator, desc=f"FDM ({config.method})")
        
        for n in iterator:
            t = n * self.dt
            
            # Solver step
            T = solver_step(T, t)
            
            # Apply boundary conditions
            T = self._apply_boundary_conditions(T, boundary_temperatures)
            
            # Store output
            if n % output_interval == 0 or n == nt - 1:
                T_history.append(T.copy())
                t_history.append(t)
        
        return np.array(T_history), np.array(t_history)
    
    def _apply_boundary_conditions(
        self,
        T: np.ndarray,
        bc: dict,
    ) -> np.ndarray:
        """Apply Dirichlet boundary conditions."""
        T[0, :] = bc.get('bottom', T[0, :])      # Bottom boundary
        T[-1, :] = bc.get('top', T[-1, :])       # Top boundary
        T[:, 0] = bc.get('left', T[:, 0])        # Left boundary
        T[:, -1] = bc.get('right', T[:, -1])     # Right boundary
        return T
    
    def _explicit_step(
        self,
        T: np.ndarray,
        rx: float,
        ry: float,
        heat_source: Optional[Callable],
        t: float,
    ) -> np.ndarray:
        """Explicit (FTCS) time stepping."""
        # Check CFL stability condition
        if rx + ry > 0.5:
            import warnings
            warnings.warn(f"CFL condition violated: rx={rx:.4f}, ry={ry:.4f}, rx+ry={rx+ry:.4f} > 0.5. Results may be unstable.")
        
        T_new = T.copy()
        
        # Interior points - use standard 5-point stencil Laplacian
        T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + \
            rx * (T[1:-1, 2:] - 2*T[1:-1, 1:-1] + T[1:-1, :-2]) + \
            ry * (T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1])
        
        # Add heat source (Q is volumetric generation rate in K/s)
        if heat_source is not None:
            Q = heat_source(self.X[1:-1, 1:-1], self.Y[1:-1, 1:-1], t)
            T_new[1:-1, 1:-1] += self.dt * Q
        
        return T_new
    
    def _build_implicit_matrix(
        self,
        rx: float,
        ry: float,
    ) -> sparse.csr_matrix:
        """Build sparse matrix for implicit method."""
        nx, ny = self.config.nx, self.config.ny
        N = (nx - 2) * (ny - 2)  # Interior points only
        
        # Coefficient matrix
        main_diag = (1 + 2*rx + 2*ry) * np.ones(N)
        x_off_diag = -rx * np.ones(N - 1)
        y_off_diag = -ry * np.ones(N - (nx - 2))
        
        # Handle boundary between rows
        for i in range(nx - 3, N - 1, nx - 2):
            x_off_diag[i] = 0
        
        diagonals = [main_diag, x_off_diag, x_off_diag, y_off_diag, y_off_diag]
        offsets = [0, -1, 1, -(nx-2), (nx-2)]
        
        A = sparse.diags(diagonals, offsets, format='csr')
        return A
    
    def _implicit_step(
        self,
        T: np.ndarray,
        A: sparse.csr_matrix,
        rx: float,
        ry: float,
        heat_source: Optional[Callable],
        t: float,
    ) -> np.ndarray:
        """Implicit (backward Euler) time stepping."""
        nx, ny = self.config.nx, self.config.ny
        
        # Build RHS vector
        b = T[1:-1, 1:-1].flatten()
        
        # Add heat source
        if heat_source is not None:
            Q = heat_source(self.X[1:-1, 1:-1], self.Y[1:-1, 1:-1], t)
            b += self.dt * Q.flatten()
        
        # Add boundary contributions
        # Left boundary
        b[::(nx-2)] += rx * T[1:-1, 0]
        # Right boundary
        b[(nx-3)::(nx-2)] += rx * T[1:-1, -1]
        # Bottom boundary
        b[:(nx-2)] += ry * T[0, 1:-1]
        # Top boundary
        b[-(nx-2):] += ry * T[-1, 1:-1]
        
        # Solve linear system
        T_interior = spsolve(A, b)
        
        # Update temperature field
        T_new = T.copy()
        T_new[1:-1, 1:-1] = T_interior.reshape(ny-2, nx-2)
        
        return T_new
    
    def _build_cn_matrix(
        self,
        rx: float,
        ry: float,
    ) -> sparse.csr_matrix:
        """Build sparse matrix for Crank-Nicolson method."""
        # Use half the coefficients for CN
        return self._build_implicit_matrix(rx/2, ry/2)
    
    def _crank_nicolson_step(
        self,
        T: np.ndarray,
        A: sparse.csr_matrix,
        rx: float,
        ry: float,
        heat_source: Optional[Callable],
        t: float,
    ) -> np.ndarray:
        """Crank-Nicolson time stepping."""
        nx, ny = self.config.nx, self.config.ny
        rx_half, ry_half = rx/2, ry/2
        
        # Explicit part (RHS)
        T_exp = T[1:-1, 1:-1] + \
            rx_half * (T[1:-1, 2:] - 2*T[1:-1, 1:-1] + T[1:-1, :-2]) + \
            ry_half * (T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1])
        
        b = T_exp.flatten()
        
        # Add heat source
        if heat_source is not None:
            Q = heat_source(self.X[1:-1, 1:-1], self.Y[1:-1, 1:-1], t)
            b += self.dt * Q.flatten()
        
        # Add boundary contributions
        b[::(nx-2)] += rx_half * T[1:-1, 0]
        b[(nx-3)::(nx-2)] += rx_half * T[1:-1, -1]
        b[:(nx-2)] += ry_half * T[0, 1:-1]
        b[-(nx-2):] += ry_half * T[-1, 1:-1]
        
        # Solve linear system
        T_interior = spsolve(A, b)
        
        # Update temperature field
        T_new = T.copy()
        T_new[1:-1, 1:-1] = T_interior.reshape(ny-2, nx-2)
        
        return T_new
    
    def get_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the computational grid."""
        return self.X, self.Y


def create_heat_source_function(
    positions: List[Tuple[float, float]],
    powers: List[float],
    sigma: float = 0.01,
    rho: float = 2700,
    cp: float = 900,
    thickness: float = 0.005,
) -> Callable:
    """
    Create a heat source function for multiple Gaussian sources.
    
    Args:
        positions: List of (x, y) positions in meters
        powers: List of powers in Watts
        sigma: Gaussian width in meters
        rho: Density (kg/m³)
        cp: Specific heat (J/(kg·K))
        thickness: Domain thickness (m)
        
    Returns:
        Function Q(x, y, t) returning volumetric heat generation (K/s)
    """
    def heat_source(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        Q = np.zeros_like(x)
        
        for (px, py), power in zip(positions, powers):
            dist2 = (x - px)**2 + (y - py)**2
            # Gaussian distribution
            gaussian = np.exp(-dist2 / (2 * sigma**2)) / (2 * np.pi * sigma**2)
            # Convert power to temperature rate
            # Q_vol = P / (rho * cp * V) where V is per unit area
            Q += power * gaussian / (rho * cp * thickness)
        
        return Q
    
    return heat_source
