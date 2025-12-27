"""
Fourier Neural Operator (FNO) for learning heat transfer operators.

The FNO learns mappings between function spaces, making it ideal for
learning the solution operator of PDEs like the heat equation.

Reference: Li et al., "Fourier Neural Operator for Parametric Partial 
Differential Equations" (2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution Layer.
    
    Applies convolution in Fourier space by learning weights
    for the low-frequency Fourier modes.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ):
        """
        Initialize spectral convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes1: Number of Fourier modes in first dimension
            modes2: Number of Fourier modes in second dimension
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        # Learnable Fourier weights (complex-valued)
        self.scale = 1 / (in_channels * out_channels)
        
        # Weights for positive and negative frequencies
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
            )
        )
    
    def compl_mul2d(
        self, 
        input: torch.Tensor, 
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Complex multiplication in Fourier space.
        
        Args:
            input: Input tensor [B, C_in, M1, M2]
            weights: Weight tensor [C_in, C_out, M1, M2]
            
        Returns:
            Output tensor [B, C_out, M1, M2]
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral convolution.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C_out, H, W]
        """
        batch_size = x.shape[0]
        
        # Transform to Fourier space
        x_ft = torch.fft.rfft2(x)
        
        # Extract shape
        h, w = x.shape[-2], x.shape[-1]
        
        # Initialize output in Fourier space
        out_ft = torch.zeros(
            batch_size, self.out_channels, h, w // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        # Apply spectral convolution on low-frequency modes
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        
        # Transform back to physical space
        x = torch.fft.irfft2(out_ft, s=(h, w))
        
        return x


class FNOBlock(nn.Module):
    """
    Single FNO block consisting of spectral convolution and local linear layer.
    """
    
    def __init__(
        self,
        width: int,
        modes1: int,
        modes2: int,
        activation: str = "gelu",
    ):
        """
        Initialize FNO block.
        
        Args:
            width: Channel width
            modes1: Fourier modes in first dimension
            modes2: Fourier modes in second dimension
            activation: Activation function
        """
        super().__init__()
        
        self.spectral_conv = SpectralConv2d(width, width, modes1, modes2)
        self.local_conv = nn.Conv2d(width, width, 1)
        
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        # Spectral convolution path
        x1 = self.spectral_conv(x)
        
        # Local convolution path
        x2 = self.local_conv(x)
        
        # Add and apply activation
        x = self.activation(x1 + x2)
        
        return x


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator for thermal simulation.
    
    Architecture:
    1. Lift input to high-dimensional channel space
    2. Apply multiple FNO blocks (spectral convolutions)
    3. Project back to output dimension
    
    Input: (geometry_embedding, physics_params) + spatial coordinates
    Output: Temperature field T(t, x, y)
    """
    
    def __init__(
        self,
        modes: int = 16,
        width: int = 64,
        num_layers: int = 4,
        padding: int = 9,
        geometry_dim: int = 512,
        physics_dim: int = 4,  # (alpha, source_x, source_y, source_intensity)
        time_steps: int = 50,
        activation: str = "gelu",
    ):
        """
        Initialize FNO.
        
        Args:
            modes: Number of Fourier modes to use
            width: Channel width in FNO blocks
            padding: Padding for non-periodic boundary
            geometry_dim: Dimension of geometry embedding
            physics_dim: Number of physics parameters
            time_steps: Number of output time steps
            activation: Activation function
        """
        super().__init__()
        
        self.modes = modes
        self.width = width
        self.padding = padding
        self.time_steps = time_steps
        
        # Input channels: geometry + physics + coordinates
        # Geometry embedding is broadcasted to spatial grid
        # Physics params are also broadcasted
        # Coordinates: (x, y) = 2 channels
        input_dim = geometry_dim + physics_dim + 2
        
        # Lift to high-dimensional space
        self.fc0 = nn.Linear(input_dim, width)
        
        # FNO blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock(width, modes, modes, activation)
            for _ in range(num_layers)
        ])
        
        # Project to output (time steps)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, time_steps)
        
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
    
    def forward(
        self,
        geometry_embedding: torch.Tensor,
        physics_params: torch.Tensor,
        grid: Optional[torch.Tensor] = None,
        resolution: int = 128,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            geometry_embedding: Pre-computed geometry embedding [B, geometry_dim]
            physics_params: Physics parameters [B, physics_dim]
            grid: Optional spatial grid [B, H, W, 2]. Generated if None.
            resolution: Spatial resolution if grid not provided
            
        Returns:
            Temperature field [B, T, H, W]
        """
        batch_size = geometry_embedding.shape[0]
        device = geometry_embedding.device
        
        # Generate grid if not provided
        if grid is None:
            grid = self._get_grid(batch_size, resolution, device)
        
        h, w = grid.shape[1], grid.shape[2]
        
        # Broadcast geometry embedding to spatial grid
        geo_spatial = geometry_embedding.unsqueeze(1).unsqueeze(1)
        geo_spatial = geo_spatial.expand(-1, h, w, -1)
        
        # Broadcast physics params to spatial grid
        physics_spatial = physics_params.unsqueeze(1).unsqueeze(1)
        physics_spatial = physics_spatial.expand(-1, h, w, -1)
        
        # Concatenate all inputs
        x = torch.cat([geo_spatial, physics_spatial, grid], dim=-1)
        
        # Lift to FNO channel dimension
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Apply padding for non-periodic boundaries
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        
        # Apply FNO blocks
        for block in self.fno_blocks:
            x = block(x)
        
        # Remove padding
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]
        
        # Project to output
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.activation(self.fc1(x))
        x = self.fc2(x)  # [B, H, W, T]
        
        # Rearrange to [B, T, H, W]
        x = x.permute(0, 3, 1, 2)
        
        return x
    
    def _get_grid(
        self,
        batch_size: int,
        resolution: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate spatial coordinate grid.
        
        Args:
            batch_size: Batch size
            resolution: Spatial resolution
            device: Device to create tensor on
            
        Returns:
            Grid tensor [B, H, W, 2]
        """
        x = torch.linspace(0, 1, resolution, device=device)
        y = torch.linspace(0, 1, resolution, device=device)
        
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)
        
        # Expand for batch
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        return grid


class ConditionalFNO(nn.Module):
    """
    FNO conditioned on geometry via FiLM (Feature-wise Linear Modulation).
    
    This architecture uses the geometry embedding to modulate the features
    at each FNO layer, allowing for stronger geometry conditioning.
    """
    
    def __init__(
        self,
        modes: int = 16,
        width: int = 64,
        num_layers: int = 4,
        geometry_dim: int = 512,
        physics_dim: int = 4,
        time_steps: int = 50,
    ):
        """
        Initialize conditional FNO.
        
        Args:
            modes: Number of Fourier modes
            width: Channel width
            num_layers: Number of FNO layers
            geometry_dim: Geometry embedding dimension
            physics_dim: Physics parameters dimension
            time_steps: Number of output time steps
        """
        super().__init__()
        
        self.width = width
        self.time_steps = time_steps
        
        # Initial projection (from coordinates only)
        self.fc0 = nn.Linear(2, width)
        
        # FNO blocks
        self.spectral_convs = nn.ModuleList([
            SpectralConv2d(width, width, modes, modes)
            for _ in range(num_layers)
        ])
        
        self.local_convs = nn.ModuleList([
            nn.Conv2d(width, width, 1)
            for _ in range(num_layers)
        ])
        
        # FiLM generators for each layer
        self.film_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(geometry_dim + physics_dim, width * 2),
                nn.ReLU(),
                nn.Linear(width * 2, width * 2),  # gamma and beta
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, time_steps)
        
        self.activation = nn.GELU()
    
    def forward(
        self,
        geometry_embedding: torch.Tensor,
        physics_params: torch.Tensor,
        resolution: int = 128,
    ) -> torch.Tensor:
        """
        Forward pass with FiLM conditioning.
        
        Args:
            geometry_embedding: Geometry embedding [B, geometry_dim]
            physics_params: Physics parameters [B, physics_dim]
            resolution: Spatial resolution
            
        Returns:
            Temperature field [B, T, H, W]
        """
        batch_size = geometry_embedding.shape[0]
        device = geometry_embedding.device
        
        # Combine geometry and physics for conditioning
        condition = torch.cat([geometry_embedding, physics_params], dim=-1)
        
        # Create coordinate grid
        grid = self._get_grid(batch_size, resolution, device)
        
        # Initial lift
        x = self.fc0(grid)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Apply FNO layers with FiLM conditioning
        for spectral_conv, local_conv, film_gen in zip(
            self.spectral_convs, self.local_convs, self.film_generators
        ):
            # Generate FiLM parameters
            film_params = film_gen(condition)
            gamma = film_params[:, :self.width].view(-1, self.width, 1, 1)
            beta = film_params[:, self.width:].view(-1, self.width, 1, 1)
            
            # Spectral + local convolution
            x1 = spectral_conv(x)
            x2 = local_conv(x)
            x = x1 + x2
            
            # Apply FiLM modulation
            x = gamma * x + beta
            x = self.activation(x)
        
        # Project to output
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.activation(self.fc1(x))
        x = self.fc2(x)  # [B, H, W, T]
        
        # Rearrange to [B, T, H, W]
        x = x.permute(0, 3, 1, 2)
        
        return x
    
    def _get_grid(
        self,
        batch_size: int,
        resolution: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate spatial coordinate grid."""
        x = torch.linspace(0, 1, resolution, device=device)
        y = torch.linspace(0, 1, resolution, device=device)
        
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        return grid
