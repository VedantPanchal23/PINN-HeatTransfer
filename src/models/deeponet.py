"""
DeepONet (Deep Operator Network) for thermal simulation.

DeepONet learns operators by decomposing them into two networks:
- Branch network: Encodes the input function (geometry + physics)
- Trunk network: Encodes the output coordinates (t, x, y)

Reference: Lu et al., "Learning nonlinear operators via DeepONet" (2021)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class BranchNetwork(nn.Module):
    """
    Branch network for encoding input functions.
    
    Takes geometry embedding and physics parameters as input.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int,
        activation: str = "relu",
    ):
        """
        Initialize branch network.
        
        Args:
            input_dim: Input dimension (geometry_dim + physics_dim)
            hidden_layers: List of hidden layer sizes
            output_dim: Output dimension (basis function count)
            activation: Activation function
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "silu":
                layers.append(nn.SiLU())
            
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [B, input_dim]
            
        Returns:
            Branch output [B, output_dim]
        """
        return self.network(x)


class TrunkNetwork(nn.Module):
    """
    Trunk network for encoding output coordinates.
    
    Takes (t, x, y) coordinates as input.
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # (t, x, y)
        hidden_layers: List[int] = [128, 128, 128],
        output_dim: int = 128,
        activation: str = "tanh",
    ):
        """
        Initialize trunk network.
        
        Args:
            input_dim: Number of coordinate dimensions
            hidden_layers: List of hidden layer sizes
            output_dim: Output dimension (must match branch output)
            activation: Activation function
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "silu":
                layers.append(nn.SiLU())
            
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            coords: Coordinates [N, input_dim] or [B, N, input_dim]
            
        Returns:
            Trunk output [N, output_dim] or [B, N, output_dim]
        """
        return self.network(coords)


class DeepONet(nn.Module):
    """
    DeepONet for learning the heat transfer operator.
    
    The operator maps:
    (geometry, physics) -> T(t, x, y)
    
    Architecture:
    - Branch: Encodes geometry embedding + physics parameters
    - Trunk: Encodes (t, x, y) coordinates
    - Output: Dot product of branch and trunk outputs
    """
    
    def __init__(
        self,
        geometry_dim: int = 512,
        physics_dim: int = 4,
        branch_layers: List[int] = [512, 256, 256, 128],
        trunk_layers: List[int] = [128, 128, 128],
        basis_dim: int = 128,
        activation: str = "relu",
    ):
        """
        Initialize DeepONet.
        
        Args:
            geometry_dim: Dimension of geometry embedding
            physics_dim: Number of physics parameters
            branch_layers: Hidden layers for branch network
            trunk_layers: Hidden layers for trunk network
            basis_dim: Number of basis functions (output dim of both networks)
            activation: Activation function
        """
        super().__init__()
        
        self.geometry_dim = geometry_dim
        self.physics_dim = physics_dim
        self.basis_dim = basis_dim
        
        # Branch network
        branch_input_dim = geometry_dim + physics_dim
        self.branch = BranchNetwork(
            branch_input_dim, branch_layers, basis_dim, activation
        )
        
        # Trunk network
        self.trunk = TrunkNetwork(
            input_dim=3,  # (t, x, y)
            hidden_layers=trunk_layers,
            output_dim=basis_dim,
            activation="tanh",  # Tanh often works better for trunk
        )
        
        # Learnable bias
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        geometry_embedding: torch.Tensor,
        physics_params: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            geometry_embedding: Geometry embedding [B, geometry_dim]
            physics_params: Physics parameters [B, physics_dim]
            coords: Query coordinates [N, 3] or [B, N, 3] for (t, x, y)
            
        Returns:
            Temperature values [B, N]
        """
        batch_size = geometry_embedding.shape[0]
        
        # Combine geometry and physics for branch input
        branch_input = torch.cat([geometry_embedding, physics_params], dim=-1)
        
        # Branch output [B, basis_dim]
        branch_out = self.branch(branch_input)
        
        # Handle different coordinate formats
        if coords.dim() == 2:
            # Same coordinates for all batch items
            num_points = coords.shape[0]
            trunk_out = self.trunk(coords)  # [N, basis_dim]
            
            # Expand for batch multiplication
            trunk_out = trunk_out.unsqueeze(0).expand(batch_size, -1, -1)
            branch_out = branch_out.unsqueeze(1)  # [B, 1, basis_dim]
            
            # Dot product: [B, N]
            output = torch.sum(branch_out * trunk_out, dim=-1) + self.bias
        else:
            # Different coordinates per batch item
            trunk_out = self.trunk(coords)  # [B, N, basis_dim]
            branch_out = branch_out.unsqueeze(1)  # [B, 1, basis_dim]
            
            # Dot product
            output = torch.sum(branch_out * trunk_out, dim=-1) + self.bias
        
        return output
    
    def predict_field(
        self,
        geometry_embedding: torch.Tensor,
        physics_params: torch.Tensor,
        time_steps: int = 50,
        resolution: int = 128,
    ) -> torch.Tensor:
        """
        Predict full temperature field T(t, x, y).
        
        Args:
            geometry_embedding: Geometry embedding [B, geometry_dim]
            physics_params: Physics parameters [B, physics_dim]
            time_steps: Number of time steps
            resolution: Spatial resolution
            
        Returns:
            Temperature field [B, T, H, W]
        """
        batch_size = geometry_embedding.shape[0]
        device = geometry_embedding.device
        
        # Create coordinate grid
        t_vals = torch.linspace(0, 1, time_steps, device=device)
        x_vals = torch.linspace(0, 1, resolution, device=device)
        y_vals = torch.linspace(0, 1, resolution, device=device)
        
        # Create full 3D grid
        T_grid, X_grid, Y_grid = torch.meshgrid(t_vals, x_vals, y_vals, indexing='ij')
        coords = torch.stack([
            T_grid.flatten(),
            X_grid.flatten(),
            Y_grid.flatten(),
        ], dim=-1)  # [T*H*W, 3]
        
        # Predict
        output = self(geometry_embedding, physics_params, coords)
        
        # Reshape to field
        output = output.view(batch_size, time_steps, resolution, resolution)
        
        return output


class ImprovedDeepONet(nn.Module):
    """
    Improved DeepONet with additional features:
    - Multi-head attention between branch and trunk
    - Skip connections
    - Separate time and space encoding
    """
    
    def __init__(
        self,
        geometry_dim: int = 512,
        physics_dim: int = 4,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
    ):
        """
        Initialize improved DeepONet.
        
        Args:
            geometry_dim: Dimension of geometry embedding
            physics_dim: Number of physics parameters
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Branch network with residual blocks
        self.branch_proj = nn.Linear(geometry_dim + physics_dim, hidden_dim)
        
        self.branch_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        
        # Separate trunk networks for time and space
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.space_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self,
        geometry_embedding: torch.Tensor,
        physics_params: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            geometry_embedding: [B, geometry_dim]
            physics_params: [B, physics_dim]
            coords: [N, 3] or [B, N, 3]
            
        Returns:
            Temperature [B, N]
        """
        batch_size = geometry_embedding.shape[0]
        
        # Branch encoding
        branch_input = torch.cat([geometry_embedding, physics_params], dim=-1)
        branch_out = self.branch_proj(branch_input)
        
        for layer in self.branch_layers:
            branch_out = branch_out + layer(branch_out)
        
        # Handle coordinate format
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(batch_size, -1, -1)
        
        num_points = coords.shape[1]
        
        # Trunk encoding (separate time and space)
        time_enc = self.time_encoder(coords[..., :1])  # [B, N, hidden]
        space_enc = self.space_encoder(coords[..., 1:])  # [B, N, hidden]
        
        trunk_out = time_enc + space_enc  # [B, N, hidden]
        
        # Cross-attention between branch and trunk
        branch_out = branch_out.unsqueeze(1)  # [B, 1, hidden]
        attn_out, _ = self.cross_attention(
            trunk_out, branch_out, branch_out
        )
        
        # Combine and predict
        combined = torch.cat([trunk_out, attn_out], dim=-1)
        output = self.output_head(combined).squeeze(-1)  # [B, N]
        
        return output
    
    def predict_field(
        self,
        geometry_embedding: torch.Tensor,
        physics_params: torch.Tensor,
        time_steps: int = 50,
        resolution: int = 128,
    ) -> torch.Tensor:
        """Predict full temperature field."""
        batch_size = geometry_embedding.shape[0]
        device = geometry_embedding.device
        
        t_vals = torch.linspace(0, 1, time_steps, device=device)
        x_vals = torch.linspace(0, 1, resolution, device=device)
        y_vals = torch.linspace(0, 1, resolution, device=device)
        
        T_grid, X_grid, Y_grid = torch.meshgrid(t_vals, x_vals, y_vals, indexing='ij')
        coords = torch.stack([
            T_grid.flatten(),
            X_grid.flatten(),
            Y_grid.flatten(),
        ], dim=-1)
        
        output = self(geometry_embedding, physics_params, coords)
        output = output.view(batch_size, time_steps, resolution, resolution)
        
        return output
