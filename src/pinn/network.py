"""
Physics-Informed Neural Network for solving the 2D transient heat equation.

The heat equation: ∂T/∂t = α * (∂²T/∂x² + ∂²T/∂y²) + Q(x, y, t)

Where:
    T: Temperature field
    α: Thermal diffusivity
    Q: Heat source term

Features:
    - Fourier feature embeddings for better high-frequency learning
    - Adaptive activation functions
    - Material-aware conditioning
    - Residual connections for deeper networks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple


class FourierFeatures(nn.Module):
    """
    Fourier feature mapping for improved coordinate encoding.
    
    Maps low-dimensional coordinates to higher-dimensional space using
    sinusoidal functions, enabling better learning of high-frequency patterns.
    
    Reference: Tancik et al., "Fourier Features Let Networks Learn 
    High Frequency Functions in Low Dimensional Domains" (2020)
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        num_frequencies: int = 32,
        scale: float = 1.0,
        learnable: bool = False,
    ):
        """
        Initialize Fourier features.
        
        Args:
            input_dim: Number of input dimensions (t, x, y)
            num_frequencies: Number of frequency components
            scale: Scaling factor for frequencies
            learnable: If True, frequencies are learnable parameters
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.output_dim = input_dim + 2 * input_dim * num_frequencies
        
        # Initialize frequency matrix
        if learnable:
            self.B = nn.Parameter(
                scale * torch.randn(num_frequencies, input_dim)
            )
        else:
            self.register_buffer(
                'B',
                scale * torch.randn(num_frequencies, input_dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier feature mapping.
        
        Args:
            x: Input coordinates [N, input_dim]
            
        Returns:
            Fourier features [N, output_dim]
        """
        # x @ B.T gives [N, num_frequencies]
        projected = 2 * np.pi * torch.matmul(x, self.B.T)
        
        # Concatenate original coordinates with sin and cos features
        return torch.cat([x, torch.sin(projected), torch.cos(projected)], dim=-1)


class AdaptiveActivation(nn.Module):
    """
    Adaptive activation function with learnable parameters.
    
    Combines multiple activation functions with learnable weights,
    allowing the network to learn optimal activation patterns.
    """
    
    def __init__(self, initial_a: float = 1.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(initial_a))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.a * x)


class ResidualBlock(nn.Module):
    """Residual block for deeper PINN networks."""
    
    def __init__(
        self,
        dim: int,
        activation: str = "tanh",
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "adaptive":
            self.activation = AdaptiveActivation()
        else:
            self.activation = nn.Tanh()
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return self.activation(out + residual)


class PINNNetwork(nn.Module):
    """
    Enhanced Physics-Informed Neural Network for heat equation.
    
    Features:
    - Optional Fourier feature encoding
    - Adaptive activation functions
    - Residual connections
    - Material property conditioning
    
    Input: (t, x, y) coordinates
    Output: Temperature T at those coordinates
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = [128, 128, 128, 128],
        activation: str = "tanh",
        input_dim: int = 3,  # (t, x, y)
        output_dim: int = 1,  # T
        use_fourier_features: bool = True,
        num_fourier_frequencies: int = 32,
        fourier_scale: float = 1.0,
        use_residual: bool = False,
        dropout: float = 0.0,
    ):
        """
        Initialize the PINN network.
        
        Args:
            hidden_layers: List of hidden layer sizes
            activation: Activation function ('tanh', 'silu', 'gelu', 'adaptive')
            input_dim: Number of input dimensions
            output_dim: Number of output dimensions
            use_fourier_features: Whether to use Fourier feature encoding
            num_fourier_frequencies: Number of Fourier frequencies
            fourier_scale: Scale for Fourier frequencies
            use_residual: Whether to use residual connections
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_fourier_features = use_fourier_features
        self.use_residual = use_residual
        
        # Fourier features
        if use_fourier_features:
            self.fourier = FourierFeatures(
                input_dim=input_dim,
                num_frequencies=num_fourier_frequencies,
                scale=fourier_scale,
            )
            actual_input_dim = self.fourier.output_dim
        else:
            self.fourier = None
            actual_input_dim = input_dim
        
        # Build network layers
        if use_residual and len(hidden_layers) >= 2:
            # First layer to project to hidden dimension
            self.input_layer = nn.Linear(actual_input_dim, hidden_layers[0])
            
            # Residual blocks (all same dimension)
            self.residual_blocks = nn.ModuleList([
                ResidualBlock(hidden_layers[0], activation, dropout)
                for _ in range(len(hidden_layers) - 1)
            ])
            
            # Output layer
            self.output_layer = nn.Linear(hidden_layers[0], output_dim)
            
            # Activation for input layer
            if activation == "tanh":
                self.input_activation = nn.Tanh()
            elif activation == "silu":
                self.input_activation = nn.SiLU()
            elif activation == "gelu":
                self.input_activation = nn.GELU()
            elif activation == "adaptive":
                self.input_activation = AdaptiveActivation()
            else:
                self.input_activation = nn.Tanh()
            
            self.network = None
        else:
            # Standard MLP
            layers = []
            prev_dim = actual_input_dim
            
            for hidden_dim in hidden_layers:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                
                if activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "silu":
                    layers.append(nn.SiLU())
                elif activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "adaptive":
                    layers.append(AdaptiveActivation())
                else:
                    layers.append(nn.Tanh())
                
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, output_dim))
            
            self.network = nn.Sequential(*layers)
            self.input_layer = None
            self.residual_blocks = None
            self.output_layer = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            t: Time coordinates [N,]
            x: X coordinates [N,]
            y: Y coordinates [N,]
            
        Returns:
            Temperature values [N, 1]
        """
        # Stack inputs
        inputs = torch.stack([t, x, y], dim=-1)
        
        # Apply Fourier features if enabled
        if self.use_fourier_features and self.fourier is not None:
            inputs = self.fourier(inputs)
        
        # Forward through network
        if self.network is not None:
            return self.network(inputs)
        else:
            # Residual network path
            h = self.input_activation(self.input_layer(inputs))
            for block in self.residual_blocks:
                h = block(h)
            return self.output_layer(h)


class MaterialAwarePINN(nn.Module):
    """
    PINN that incorporates material properties as additional inputs.
    
    This allows training a single model that works across different materials
    by conditioning on thermal diffusivity and other properties.
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = [256, 256, 256, 256],
        activation: str = "tanh",
        num_material_features: int = 4,  # k, rho, cp, alpha
        use_fourier_features: bool = True,
        num_fourier_frequencies: int = 32,
    ):
        """
        Initialize material-aware PINN.
        
        Args:
            hidden_layers: List of hidden layer sizes
            activation: Activation function
            num_material_features: Number of material property inputs
            use_fourier_features: Whether to use Fourier features for coordinates
            num_fourier_frequencies: Number of Fourier frequencies
        """
        super().__init__()
        
        self.num_material_features = num_material_features
        
        # Fourier features for coordinates only
        if use_fourier_features:
            self.fourier = FourierFeatures(
                input_dim=3,  # t, x, y
                num_frequencies=num_fourier_frequencies,
            )
            coord_dim = self.fourier.output_dim
        else:
            self.fourier = None
            coord_dim = 3
        
        # Material feature encoder
        self.material_encoder = nn.Sequential(
            nn.Linear(num_material_features, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        
        # Main network
        input_dim = coord_dim + 64  # Fourier coords + material encoding
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "silu":
                layers.append(nn.SiLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        material_props: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with material conditioning.
        
        Args:
            t: Time coordinates [N,]
            x: X coordinates [N,]
            y: Y coordinates [N,]
            material_props: Material properties [N, num_material_features] or [num_material_features,]
                           Expected order: [k, rho, cp, alpha] (normalized)
            
        Returns:
            Temperature values [N, 1]
        """
        # Stack coordinates
        coords = torch.stack([t, x, y], dim=-1)
        
        # Apply Fourier features
        if self.fourier is not None:
            coords = self.fourier(coords)
        
        # Ensure material_props is properly shaped
        if material_props.dim() == 1:
            material_props = material_props.unsqueeze(0).expand(t.shape[0], -1)
        
        # Encode material properties
        material_encoding = self.material_encoder(material_props)
        
        # Concatenate and forward
        inputs = torch.cat([coords, material_encoding], dim=-1)
        return self.network(inputs)


class PINNNetworkWithGeometry(nn.Module):
    """
    PINN network that also takes geometry encoding as input.
    
    This allows training a single PINN that can handle multiple geometries
    by conditioning on the geometry embedding.
    """
    
    def __init__(
        self,
        geometry_dim: int = 512,
        hidden_layers: List[int] = [256, 256, 256, 256],
        activation: str = "tanh",
        use_fourier_features: bool = True,
        num_fourier_frequencies: int = 32,
    ):
        """
        Initialize the geometry-conditioned PINN.
        
        Args:
            geometry_dim: Dimension of geometry embedding
            hidden_layers: List of hidden layer sizes
            activation: Activation function
            use_fourier_features: Whether to use Fourier features
            num_fourier_frequencies: Number of Fourier frequencies
        """
        super().__init__()
        
        self.geometry_dim = geometry_dim
        
        # Fourier features for coordinates
        if use_fourier_features:
            self.fourier = FourierFeatures(
                input_dim=3,
                num_frequencies=num_fourier_frequencies,
            )
            coord_dim = self.fourier.output_dim
        else:
            self.fourier = None
            coord_dim = 3
        
        # Input: Fourier coords + geometry_embedding
        input_dim = coord_dim + geometry_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "silu":
                layers.append(nn.SiLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        geometry_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with geometry conditioning.
        
        Args:
            t: Time coordinates [N,]
            x: X coordinates [N,]
            y: Y coordinates [N,]
            geometry_embedding: Geometry embedding [N, geometry_dim] or [geometry_dim,]
            
        Returns:
            Temperature values [N, 1]
        """
        # Stack coordinates
        coords = torch.stack([t, x, y], dim=-1)
        
        # Apply Fourier features
        if self.fourier is not None:
            coords = self.fourier(coords)
        
        # Ensure geometry embedding is properly shaped
        if geometry_embedding.dim() == 1:
            geometry_embedding = geometry_embedding.unsqueeze(0).expand(t.shape[0], -1)
        
        inputs = torch.cat([coords, geometry_embedding], dim=-1)
        
        return self.network(inputs)

