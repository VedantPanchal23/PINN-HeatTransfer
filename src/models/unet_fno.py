"""
UNet-FNO Hybrid Architecture for thermal simulation.

Combines the spatial hierarchy of U-Net with the spectral properties of FNO
for improved multi-scale feature learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from .fno import SpectralConv2d


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
    ):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling block with spectral convolution."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 16,
    ):
        super().__init__()
        
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.spectral = SpectralConv2d(out_channels, out_channels, modes, modes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(x)
        x = self.conv(x)
        x = x + self.spectral(x)  # Residual spectral connection
        return x


class UpBlock(nn.Module):
    """Upsampling block with spectral convolution."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int = 16,
    ):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.spectral = SpectralConv2d(out_channels, out_channels, modes, modes)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # Handle size mismatch
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2])
        
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        x = x + self.spectral(x)
        return x


class UNetFNO(nn.Module):
    """
    U-Net with Fourier Neural Operator layers.
    
    This architecture combines:
    - U-Net's encoder-decoder structure for multi-scale processing
    - FNO's spectral convolutions for global receptive fields
    - Skip connections for preserving high-frequency details
    """
    
    def __init__(
        self,
        geometry_dim: int = 512,
        physics_dim: int = 4,
        base_channels: int = 64,
        time_steps: int = 50,
        modes: int = 16,
        resolution: int = 128,
    ):
        """
        Initialize UNet-FNO.
        
        Args:
            geometry_dim: Geometry embedding dimension
            physics_dim: Physics parameters dimension
            base_channels: Base channel count (doubled at each level)
            time_steps: Number of output time steps
            modes: Fourier modes for spectral convolutions
            resolution: Input spatial resolution
        """
        super().__init__()
        
        self.geometry_dim = geometry_dim
        self.physics_dim = physics_dim
        self.time_steps = time_steps
        self.resolution = resolution
        
        # Conditioning network (projects geometry + physics to spatial map)
        self.condition_proj = nn.Sequential(
            nn.Linear(geometry_dim + physics_dim, base_channels * 4),
            nn.GELU(),
            nn.Linear(base_channels * 4, base_channels * 4),
        )
        
        # Input projection (coordinates + condition)
        input_channels = 2 + base_channels  # (x, y) + condition channels
        
        # Encoder
        self.inc = DoubleConv(input_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels * 2, modes)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, modes // 2)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, modes // 4)
        
        # Bottleneck with FNO
        self.bottleneck = nn.Sequential(
            DoubleConv(base_channels * 8, base_channels * 8),
            SpectralConv2d(base_channels * 8, base_channels * 8, modes // 4, modes // 4),
        )
        
        # Decoder
        self.up1 = UpBlock(base_channels * 16, base_channels * 4, modes // 4)
        self.up2 = UpBlock(base_channels * 8, base_channels * 2, modes // 2)
        self.up3 = UpBlock(base_channels * 4, base_channels, modes)
        
        # Output projection
        self.outc = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 1),
            nn.GELU(),
            nn.Conv2d(base_channels, time_steps, 1),
        )
    
    def forward(
        self,
        geometry_embedding: torch.Tensor,
        physics_params: torch.Tensor,
        resolution: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            geometry_embedding: [B, geometry_dim]
            physics_params: [B, physics_dim]
            resolution: Override spatial resolution
            
        Returns:
            Temperature field [B, T, H, W]
        """
        if resolution is None:
            resolution = self.resolution
        
        batch_size = geometry_embedding.shape[0]
        device = geometry_embedding.device
        
        # Create coordinate grid
        grid = self._get_grid(batch_size, resolution, device)  # [B, H, W, 2]
        
        # Project conditioning
        condition = torch.cat([geometry_embedding, physics_params], dim=-1)
        condition = self.condition_proj(condition)  # [B, base_channels * 4]
        
        # Reshape and broadcast condition to spatial dimensions
        cond_spatial = condition.view(batch_size, -1, 1, 1)
        cond_spatial = cond_spatial.expand(-1, -1, resolution, resolution)
        
        # Combine grid and condition
        grid_channels = grid.permute(0, 3, 1, 2)  # [B, 2, H, W]
        x = torch.cat([grid_channels, cond_spatial], dim=1)  # [B, 2 + cond_dim, H, W]
        
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bottleneck
        x = self.bottleneck(x4)
        
        # Decoder with skip connections
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # Output
        out = self.outc(x)  # [B, T, H, W]
        
        return out
    
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


class LightweightUNetFNO(nn.Module):
    """
    Lightweight version of UNet-FNO for faster inference.
    
    Reduces parameters while maintaining performance through:
    - Depthwise separable convolutions
    - Fewer channels
    - Single spectral layer at bottleneck
    """
    
    def __init__(
        self,
        geometry_dim: int = 512,
        physics_dim: int = 4,
        base_channels: int = 32,
        time_steps: int = 50,
        modes: int = 12,
    ):
        super().__init__()
        
        self.time_steps = time_steps
        
        # Efficient conditioning
        self.condition_proj = nn.Linear(geometry_dim + physics_dim, base_channels)
        
        # Depthwise separable encoder
        self.enc1 = self._make_dw_sep_block(2 + base_channels, base_channels)
        self.enc2 = self._make_dw_sep_block(base_channels, base_channels * 2)
        self.enc3 = self._make_dw_sep_block(base_channels * 2, base_channels * 4)
        
        self.pool = nn.MaxPool2d(2)
        
        # Spectral bottleneck
        self.spectral = SpectralConv2d(
            base_channels * 4, base_channels * 4, modes, modes
        )
        
        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = self._make_dw_sep_block(base_channels * 8, base_channels * 2)
        self.dec2 = self._make_dw_sep_block(base_channels * 4, base_channels)
        self.dec1 = self._make_dw_sep_block(base_channels * 2, base_channels)
        
        # Output
        self.out = nn.Conv2d(base_channels, time_steps, 1)
    
    def _make_dw_sep_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Create depthwise separable convolution block."""
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.GELU(),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
    
    def forward(
        self,
        geometry_embedding: torch.Tensor,
        physics_params: torch.Tensor,
        resolution: int = 128,
    ) -> torch.Tensor:
        batch_size = geometry_embedding.shape[0]
        device = geometry_embedding.device
        
        # Grid
        x = torch.linspace(0, 1, resolution, device=device)
        y = torch.linspace(0, 1, resolution, device=device)
        gx, gy = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Condition
        cond = torch.cat([geometry_embedding, physics_params], dim=-1)
        cond = self.condition_proj(cond)
        cond = cond.view(batch_size, -1, 1, 1).expand(-1, -1, resolution, resolution)
        
        # Input
        x = torch.cat([grid, cond], dim=1)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.spectral(e3)
        
        # Decoder
        d3 = self.dec3(torch.cat([self.up(b), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e1], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), grid], dim=1))
        
        return self.out(d1)
