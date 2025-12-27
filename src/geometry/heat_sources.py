"""
Heat source definition and placement for thermal simulations.

Supports various heat source types:
- Point sources
- Area sources
- Gaussian distributions
- Time-varying sources
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Union
from enum import Enum


class HeatSourceType(Enum):
    """Types of heat sources."""
    POINT = "point"
    RECTANGULAR = "rectangular"
    CIRCULAR = "circular"
    GAUSSIAN = "gaussian"
    CUSTOM = "custom"


@dataclass
class HeatSource:
    """
    Definition of a single heat source.
    
    Attributes:
        source_type: Type of heat source
        position: (x, y) position in normalized coordinates [0, 1]
        power: Heat generation rate in Watts
        size: Size parameter (radius for circular, (w, h) for rectangular)
        time_profile: Optional time-varying profile function
        name: Optional name for the heat source
    """
    source_type: HeatSourceType
    position: Tuple[float, float]
    power: float  # Watts
    size: Union[float, Tuple[float, float]] = 0.05
    time_profile: Optional[Callable[[float], float]] = None
    name: str = "heat_source"
    
    def get_power_at_time(self, t: float) -> float:
        """Get power at a specific time."""
        if self.time_profile is None:
            return self.power
        return self.power * self.time_profile(t)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "type": self.source_type.value,
            "position": self.position,
            "power": self.power,
            "size": self.size,
            "name": self.name,
        }


@dataclass
class HeatSourceConfiguration:
    """
    Collection of heat sources for a simulation.
    
    Provides methods to:
    - Add/remove heat sources
    - Generate heat source field Q(x, y, t)
    - Calculate total power
    """
    sources: List[HeatSource] = field(default_factory=list)
    domain_size: Tuple[float, float] = (0.1, 0.1)  # meters
    
    def add_point_source(
        self,
        x: float,
        y: float,
        power: float,
        name: str = None,
    ) -> "HeatSourceConfiguration":
        """Add a point heat source."""
        source = HeatSource(
            source_type=HeatSourceType.POINT,
            position=(x, y),
            power=power,
            size=0.01,  # Small radius for approximation
            name=name or f"point_{len(self.sources)}",
        )
        self.sources.append(source)
        return self
    
    def add_rectangular_source(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        power: float,
        name: str = None,
    ) -> "HeatSourceConfiguration":
        """Add a rectangular heat source area."""
        source = HeatSource(
            source_type=HeatSourceType.RECTANGULAR,
            position=(x, y),
            power=power,
            size=(width, height),
            name=name or f"rect_{len(self.sources)}",
        )
        self.sources.append(source)
        return self
    
    def add_circular_source(
        self,
        x: float,
        y: float,
        radius: float,
        power: float,
        name: str = None,
    ) -> "HeatSourceConfiguration":
        """Add a circular heat source area."""
        source = HeatSource(
            source_type=HeatSourceType.CIRCULAR,
            position=(x, y),
            power=power,
            size=radius,
            name=name or f"circ_{len(self.sources)}",
        )
        self.sources.append(source)
        return self
    
    def add_gaussian_source(
        self,
        x: float,
        y: float,
        sigma: float,
        power: float,
        name: str = None,
    ) -> "HeatSourceConfiguration":
        """Add a Gaussian-distributed heat source."""
        source = HeatSource(
            source_type=HeatSourceType.GAUSSIAN,
            position=(x, y),
            power=power,
            size=sigma,
            name=name or f"gauss_{len(self.sources)}",
        )
        self.sources.append(source)
        return self
    
    def add_pulsed_source(
        self,
        x: float,
        y: float,
        power: float,
        period: float,
        duty_cycle: float = 0.5,
        name: str = None,
    ) -> "HeatSourceConfiguration":
        """Add a pulsed (time-varying) heat source."""
        def pulse_profile(t: float) -> float:
            phase = (t % period) / period
            return 1.0 if phase < duty_cycle else 0.0
        
        source = HeatSource(
            source_type=HeatSourceType.POINT,
            position=(x, y),
            power=power,
            size=0.01,
            time_profile=pulse_profile,
            name=name or f"pulse_{len(self.sources)}",
        )
        self.sources.append(source)
        return self
    
    @property
    def total_power(self) -> float:
        """Total power from all sources."""
        return sum(s.power for s in self.sources)
    
    @property
    def num_sources(self) -> int:
        """Number of heat sources."""
        return len(self.sources)
    
    def generate_field(
        self,
        resolution: int,
        time: float = 0.0,
    ) -> np.ndarray:
        """
        Generate the heat source field Q(x, y) at a given time.
        
        Args:
            resolution: Grid resolution
            time: Time point for time-varying sources
            
        Returns:
            Heat source field [resolution, resolution] in W/m³
        """
        Q = np.zeros((resolution, resolution), dtype=np.float32)
        
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        X, Y = np.meshgrid(x, y)
        
        dx = self.domain_size[0] / resolution
        dy = self.domain_size[1] / resolution
        thickness = 0.001  # Assume 1mm thickness for 2D
        cell_volume = dx * dy * thickness
        
        for source in self.sources:
            power = source.get_power_at_time(time)
            sx, sy = source.position
            
            if source.source_type == HeatSourceType.POINT:
                # Approximate point source as small Gaussian
                sigma = max(source.size, 1e-6)  # Avoid zero sigma
                dist2 = (X - sx)**2 + (Y - sy)**2
                field = np.exp(-dist2 / (2 * sigma**2))
                field_sum = field.sum()
                if field_sum > 1e-10:
                    field = field / (field_sum * cell_volume)
                Q += power * field
            
            elif source.source_type == HeatSourceType.RECTANGULAR:
                w, h = source.size if isinstance(source.size, tuple) else (source.size, source.size)
                mask = (
                    (X >= sx - w/2) & (X <= sx + w/2) &
                    (Y >= sy - h/2) & (Y <= sy + h/2)
                )
                # Calculate actual area covered in physical units
                physical_w = w * self.domain_size[0]
                physical_h = h * self.domain_size[1]
                area = physical_w * physical_h
                volume = area * thickness
                if volume > 1e-10 and mask.sum() > 0:
                    Q[mask] += power / volume
            
            elif source.source_type == HeatSourceType.CIRCULAR:
                r = source.size if isinstance(source.size, (int, float)) else source.size[0]
                dist = np.sqrt((X - sx)**2 + (Y - sy)**2)
                mask = dist <= r
                # Physical radius
                physical_r = r * min(self.domain_size[0], self.domain_size[1])
                area = np.pi * physical_r**2
                volume = area * thickness
                if volume > 1e-10 and mask.sum() > 0:
                    Q[mask] += power / volume
            
            elif source.source_type == HeatSourceType.GAUSSIAN:
                sigma = max(source.size if isinstance(source.size, (int, float)) else source.size[0], 1e-6)
                dist2 = (X - sx)**2 + (Y - sy)**2
                field = np.exp(-dist2 / (2 * sigma**2))
                field_sum = field.sum()
                if field_sum > 1e-10:
                    field = field / (field_sum * cell_volume)
                Q += power * field
        
        return Q
    
    def get_source_function(self) -> Callable:
        """
        Get a callable heat source function Q(t, x, y).
        
        Returns:
            Function that takes (t, x, y) tensors and returns heat generation rate
        """
        def heat_source_fn(t, x, y):
            """Heat source function for PINN."""
            import torch
            
            Q = torch.zeros_like(x)
            
            for source in self.sources:
                power = source.power
                sx, sy = source.position
                
                if source.source_type == HeatSourceType.GAUSSIAN:
                    sigma = source.size
                    dist2 = (x - sx)**2 + (y - sy)**2
                    # Normalized Gaussian
                    field = torch.exp(-dist2 / (2 * sigma**2))
                    field = field / (2 * np.pi * sigma**2)
                    
                    # Apply time profile if exists
                    if source.time_profile is not None:
                        # For tensor t, we need element-wise evaluation
                        if isinstance(t, torch.Tensor):
                            time_factor = torch.tensor([
                                source.time_profile(ti.item()) 
                                for ti in t.flatten()
                            ], device=t.device, dtype=t.dtype).reshape(t.shape)
                        else:
                            time_factor = source.time_profile(t)
                        Q = Q + power * field * time_factor
                    else:
                        Q = Q + power * field
                
                elif source.source_type == HeatSourceType.POINT:
                    sigma = source.size
                    dist2 = (x - sx)**2 + (y - sy)**2
                    field = torch.exp(-dist2 / (2 * sigma**2))
                    field = field / (2 * np.pi * sigma**2)
                    Q = Q + power * field
            
            return Q
        
        return heat_source_fn
    
    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Heat Source Configuration:",
            f"  Total Sources: {self.num_sources}",
            f"  Total Power: {self.total_power:.2f} W",
            f"  Domain Size: {self.domain_size[0]*100:.1f} x {self.domain_size[1]*100:.1f} cm",
            f"",
            f"  Sources:",
        ]
        
        for i, s in enumerate(self.sources):
            lines.append(
                f"    {i+1}. {s.name}: {s.source_type.value}, "
                f"pos=({s.position[0]:.2f}, {s.position[1]:.2f}), "
                f"power={s.power:.1f}W"
            )
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "domain_size": self.domain_size,
            "total_power": self.total_power,
            "sources": [s.to_dict() for s in self.sources],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "HeatSourceConfiguration":
        """Create from dictionary."""
        config = cls(domain_size=tuple(data.get("domain_size", (0.1, 0.1))))
        
        for s_data in data.get("sources", []):
            source = HeatSource(
                source_type=HeatSourceType(s_data["type"]),
                position=tuple(s_data["position"]),
                power=s_data["power"],
                size=s_data.get("size", 0.05),
                name=s_data.get("name", "source"),
            )
            config.sources.append(source)
        
        return config


# Convenience functions
def create_single_chip_source(
    chip_x: float = 0.5,
    chip_y: float = 0.5,
    chip_power: float = 10.0,  # Watts
    chip_size: float = 0.1,    # 10% of domain
    domain_size: Tuple[float, float] = (0.1, 0.1),
) -> HeatSourceConfiguration:
    """Create a simple single-chip heat source configuration."""
    config = HeatSourceConfiguration(domain_size=domain_size)
    config.add_gaussian_source(chip_x, chip_y, chip_size/3, chip_power, "chip")
    return config


def create_multi_chip_source(
    chip_positions: List[Tuple[float, float]],
    chip_powers: List[float],
    chip_size: float = 0.05,
    domain_size: Tuple[float, float] = (0.1, 0.1),
) -> HeatSourceConfiguration:
    """Create a multi-chip heat source configuration."""
    config = HeatSourceConfiguration(domain_size=domain_size)
    
    for i, (pos, power) in enumerate(zip(chip_positions, chip_powers)):
        config.add_gaussian_source(
            pos[0], pos[1], chip_size/3, power, f"chip_{i+1}"
        )
    
    return config
