"""
Test configuration for pytest.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_geometry():
    """Create a sample geometry mask."""
    import numpy as np
    
    resolution = 64
    mask = np.zeros((resolution, resolution), dtype=np.float32)
    
    # Create a simple circle
    center = resolution // 2
    radius = resolution // 4
    
    y, x = np.ogrid[:resolution, :resolution]
    dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    mask[dist <= radius] = 1.0
    
    return mask


@pytest.fixture
def sample_time_steps():
    """Create sample time steps."""
    import numpy as np
    return np.linspace(0, 1.0, 10)


@pytest.fixture
def sample_physics_params():
    """Create sample physics parameters."""
    return {
        "alpha": 0.1,
        "heat_source": {
            "x": 0.5,
            "y": 0.5,
            "intensity": 1.0,
            "radius": 0.1,
        },
    }


@pytest.fixture
def device():
    """Get available device."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
