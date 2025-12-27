"""
Shape generation functions for 2D geometry masks.
All functions return binary numpy arrays (0 = outside, 1 = inside domain).
"""

import numpy as np
from typing import Tuple, Optional, List
import cv2


def create_circle(
    resolution: int,
    center: Optional[Tuple[float, float]] = None,
    radius: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Create a circular geometry mask.
    
    Args:
        resolution: Size of the output image (resolution x resolution)
        center: Center of circle as (x, y) in [0, 1] range. Random if None.
        radius: Radius in [0, 0.5] range. Random if None.
        rng: Random number generator for reproducibility
        
    Returns:
        Binary mask array of shape (resolution, resolution)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if center is None:
        # Ensure circle fits within domain
        margin = 0.15
        center = (
            rng.uniform(margin + 0.1, 1 - margin - 0.1),
            rng.uniform(margin + 0.1, 1 - margin - 0.1),
        )
    
    if radius is None:
        # Max radius that fits
        max_r = min(center[0], 1 - center[0], center[1], 1 - center[1]) * 0.9
        radius = rng.uniform(0.1, max(0.11, max_r))
    
    # Create coordinate grid
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Distance from center
    dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    mask = (dist <= radius).astype(np.float32)
    return mask


def create_rectangle(
    resolution: int,
    corner: Optional[Tuple[float, float]] = None,
    size: Optional[Tuple[float, float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Create a rectangular geometry mask.
    
    Args:
        resolution: Size of the output image
        corner: Bottom-left corner as (x, y) in [0, 1] range
        size: Width and height as (w, h) in [0, 1] range
        rng: Random number generator
        
    Returns:
        Binary mask array
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if corner is None:
        corner = (rng.uniform(0.05, 0.3), rng.uniform(0.05, 0.3))
    
    if size is None:
        max_w = 1 - corner[0] - 0.05
        max_h = 1 - corner[1] - 0.05
        size = (
            rng.uniform(0.3, max(0.31, max_w)),
            rng.uniform(0.3, max(0.31, max_h)),
        )
    
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    mask = (
        (X >= corner[0]) & (X <= corner[0] + size[0]) &
        (Y >= corner[1]) & (Y <= corner[1] + size[1])
    ).astype(np.float32)
    
    return mask


def create_ellipse(
    resolution: int,
    center: Optional[Tuple[float, float]] = None,
    axes: Optional[Tuple[float, float]] = None,
    angle: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Create an elliptical geometry mask.
    
    Args:
        resolution: Size of the output image
        center: Center of ellipse as (x, y) in [0, 1] range
        axes: Semi-axes lengths as (a, b) in [0, 0.5] range
        angle: Rotation angle in radians
        rng: Random number generator
        
    Returns:
        Binary mask array
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if center is None:
        center = (rng.uniform(0.25, 0.75), rng.uniform(0.25, 0.75))
    
    if axes is None:
        max_a = min(center[0], 1 - center[0], center[1], 1 - center[1]) * 0.8
        axes = (
            rng.uniform(0.1, max(0.11, max_a)),
            rng.uniform(0.1, max(0.11, max_a)),
        )
    
    if angle is None:
        angle = rng.uniform(0, np.pi)
    
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Rotate coordinates
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    X_rot = cos_a * (X - center[0]) + sin_a * (Y - center[1])
    Y_rot = -sin_a * (X - center[0]) + cos_a * (Y - center[1])
    
    # Ellipse equation
    dist = (X_rot / axes[0])**2 + (Y_rot / axes[1])**2
    mask = (dist <= 1).astype(np.float32)
    
    return mask


def create_polygon(
    resolution: int,
    num_vertices: Optional[int] = None,
    vertices: Optional[List[Tuple[float, float]]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Create a convex polygon geometry mask.
    
    Args:
        resolution: Size of the output image
        num_vertices: Number of vertices (3-8)
        vertices: List of vertex coordinates. Random if None.
        rng: Random number generator
        
    Returns:
        Binary mask array
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if vertices is None:
        if num_vertices is None:
            num_vertices = rng.integers(3, 9)
        
        # Generate random convex polygon using angles
        center = (rng.uniform(0.35, 0.65), rng.uniform(0.35, 0.65))
        angles = np.sort(rng.uniform(0, 2 * np.pi, num_vertices))
        radii = rng.uniform(0.15, 0.3, num_vertices)
        
        vertices = [
            (center[0] + r * np.cos(a), center[1] + r * np.sin(a))
            for a, r in zip(angles, radii)
        ]
    
    # Convert to pixel coordinates
    pts = np.array([(int(v[0] * resolution), int(v[1] * resolution)) 
                    for v in vertices], dtype=np.int32)
    
    mask = np.zeros((resolution, resolution), dtype=np.float32)
    cv2.fillPoly(mask, [pts], 1.0)
    
    return mask


def create_l_shape(
    resolution: int,
    corner: Optional[Tuple[float, float]] = None,
    outer_size: Optional[Tuple[float, float]] = None,
    inner_size: Optional[Tuple[float, float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Create an L-shaped geometry mask.
    
    Args:
        resolution: Size of the output image
        corner: Bottom-left corner position
        outer_size: Outer rectangle size
        inner_size: Size of the cut-out rectangle
        rng: Random number generator
        
    Returns:
        Binary mask array
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if corner is None:
        corner = (rng.uniform(0.05, 0.2), rng.uniform(0.05, 0.2))
    
    if outer_size is None:
        max_w = 1 - corner[0] - 0.05
        max_h = 1 - corner[1] - 0.05
        outer_size = (
            rng.uniform(0.5, max(0.51, max_w)),
            rng.uniform(0.5, max(0.51, max_h)),
        )
    
    if inner_size is None:
        inner_size = (
            rng.uniform(0.3, 0.5) * outer_size[0],
            rng.uniform(0.3, 0.5) * outer_size[1],
        )
    
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Outer rectangle
    outer = (
        (X >= corner[0]) & (X <= corner[0] + outer_size[0]) &
        (Y >= corner[1]) & (Y <= corner[1] + outer_size[1])
    )
    
    # Inner cut-out (top-right corner)
    cut_x = corner[0] + outer_size[0] - inner_size[0]
    cut_y = corner[1] + outer_size[1] - inner_size[1]
    inner = (
        (X >= cut_x) & (X <= corner[0] + outer_size[0]) &
        (Y >= cut_y) & (Y <= corner[1] + outer_size[1])
    )
    
    mask = (outer & ~inner).astype(np.float32)
    return mask


def create_t_shape(
    resolution: int,
    center: Optional[Tuple[float, float]] = None,
    stem_size: Optional[Tuple[float, float]] = None,
    top_size: Optional[Tuple[float, float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Create a T-shaped geometry mask.
    
    Args:
        resolution: Size of the output image
        center: Center position of the T
        stem_size: Width and height of stem (vertical part)
        top_size: Width and height of top (horizontal part)
        rng: Random number generator
        
    Returns:
        Binary mask array
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if center is None:
        center = (rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7))
    
    if stem_size is None:
        stem_size = (rng.uniform(0.1, 0.2), rng.uniform(0.3, 0.5))
    
    if top_size is None:
        top_size = (rng.uniform(0.4, 0.6), rng.uniform(0.1, 0.15))
    
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Stem (vertical part)
    stem = (
        (X >= center[0] - stem_size[0]/2) & (X <= center[0] + stem_size[0]/2) &
        (Y >= center[1] - stem_size[1]/2) & (Y <= center[1] + stem_size[1]/2)
    )
    
    # Top (horizontal part at top of stem)
    top_y = center[1] + stem_size[1]/2 - top_size[1]
    top = (
        (X >= center[0] - top_size[0]/2) & (X <= center[0] + top_size[0]/2) &
        (Y >= top_y) & (Y <= center[1] + stem_size[1]/2)
    )
    
    mask = (stem | top).astype(np.float32)
    return mask


def create_complex_shape(
    resolution: int,
    num_primitives: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Create a complex shape by combining multiple primitives with boolean operations.
    
    Args:
        resolution: Size of the output image
        num_primitives: Number of primitives to combine
        rng: Random number generator
        
    Returns:
        Binary mask array
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if num_primitives is None:
        num_primitives = rng.integers(2, 5)
    
    # Start with a base shape
    shape_funcs = [create_circle, create_rectangle, create_ellipse]
    
    mask = shape_funcs[rng.integers(0, len(shape_funcs))](resolution, rng=rng)
    
    for _ in range(num_primitives - 1):
        new_shape = shape_funcs[rng.integers(0, len(shape_funcs))](resolution, rng=rng)
        
        # Random boolean operation
        op = rng.integers(0, 3)
        if op == 0:  # Union
            mask = np.maximum(mask, new_shape)
        elif op == 1:  # Intersection
            mask = np.minimum(mask, new_shape)
        else:  # Difference
            mask = np.maximum(mask - new_shape, 0)
    
    # Ensure non-trivial result
    if mask.sum() < 0.05 * resolution * resolution:
        # Fallback to simple circle if too small
        mask = create_circle(resolution, rng=rng)
    
    return mask.astype(np.float32)
