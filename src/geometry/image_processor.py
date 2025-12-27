"""
Image-based geometry processing for PINN thermal simulations.

Converts PNG images, CAD files, and parametric definitions into
computational domains with signed distance functions (SDF).
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass
from scipy import ndimage
from PIL import Image
import cv2


@dataclass
class DomainInfo:
    """Information about the computational domain."""
    mask: np.ndarray                    # Binary mask (1=inside, 0=outside)
    sdf: np.ndarray                     # Signed distance function
    boundary_points: np.ndarray         # Boundary coordinates [N, 2]
    area: float                         # Domain area (m²)
    perimeter: float                    # Domain perimeter (m)
    centroid: Tuple[float, float]       # Domain centroid
    bounding_box: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    resolution: Tuple[int, int]         # (height, width)
    physical_size: Tuple[float, float]  # (height_m, width_m) in meters


class ImageToGeometry:
    """
    Convert images to computational geometry domains.
    
    Supports:
    - PNG/JPG/BMP images (black = boundary/outside, white = inside)
    - Grayscale and color images
    - Automatic boundary detection
    - SDF computation for physics-informed training
    """
    
    def __init__(
        self,
        target_resolution: int = 128,
        physical_size: Tuple[float, float] = (0.1, 0.1),  # 10cm x 10cm default
        threshold: float = 0.5,
        invert: bool = False,
    ):
        """
        Initialize image-to-geometry converter.
        
        Args:
            target_resolution: Target resolution for the domain
            physical_size: Physical size (height, width) in meters
            threshold: Binarization threshold (0-1)
            invert: If True, black = inside, white = outside
        """
        self.target_resolution = target_resolution
        self.physical_size = physical_size
        self.threshold = threshold
        self.invert = invert
    
    def process(self, image_path: Union[str, Path]) -> DomainInfo:
        """
        Process an image file to create a computational domain.
        
        Args:
            image_path: Path to image file (PNG, JPG, BMP)
            
        Returns:
            DomainInfo object with domain properties
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load and preprocess image
        img = Image.open(image_path)
        
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize to target resolution
        img = img.resize(
            (self.target_resolution, self.target_resolution),
            Image.Resampling.LANCZOS
        )
        
        # Convert to numpy array and normalize
        mask = np.array(img, dtype=np.float32) / 255.0
        
        # Binarize
        mask = (mask > self.threshold).astype(np.float32)
        
        # Invert if needed (some images have black = inside)
        if self.invert:
            mask = 1.0 - mask
        
        return self._create_domain_info(mask)
    
    def process_from_array(self, array: np.ndarray) -> DomainInfo:
        """
        Process a numpy array to create a computational domain.
        
        Args:
            array: Binary or grayscale array
            
        Returns:
            DomainInfo object with domain properties
        """
        # Resize if needed
        if array.shape[0] != self.target_resolution or array.shape[1] != self.target_resolution:
            array = cv2.resize(
                array.astype(np.float32),
                (self.target_resolution, self.target_resolution),
                interpolation=cv2.INTER_LINEAR
            )
        
        # Binarize
        mask = (array > self.threshold).astype(np.float32)
        
        if self.invert:
            mask = 1.0 - mask
        
        return self._create_domain_info(mask)
    
    def _create_domain_info(self, mask: np.ndarray) -> DomainInfo:
        """Create DomainInfo from binary mask."""
        # Compute signed distance function
        sdf = self._compute_sdf(mask)
        
        # Find boundary points
        boundary_points = self._find_boundary_points(mask)
        
        # Compute geometric properties
        h, w = mask.shape
        pixel_area = (self.physical_size[0] / h) * (self.physical_size[1] / w)
        area = np.sum(mask) * pixel_area
        
        perimeter = self._compute_perimeter(mask, pixel_area)
        centroid = self._compute_centroid(mask)
        bbox = self._compute_bounding_box(mask)
        
        return DomainInfo(
            mask=mask,
            sdf=sdf,
            boundary_points=boundary_points,
            area=area,
            perimeter=perimeter,
            centroid=centroid,
            bounding_box=bbox,
            resolution=(h, w),
            physical_size=self.physical_size,
        )
    
    def _compute_sdf(self, mask: np.ndarray) -> np.ndarray:
        """
        Compute signed distance function from binary mask.
        
        Positive values = inside domain
        Negative values = outside domain
        Zero = on boundary
        """
        # Distance transform for inside
        dist_inside = ndimage.distance_transform_edt(mask)
        
        # Distance transform for outside
        dist_outside = ndimage.distance_transform_edt(1 - mask)
        
        # Signed distance: positive inside, negative outside
        sdf = dist_inside - dist_outside
        
        # Normalize to physical units
        h, w = mask.shape
        scale = min(self.physical_size[0] / h, self.physical_size[1] / w)
        sdf = sdf * scale
        
        return sdf.astype(np.float32)
    
    def _find_boundary_points(self, mask: np.ndarray) -> np.ndarray:
        """Find boundary points using edge detection."""
        # Use Sobel operator to find edges
        edges_x = ndimage.sobel(mask, axis=0)
        edges_y = ndimage.sobel(mask, axis=1)
        edges = np.sqrt(edges_x**2 + edges_y**2)
        
        # Threshold to get boundary pixels
        boundary_mask = edges > 0.1
        
        # Get coordinates
        y_coords, x_coords = np.where(boundary_mask)
        
        # Convert to normalized coordinates [0, 1]
        h, w = mask.shape
        x_norm = x_coords / w
        y_norm = y_coords / h
        
        return np.stack([x_norm, y_norm], axis=1).astype(np.float32)
    
    def _compute_perimeter(self, mask: np.ndarray, pixel_area: float) -> float:
        """Compute approximate perimeter length."""
        # Find contours
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        
        if not contours:
            return 0.0
        
        # Sum perimeters of all contours
        pixel_length = np.sqrt(pixel_area)
        total_perimeter = sum(cv2.arcLength(c, closed=True) for c in contours)
        
        return total_perimeter * pixel_length
    
    def _compute_centroid(self, mask: np.ndarray) -> Tuple[float, float]:
        """Compute centroid of the domain."""
        h, w = mask.shape
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        total_mass = np.sum(mask)
        if total_mass == 0:
            return (0.5, 0.5)
        
        cx = np.sum(x_coords * mask) / total_mass / w
        cy = np.sum(y_coords * mask) / total_mass / h
        
        return (cx, cy)
    
    def _compute_bounding_box(
        self, mask: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """Compute bounding box in normalized coordinates."""
        y_coords, x_coords = np.where(mask > 0)
        
        if len(x_coords) == 0:
            return (0.0, 0.0, 1.0, 1.0)
        
        h, w = mask.shape
        x_min = x_coords.min() / w
        x_max = x_coords.max() / w
        y_min = y_coords.min() / h
        y_max = y_coords.max() / h
        
        return (x_min, y_min, x_max, y_max)


class HeatSinkGenerator:
    """
    Generate parametric heat sink geometries.
    
    Supports common heat sink configurations:
    - Straight fins
    - Pin fins
    - Corrugated fins
    - Custom configurations
    """
    
    def __init__(
        self,
        resolution: int = 128,
        physical_size: Tuple[float, float] = (0.1, 0.1),
    ):
        self.resolution = resolution
        self.physical_size = physical_size
        self.converter = ImageToGeometry(resolution, physical_size)
    
    def create_straight_fin_heatsink(
        self,
        base_height: float = 0.1,    # Fraction of total height
        num_fins: int = 5,
        fin_height: float = 0.6,     # Fraction of total height
        fin_width: float = 0.08,     # Fraction of total width
        fin_spacing: float = None,   # Auto-calculated if None
    ) -> DomainInfo:
        """
        Create a straight-fin heat sink geometry.
        
        Args:
            base_height: Height of the base plate (fraction)
            num_fins: Number of fins
            fin_height: Height of each fin (fraction)
            fin_width: Width of each fin (fraction)
            fin_spacing: Spacing between fins (auto if None)
            
        Returns:
            DomainInfo for the heat sink geometry
        """
        mask = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        
        # Create base plate
        base_pixels = int(base_height * self.resolution)
        mask[:base_pixels, :] = 1.0
        
        # Calculate fin positions
        if fin_spacing is None:
            total_fin_width = num_fins * fin_width
            remaining = 1.0 - total_fin_width
            fin_spacing = remaining / (num_fins + 1)
        
        fin_width_px = int(fin_width * self.resolution)
        fin_height_px = int(fin_height * self.resolution)
        
        for i in range(num_fins):
            x_start = int((fin_spacing * (i + 1) + fin_width * i) * self.resolution)
            x_end = x_start + fin_width_px
            y_end = base_pixels + fin_height_px
            
            mask[base_pixels:y_end, x_start:x_end] = 1.0
        
        return self.converter.process_from_array(mask)
    
    def create_pin_fin_heatsink(
        self,
        base_height: float = 0.1,
        num_pins_x: int = 4,
        num_pins_y: int = 4,
        pin_radius: float = 0.05,
        pin_height: float = 0.6,
    ) -> DomainInfo:
        """
        Create a pin-fin heat sink geometry (top view).
        
        For 2D simulation, this creates a cross-section showing
        pin positions on the base.
        """
        mask = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        
        # Create base plate (full coverage for top view)
        mask[:, :] = 1.0
        
        # Create pin positions as higher thermal mass regions
        # This is represented by the mask - pins are part of domain
        x = np.linspace(0, 1, self.resolution)
        y = np.linspace(0, 1, self.resolution)
        X, Y = np.meshgrid(x, y)
        
        spacing_x = 1.0 / (num_pins_x + 1)
        spacing_y = 1.0 / (num_pins_y + 1)
        
        for i in range(num_pins_x):
            for j in range(num_pins_y):
                cx = spacing_x * (i + 1)
                cy = spacing_y * (j + 1)
                
                dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
                pin_mask = (dist <= pin_radius)
                mask[pin_mask] = 1.0
        
        return self.converter.process_from_array(mask)
    
    def create_l_shaped_domain(
        self,
        outer_width: float = 0.8,
        outer_height: float = 0.8,
        cutout_width: float = 0.4,
        cutout_height: float = 0.4,
    ) -> DomainInfo:
        """Create an L-shaped domain (common for corner heat sinks)."""
        mask = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        
        # Create outer rectangle
        x_end = int(outer_width * self.resolution)
        y_end = int(outer_height * self.resolution)
        mask[:y_end, :x_end] = 1.0
        
        # Create cutout
        cut_x_start = int((outer_width - cutout_width) * self.resolution)
        cut_y_start = int((outer_height - cutout_height) * self.resolution)
        mask[cut_y_start:y_end, cut_x_start:x_end] = 0.0
        
        return self.converter.process_from_array(mask)
    
    def create_custom_from_points(
        self,
        points: List[Tuple[float, float]],
    ) -> DomainInfo:
        """
        Create a custom polygon domain from vertex points.
        
        Args:
            points: List of (x, y) coordinates in [0, 1] range
            
        Returns:
            DomainInfo for the polygon domain
        """
        mask = np.zeros((self.resolution, self.resolution), dtype=np.uint8)
        
        # Convert normalized points to pixel coordinates
        pts = np.array([
            [int(p[0] * self.resolution), int(p[1] * self.resolution)]
            for p in points
        ], dtype=np.int32)
        
        # Fill polygon
        cv2.fillPoly(mask, [pts], 1)
        
        return self.converter.process_from_array(mask.astype(np.float32))
