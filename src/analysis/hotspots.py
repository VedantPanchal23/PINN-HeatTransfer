"""
Hotspot detection and analysis for thermal simulations.

Identifies critical thermal regions in the temperature field.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy import ndimage


@dataclass
class Hotspot:
    """Information about a thermal hotspot."""
    location: Tuple[float, float]    # (x, y) normalized coordinates
    temperature: float               # Maximum temperature at hotspot
    area: float                      # Approximate area of hotspot region
    severity: str                    # "low", "medium", "high", "critical"
    time_first_detected: float       # Time when hotspot first appeared
    
    def to_dict(self) -> dict:
        return {
            "location": self.location,
            "temperature": self.temperature,
            "area": self.area,
            "severity": self.severity,
            "time_first_detected": self.time_first_detected,
        }


class HotspotDetector:
    """
    Detect and analyze thermal hotspots in temperature fields.
    
    Uses local maxima detection and region growing to identify
    areas of elevated temperature.
    """
    
    def __init__(
        self,
        threshold_percentile: float = 90.0,
        min_hotspot_area: float = 0.01,
        critical_temp: Optional[float] = None,
    ):
        """
        Initialize hotspot detector.
        
        Args:
            threshold_percentile: Temperature percentile for hotspot threshold
            min_hotspot_area: Minimum area for a region to be considered a hotspot
            critical_temp: Optional critical temperature for severity classification
        """
        self.threshold_percentile = threshold_percentile
        self.min_hotspot_area = min_hotspot_area
        self.critical_temp = critical_temp
    
    def detect(
        self,
        temperature_field: np.ndarray,
        time_index: int = -1,
        max_operating_temp: Optional[float] = None,
    ) -> List[Hotspot]:
        """
        Detect hotspots in a temperature field.
        
        Args:
            temperature_field: Temperature array [T, H, W] or [H, W]
            time_index: Time index to analyze (default: last timestep)
            max_operating_temp: Maximum operating temperature for severity
            
        Returns:
            List of detected hotspots
        """
        # Validate input
        if temperature_field is None or temperature_field.size == 0:
            return []
        
        # Handle NaN and Inf values
        if not np.isfinite(temperature_field).all():
            temperature_field = np.nan_to_num(temperature_field, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Handle 3D vs 2D input
        if temperature_field.ndim == 3:
            if temperature_field.shape[0] == 0:
                return []
            T_field = temperature_field[time_index]
            time_fields = temperature_field
        else:
            T_field = temperature_field
            time_fields = temperature_field[np.newaxis, ...]
        
        h, w = T_field.shape
        
        # Handle empty or uniform temperature fields
        if h == 0 or w == 0 or np.std(T_field) < 1e-10:
            return []
        
        # Calculate threshold
        threshold = np.percentile(T_field, self.threshold_percentile)
        
        # Create binary mask of hot regions
        hot_mask = T_field > threshold
        
        # Label connected components
        labeled, num_features = ndimage.label(hot_mask)
        
        hotspots = []
        
        for i in range(1, num_features + 1):
            region_mask = labeled == i
            
            # Calculate area (normalized)
            area = np.sum(region_mask) / (h * w)
            
            # Skip small regions
            if area < self.min_hotspot_area:
                continue
            
            # Find maximum temperature in region
            region_temps = T_field[region_mask]
            max_temp = np.max(region_temps)
            
            # Find location of maximum
            max_idx = np.unravel_index(
                np.argmax(T_field * region_mask), T_field.shape
            )
            location = (max_idx[1] / w, max_idx[0] / h)  # (x, y) normalized
            
            # Determine severity
            severity = self._classify_severity(max_temp, max_operating_temp)
            
            # Find when hotspot first appeared
            first_time = self._find_first_detection(
                time_fields, location, threshold
            )
            
            hotspots.append(Hotspot(
                location=location,
                temperature=max_temp,
                area=area,
                severity=severity,
                time_first_detected=first_time,
            ))
        
        # Sort by temperature (hottest first)
        hotspots.sort(key=lambda h: h.temperature, reverse=True)
        
        return hotspots
    
    def _classify_severity(
        self,
        temperature: float,
        max_operating_temp: Optional[float],
    ) -> str:
        """Classify hotspot severity."""
        if max_operating_temp is None:
            # Use relative classification
            if self.critical_temp and temperature >= self.critical_temp:
                return "critical"
            return "medium"
        
        ratio = temperature / max_operating_temp
        
        if ratio >= 1.0:
            return "critical"
        elif ratio >= 0.9:
            return "high"
        elif ratio >= 0.75:
            return "medium"
        else:
            return "low"
    
    def _find_first_detection(
        self,
        time_fields: np.ndarray,
        location: Tuple[float, float],
        threshold: float,
    ) -> float:
        """Find the first time index where hotspot appears."""
        if time_fields.ndim != 3:
            return 0.0
        
        num_times, h, w = time_fields.shape
        x_idx = int(location[0] * w)
        y_idx = int(location[1] * h)
        
        # Clamp indices
        x_idx = min(max(x_idx, 0), w - 1)
        y_idx = min(max(y_idx, 0), h - 1)
        
        for t in range(num_times):
            if time_fields[t, y_idx, x_idx] > threshold:
                return t / num_times  # Normalized time
        
        return 1.0
    
    def track_hotspots(
        self,
        temperature_field: np.ndarray,
    ) -> List[List[Hotspot]]:
        """
        Track hotspots across all timesteps.
        
        Args:
            temperature_field: Temperature array [T, H, W]
            
        Returns:
            List of hotspot lists for each timestep
        """
        if temperature_field.ndim != 3:
            return [self.detect(temperature_field)]
        
        num_times = temperature_field.shape[0]
        all_hotspots = []
        
        for t in range(num_times):
            hotspots = self.detect(temperature_field, time_index=t)
            all_hotspots.append(hotspots)
        
        return all_hotspots
    
    def get_worst_hotspot(
        self,
        temperature_field: np.ndarray,
    ) -> Optional[Hotspot]:
        """Get the single worst hotspot across all time."""
        if temperature_field.ndim == 3:
            # Find time with maximum temperature
            max_temps = temperature_field.max(axis=(1, 2))
            worst_time = np.argmax(max_temps)
            hotspots = self.detect(temperature_field, time_index=worst_time)
        else:
            hotspots = self.detect(temperature_field)
        
        return hotspots[0] if hotspots else None
    
    def summary(self, hotspots: List[Hotspot]) -> str:
        """Generate summary of detected hotspots."""
        if not hotspots:
            return "No significant hotspots detected."
        
        lines = [
            f"Detected {len(hotspots)} hotspot(s):",
            "",
        ]
        
        for i, h in enumerate(hotspots):
            lines.append(
                f"  {i+1}. Location: ({h.location[0]:.2f}, {h.location[1]:.2f})"
            )
            lines.append(f"      Temperature: {h.temperature:.1f}°C")
            lines.append(f"      Severity: {h.severity.upper()}")
            lines.append(f"      Area: {h.area*100:.1f}% of domain")
            lines.append("")
        
        return "\n".join(lines)
