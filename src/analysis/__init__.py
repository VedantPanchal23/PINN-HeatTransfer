"""
Analysis module for thermal simulation results.

Provides:
- Hotspot detection
- Performance metrics
- Thermal resistance calculation
- Efficiency analysis
"""

from .hotspots import HotspotDetector, Hotspot
from .performance import PerformanceAnalyzer, ThermalPerformanceMetrics
from .recommendations import RecommendationEngine

__all__ = [
    "HotspotDetector",
    "Hotspot",
    "PerformanceAnalyzer",
    "ThermalPerformanceMetrics",
    "RecommendationEngine",
]
