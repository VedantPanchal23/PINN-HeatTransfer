"""
Intelligent recommendation engine for thermal design optimization.

Generates actionable suggestions based on simulation results.
"""

from typing import List, Optional, Dict
from dataclasses import dataclass

from .hotspots import Hotspot, HotspotDetector
from .performance import ThermalPerformanceMetrics
from ..materials import EffectiveProperties


@dataclass
class Recommendation:
    """A design recommendation."""
    category: str           # "material", "geometry", "cooling", "design"
    priority: str           # "high", "medium", "low"
    title: str
    description: str
    expected_improvement: str
    
    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "priority": self.priority,
            "title": self.title,
            "description": self.description,
            "expected_improvement": self.expected_improvement,
        }


class RecommendationEngine:
    """
    Generate intelligent recommendations for thermal design improvement.
    
    Analyzes:
    - Temperature distribution and hotspots
    - Material properties
    - Performance metrics
    - Safety margins
    """
    
    def __init__(self):
        self.recommendations: List[Recommendation] = []
    
    def analyze(
        self,
        metrics: ThermalPerformanceMetrics,
        material_props: EffectiveProperties,
        hotspots: List[Hotspot],
        max_operating_temp: float,
        heat_source_power: float,
    ) -> List[Recommendation]:
        """
        Generate recommendations based on simulation results.
        
        Args:
            metrics: Performance metrics
            material_props: Material properties
            hotspots: Detected hotspots
            max_operating_temp: Maximum safe operating temperature
            heat_source_power: Total heat source power
            
        Returns:
            List of recommendations sorted by priority
        """
        self.recommendations = []
        
        # Check temperature safety
        self._check_temperature_safety(metrics, max_operating_temp)
        
        # Check thermal resistance
        self._check_thermal_resistance(metrics, heat_source_power)
        
        # Check material selection
        self._check_material_selection(material_props, metrics)
        
        # Check hotspots
        self._check_hotspots(hotspots, max_operating_temp)
        
        # Check cooling efficiency
        self._check_cooling_efficiency(metrics)
        
        # Check temperature uniformity
        self._check_uniformity(metrics)
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        self.recommendations.sort(key=lambda r: priority_order.get(r.priority, 3))
        
        return self.recommendations
    
    def _check_temperature_safety(
        self,
        metrics: ThermalPerformanceMetrics,
        max_operating_temp: float,
    ) -> None:
        """Check if operating temperature is safe."""
        margin = max_operating_temp - metrics.max_temperature
        margin_ratio = margin / max_operating_temp
        
        if metrics.max_temperature >= max_operating_temp:
            self.recommendations.append(Recommendation(
                category="critical",
                priority="high",
                title="Temperature Exceeds Safe Limit",
                description=f"Maximum temperature ({metrics.max_temperature:.1f}°C) exceeds "
                           f"safe operating limit ({max_operating_temp:.1f}°C). "
                           "Immediate design changes required.",
                expected_improvement="Reduce temperature to safe range",
            ))
        elif margin_ratio < 0.1:
            self.recommendations.append(Recommendation(
                category="cooling",
                priority="high",
                title="Low Temperature Safety Margin",
                description=f"Only {margin:.1f}°C margin to max operating temperature. "
                           "Consider adding active cooling or improving heat dissipation.",
                expected_improvement="10-20% temperature reduction with active cooling",
            ))
        elif margin_ratio < 0.2:
            self.recommendations.append(Recommendation(
                category="cooling",
                priority="medium",
                title="Moderate Temperature Margin",
                description=f"Temperature margin of {margin:.1f}°C is acceptable but could be improved.",
                expected_improvement="Improved reliability at elevated ambient temperatures",
            ))
    
    def _check_thermal_resistance(
        self,
        metrics: ThermalPerformanceMetrics,
        heat_source_power: float,
    ) -> None:
        """Check thermal resistance values."""
        if heat_source_power <= 0:
            return
        
        # Typical thresholds for electronics cooling
        if metrics.thermal_resistance > 5.0:
            self.recommendations.append(Recommendation(
                category="design",
                priority="high",
                title="High Thermal Resistance",
                description=f"Thermal resistance of {metrics.thermal_resistance:.2f} °C/W is high. "
                           "Consider increasing heat sink surface area or adding fins.",
                expected_improvement="30-50% reduction in thermal resistance",
            ))
        elif metrics.thermal_resistance > 2.0:
            self.recommendations.append(Recommendation(
                category="design",
                priority="medium",
                title="Moderate Thermal Resistance",
                description=f"Thermal resistance of {metrics.thermal_resistance:.2f} °C/W could be improved.",
                expected_improvement="20-30% reduction with optimized fin design",
            ))
    
    def _check_material_selection(
        self,
        material_props: EffectiveProperties,
        metrics: ThermalPerformanceMetrics,
    ) -> None:
        """Check material selection and suggest improvements."""
        k = material_props.thermal_conductivity
        
        # Check if copper content could help
        if k < 200 and "copper" not in material_props.composition:
            self.recommendations.append(Recommendation(
                category="material",
                priority="medium",
                title="Consider Adding Copper",
                description=f"Current thermal conductivity is {k:.0f} W/(m·K). "
                           "Adding copper (385 W/(m·K)) could significantly improve heat spreading.",
                expected_improvement="20-40% improvement in thermal conductivity",
            ))
        
        # Check for high-performance alternatives
        if k < 100:
            self.recommendations.append(Recommendation(
                category="material",
                priority="medium",
                title="Low Thermal Conductivity Material",
                description=f"Thermal conductivity of {k:.0f} W/(m·K) limits heat dissipation. "
                           "Consider aluminum (205), copper (385), or aluminum nitride (170) for better performance.",
                expected_improvement="2-4x improvement in heat spreading",
            ))
    
    def _check_hotspots(
        self,
        hotspots: List[Hotspot],
        max_operating_temp: float,
    ) -> None:
        """Check hotspot severity and suggest mitigations."""
        critical_hotspots = [h for h in hotspots if h.severity == "critical"]
        high_hotspots = [h for h in hotspots if h.severity == "high"]
        
        if critical_hotspots:
            h = critical_hotspots[0]
            self.recommendations.append(Recommendation(
                category="geometry",
                priority="high",
                title="Critical Hotspot Detected",
                description=f"Critical hotspot at ({h.location[0]:.2f}, {h.location[1]:.2f}) "
                           f"reaching {h.temperature:.1f}°C. Add local cooling or heat spreading.",
                expected_improvement="Eliminate critical hotspot",
            ))
        
        if high_hotspots:
            self.recommendations.append(Recommendation(
                category="geometry",
                priority="medium",
                title="High-Severity Hotspots",
                description=f"{len(high_hotspots)} high-severity hotspot(s) detected. "
                           "Consider redistributing heat sources or adding fins at hotspot locations.",
                expected_improvement="10-20% reduction in hotspot temperature",
            ))
    
    def _check_cooling_efficiency(
        self,
        metrics: ThermalPerformanceMetrics,
    ) -> None:
        """Check cooling efficiency."""
        if metrics.cooling_efficiency < 50:
            self.recommendations.append(Recommendation(
                category="cooling",
                priority="high",
                title="Low Cooling Efficiency",
                description=f"Cooling efficiency of {metrics.cooling_efficiency:.1f}% is poor. "
                           "Heat is not being effectively removed from the system.",
                expected_improvement="Improve to >80% efficiency with forced convection",
            ))
        elif metrics.cooling_efficiency < 80:
            self.recommendations.append(Recommendation(
                category="cooling",
                priority="medium",
                title="Moderate Cooling Efficiency",
                description=f"Cooling efficiency of {metrics.cooling_efficiency:.1f}% is acceptable "
                           "but could be improved with better airflow or larger heat sink.",
                expected_improvement="10-20% improvement in efficiency",
            ))
    
    def _check_uniformity(
        self,
        metrics: ThermalPerformanceMetrics,
    ) -> None:
        """Check temperature uniformity."""
        temp_range = metrics.max_temperature - metrics.min_temperature
        
        if temp_range > 50:
            self.recommendations.append(Recommendation(
                category="design",
                priority="medium",
                title="High Temperature Non-Uniformity",
                description=f"Temperature varies by {temp_range:.1f}°C across the domain. "
                           "This may cause thermal stress. Consider heat spreading layers.",
                expected_improvement="Reduce temperature variation by 30-50%",
            ))
        
        if metrics.max_temperature_gradient > 20:
            self.recommendations.append(Recommendation(
                category="design",
                priority="medium",
                title="High Temperature Gradient",
                description=f"Maximum gradient of {metrics.max_temperature_gradient:.1f} °C/mm "
                           "may cause thermal stress and reliability issues.",
                expected_improvement="Reduce gradient with thermal interface materials",
            ))
    
    def generate_report(self) -> str:
        """Generate a formatted recommendations report."""
        if not self.recommendations:
            return "No recommendations - design meets all criteria."
        
        lines = [
            "=" * 60,
            "THERMAL DESIGN RECOMMENDATIONS",
            "=" * 60,
            "",
        ]
        
        # Group by priority
        high_priority = [r for r in self.recommendations if r.priority == "high"]
        medium_priority = [r for r in self.recommendations if r.priority == "medium"]
        low_priority = [r for r in self.recommendations if r.priority == "low"]
        
        if high_priority:
            lines.append("🔴 HIGH PRIORITY:")
            lines.append("-" * 40)
            for r in high_priority:
                lines.append(f"  [{r.category.upper()}] {r.title}")
                lines.append(f"    {r.description}")
                lines.append(f"    → Expected: {r.expected_improvement}")
                lines.append("")
        
        if medium_priority:
            lines.append("🟡 MEDIUM PRIORITY:")
            lines.append("-" * 40)
            for r in medium_priority:
                lines.append(f"  [{r.category.upper()}] {r.title}")
                lines.append(f"    {r.description}")
                lines.append(f"    → Expected: {r.expected_improvement}")
                lines.append("")
        
        if low_priority:
            lines.append("🟢 LOW PRIORITY:")
            lines.append("-" * 40)
            for r in low_priority:
                lines.append(f"  [{r.category.upper()}] {r.title}")
                lines.append(f"    {r.description}")
                lines.append(f"    → Expected: {r.expected_improvement}")
                lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
