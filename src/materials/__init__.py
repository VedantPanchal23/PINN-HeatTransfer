"""
Materials module for thermal property database and mixture calculations.
"""

from .database import MaterialDatabase, Material
from .mixture import MixtureCalculator, EffectiveProperties
from .thermal_limits import ThermalLimitAnalyzer

__all__ = [
    "MaterialDatabase",
    "Material",
    "MixtureCalculator",
    "EffectiveProperties",
    "ThermalLimitAnalyzer",
]
