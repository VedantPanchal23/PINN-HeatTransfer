"""
Comprehensive material database for thermal simulations.

Contains thermal properties for 30+ common engineering materials including:
- Thermal conductivity (k)
- Density (ρ)
- Specific heat capacity (c)
- Thermal diffusivity (α = k / (ρ * c))
- Melting point
- Maximum safe operating temperature
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
import json
from pathlib import Path


@dataclass
class Material:
    """Material thermal properties."""
    name: str
    thermal_conductivity: float  # W/(m·K)
    density: float               # kg/m³
    specific_heat: float         # J/(kg·K)
    melting_point: float         # °C
    max_operating_temp: float    # °C (safe continuous use)
    category: str                # metal, ceramic, polymer, composite
    
    @property
    def thermal_diffusivity(self) -> float:
        """Calculate thermal diffusivity α = k / (ρ * c) [m²/s]."""
        return self.thermal_conductivity / (self.density * self.specific_heat)
    
    @property
    def volumetric_heat_capacity(self) -> float:
        """Calculate volumetric heat capacity ρ * c [J/(m³·K)]."""
        return self.density * self.specific_heat
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "thermal_conductivity": self.thermal_conductivity,
            "density": self.density,
            "specific_heat": self.specific_heat,
            "melting_point": self.melting_point,
            "max_operating_temp": self.max_operating_temp,
            "category": self.category,
            "thermal_diffusivity": self.thermal_diffusivity,
        }


# Comprehensive material database
MATERIALS_DATA: Dict[str, Dict] = {
    # ============== METALS ==============
    "aluminum": {
        "thermal_conductivity": 205.0,
        "density": 2700.0,
        "specific_heat": 900.0,
        "melting_point": 660.0,
        "max_operating_temp": 400.0,
        "category": "metal",
    },
    "aluminum_6061": {
        "thermal_conductivity": 167.0,
        "density": 2700.0,
        "specific_heat": 896.0,
        "melting_point": 652.0,
        "max_operating_temp": 400.0,
        "category": "metal",
    },
    "copper": {
        "thermal_conductivity": 385.0,
        "density": 8960.0,
        "specific_heat": 385.0,
        "melting_point": 1085.0,
        "max_operating_temp": 800.0,
        "category": "metal",
    },
    "gold": {
        "thermal_conductivity": 315.0,
        "density": 19300.0,
        "specific_heat": 129.0,
        "melting_point": 1064.0,
        "max_operating_temp": 800.0,
        "category": "metal",
    },
    "silver": {
        "thermal_conductivity": 429.0,
        "density": 10490.0,
        "specific_heat": 235.0,
        "melting_point": 962.0,
        "max_operating_temp": 700.0,
        "category": "metal",
    },
    "iron": {
        "thermal_conductivity": 80.0,
        "density": 7870.0,
        "specific_heat": 450.0,
        "melting_point": 1538.0,
        "max_operating_temp": 1000.0,
        "category": "metal",
    },
    "steel_carbon": {
        "thermal_conductivity": 50.0,
        "density": 7850.0,
        "specific_heat": 490.0,
        "melting_point": 1510.0,
        "max_operating_temp": 800.0,
        "category": "metal",
    },
    "steel_stainless_304": {
        "thermal_conductivity": 16.0,
        "density": 8000.0,
        "specific_heat": 500.0,
        "melting_point": 1450.0,
        "max_operating_temp": 870.0,
        "category": "metal",
    },
    "steel_stainless_316": {
        "thermal_conductivity": 16.3,
        "density": 8000.0,
        "specific_heat": 500.0,
        "melting_point": 1400.0,
        "max_operating_temp": 870.0,
        "category": "metal",
    },
    "titanium": {
        "thermal_conductivity": 21.9,
        "density": 4500.0,
        "specific_heat": 520.0,
        "melting_point": 1668.0,
        "max_operating_temp": 600.0,
        "category": "metal",
    },
    "nickel": {
        "thermal_conductivity": 91.0,
        "density": 8900.0,
        "specific_heat": 444.0,
        "melting_point": 1455.0,
        "max_operating_temp": 1000.0,
        "category": "metal",
    },
    "zinc": {
        "thermal_conductivity": 116.0,
        "density": 7140.0,
        "specific_heat": 388.0,
        "melting_point": 420.0,
        "max_operating_temp": 300.0,
        "category": "metal",
    },
    "brass": {
        "thermal_conductivity": 109.0,
        "density": 8500.0,
        "specific_heat": 380.0,
        "melting_point": 930.0,
        "max_operating_temp": 600.0,
        "category": "metal",
    },
    "bronze": {
        "thermal_conductivity": 50.0,
        "density": 8800.0,
        "specific_heat": 380.0,
        "melting_point": 1000.0,
        "max_operating_temp": 600.0,
        "category": "metal",
    },
    "lead": {
        "thermal_conductivity": 35.0,
        "density": 11340.0,
        "specific_heat": 129.0,
        "melting_point": 327.0,
        "max_operating_temp": 200.0,
        "category": "metal",
    },
    "magnesium": {
        "thermal_conductivity": 156.0,
        "density": 1740.0,
        "specific_heat": 1020.0,
        "melting_point": 650.0,
        "max_operating_temp": 400.0,
        "category": "metal",
    },
    "tungsten": {
        "thermal_conductivity": 173.0,
        "density": 19300.0,
        "specific_heat": 134.0,
        "melting_point": 3422.0,
        "max_operating_temp": 2000.0,
        "category": "metal",
    },
    "inconel_625": {
        "thermal_conductivity": 9.8,
        "density": 8440.0,
        "specific_heat": 410.0,
        "melting_point": 1350.0,
        "max_operating_temp": 980.0,
        "category": "metal",
    },
    
    # ============== CERAMICS ==============
    "alumina": {
        "thermal_conductivity": 30.0,
        "density": 3950.0,
        "specific_heat": 880.0,
        "melting_point": 2072.0,
        "max_operating_temp": 1700.0,
        "category": "ceramic",
    },
    "silicon_carbide": {
        "thermal_conductivity": 120.0,
        "density": 3210.0,
        "specific_heat": 750.0,
        "melting_point": 2730.0,
        "max_operating_temp": 1650.0,
        "category": "ceramic",
    },
    "aluminum_nitride": {
        "thermal_conductivity": 170.0,
        "density": 3260.0,
        "specific_heat": 740.0,
        "melting_point": 2200.0,
        "max_operating_temp": 1000.0,
        "category": "ceramic",
    },
    "zirconia": {
        "thermal_conductivity": 2.0,
        "density": 5680.0,
        "specific_heat": 480.0,
        "melting_point": 2715.0,
        "max_operating_temp": 2200.0,
        "category": "ceramic",
    },
    "glass": {
        "thermal_conductivity": 1.0,
        "density": 2500.0,
        "specific_heat": 840.0,
        "melting_point": 1400.0,
        "max_operating_temp": 500.0,
        "category": "ceramic",
    },
    
    # ============== POLYMERS ==============
    "abs_plastic": {
        "thermal_conductivity": 0.17,
        "density": 1050.0,
        "specific_heat": 1400.0,
        "melting_point": 105.0,
        "max_operating_temp": 80.0,
        "category": "polymer",
    },
    "nylon": {
        "thermal_conductivity": 0.25,
        "density": 1150.0,
        "specific_heat": 1700.0,
        "melting_point": 220.0,
        "max_operating_temp": 120.0,
        "category": "polymer",
    },
    "pvc": {
        "thermal_conductivity": 0.19,
        "density": 1400.0,
        "specific_heat": 900.0,
        "melting_point": 160.0,
        "max_operating_temp": 60.0,
        "category": "polymer",
    },
    "ptfe_teflon": {
        "thermal_conductivity": 0.25,
        "density": 2200.0,
        "specific_heat": 1000.0,
        "melting_point": 327.0,
        "max_operating_temp": 260.0,
        "category": "polymer",
    },
    "epoxy": {
        "thermal_conductivity": 0.2,
        "density": 1200.0,
        "specific_heat": 1000.0,
        "melting_point": 150.0,
        "max_operating_temp": 120.0,
        "category": "polymer",
    },
    
    # ============== COMPOSITES ==============
    "carbon_fiber_composite": {
        "thermal_conductivity": 7.0,
        "density": 1600.0,
        "specific_heat": 800.0,
        "melting_point": 3500.0,
        "max_operating_temp": 300.0,
        "category": "composite",
    },
    "fiberglass": {
        "thermal_conductivity": 0.4,
        "density": 1800.0,
        "specific_heat": 800.0,
        "melting_point": 1100.0,
        "max_operating_temp": 300.0,
        "category": "composite",
    },
    "thermal_paste": {
        "thermal_conductivity": 8.0,
        "density": 2500.0,
        "specific_heat": 800.0,
        "melting_point": 200.0,
        "max_operating_temp": 150.0,
        "category": "composite",
    },
    
    # ============== SEMICONDUCTORS ==============
    "silicon": {
        "thermal_conductivity": 150.0,
        "density": 2330.0,
        "specific_heat": 700.0,
        "melting_point": 1414.0,
        "max_operating_temp": 200.0,
        "category": "semiconductor",
    },
    "germanium": {
        "thermal_conductivity": 60.0,
        "density": 5323.0,
        "specific_heat": 320.0,
        "melting_point": 938.0,
        "max_operating_temp": 150.0,
        "category": "semiconductor",
    },
    "gallium_arsenide": {
        "thermal_conductivity": 55.0,
        "density": 5320.0,
        "specific_heat": 330.0,
        "melting_point": 1238.0,
        "max_operating_temp": 200.0,
        "category": "semiconductor",
    },
}


class MaterialDatabase:
    """
    Material database for looking up thermal properties.
    
    Supports:
    - Getting material by name
    - Searching materials by category
    - Listing all available materials
    - Fuzzy matching for material names
    """
    
    def __init__(self, custom_materials_path: Optional[Path] = None):
        """
        Initialize the material database.
        
        Args:
            custom_materials_path: Optional path to custom materials JSON file
        """
        self._materials: Dict[str, Material] = {}
        
        # Load built-in materials
        for name, props in MATERIALS_DATA.items():
            self._materials[name] = Material(name=name, **props)
        
        # Load custom materials if provided
        if custom_materials_path and custom_materials_path.exists():
            self._load_custom_materials(custom_materials_path)
    
    def _load_custom_materials(self, path: Path) -> None:
        """Load custom materials from JSON file."""
        with open(path, 'r') as f:
            custom_data = json.load(f)
        
        for name, props in custom_data.items():
            self._materials[name.lower()] = Material(name=name, **props)
    
    def get(self, name: str) -> Optional[Material]:
        """
        Get material by name.
        
        Args:
            name: Material name (case-insensitive)
            
        Returns:
            Material object or None if not found
        """
        normalized = name.lower().replace(" ", "_").replace("-", "_")
        return self._materials.get(normalized)
    
    def get_or_raise(self, name: str) -> Material:
        """
        Get material by name or raise error.
        
        Args:
            name: Material name
            
        Returns:
            Material object
            
        Raises:
            ValueError: If material not found
        """
        material = self.get(name)
        if material is None:
            available = ", ".join(list(self._materials.keys())[:10])
            raise ValueError(
                f"Material '{name}' not found. "
                f"Available materials include: {available}..."
            )
        return material
    
    def search(self, query: str) -> List[Material]:
        """
        Search materials by name (partial match).
        
        Args:
            query: Search query
            
        Returns:
            List of matching materials
        """
        query = query.lower()
        return [
            mat for name, mat in self._materials.items()
            if query in name
        ]
    
    def get_by_category(self, category: str) -> List[Material]:
        """
        Get all materials in a category.
        
        Args:
            category: Category name (metal, ceramic, polymer, composite, semiconductor)
            
        Returns:
            List of materials in category
        """
        return [
            mat for mat in self._materials.values()
            if mat.category.lower() == category.lower()
        ]
    
    def list_all(self) -> List[str]:
        """Get list of all material names."""
        return list(self._materials.keys())
    
    def list_categories(self) -> List[str]:
        """Get list of all categories."""
        return list(set(mat.category for mat in self._materials.values()))
    
    def add_material(self, material: Material) -> None:
        """Add a custom material to the database."""
        self._materials[material.name.lower()] = material
    
    def save_custom_materials(self, path: Path) -> None:
        """Save custom materials to JSON file."""
        custom = {
            name: mat.to_dict() 
            for name, mat in self._materials.items()
            if name not in MATERIALS_DATA
        }
        with open(path, 'w') as f:
            json.dump(custom, f, indent=2)
    
    def __len__(self) -> int:
        return len(self._materials)
    
    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None
    
    def __repr__(self) -> str:
        return f"MaterialDatabase({len(self)} materials)"


# Global database instance
_db: Optional[MaterialDatabase] = None


def get_database() -> MaterialDatabase:
    """Get global material database instance."""
    global _db
    if _db is None:
        _db = MaterialDatabase()
    return _db
