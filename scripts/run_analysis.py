"""
Main thermal analysis script.

Usage:
    python scripts/run_analysis.py --config configs/analysis.yaml
    python scripts/run_analysis.py --material "70% aluminum, 30% copper" --power 100
"""

import argparse
import sys
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.materials import MaterialDatabase, MixtureCalculator
from src.materials.thermal_limits import ThermalLimitAnalyzer
from src.geometry.image_processor import ImageToGeometry, HeatSinkGenerator, DomainInfo
from src.geometry.heat_sources import HeatSource, HeatSourceType, HeatSourceConfiguration
from src.pinn.boundary_conditions import (
    BCType, BoundaryLocation, BoundaryCondition, BoundaryConditionSet
)
from src.pinn.enhanced_solver import EnhancedPINNSolver, EnhancedPINNConfig
from src.analysis.hotspots import HotspotDetector
from src.analysis.performance import PerformanceAnalyzer
from src.analysis.recommendations import RecommendationEngine
from src.validation.fdm_solver import FDMSolver, FDMConfig
from src.validation.comparison import ValidationComparison
from src.visualization.reports import ReportGenerator, ReportConfig


def parse_material_string(material_str: str) -> dict:
    """
    Parse material string like "70% aluminum, 30% copper" to composition dict.
    
    Returns:
        Dictionary mapping material names to fractions
    """
    composition = {}
    
    # Split by comma
    parts = material_str.split(',')
    
    for part in parts:
        part = part.strip()
        
        # Extract percentage and material name
        if '%' in part:
            pct_str, name = part.split('%')
            percentage = float(pct_str.strip())
            material_name = name.strip().lower().replace(' ', '_')
            composition[material_name] = percentage / 100.0
        else:
            # Assume 100% single material
            material_name = part.lower().replace(' ', '_')
            composition[material_name] = 1.0
    
    # Normalize to sum to 1
    total = sum(composition.values())
    if total > 0:
        composition = {k: v/total for k, v in composition.items()}
    
    return composition


def create_heat_source_config(
    power: float,
    source_type: str = "point",
    position: tuple = (0.5, 0.5),
    size: float = 0.01,
) -> HeatSourceConfiguration:
    """Create heat source configuration."""
    
    type_map = {
        "point": HeatSourceType.POINT,
        "rectangular": HeatSourceType.RECTANGULAR,
        "circular": HeatSourceType.CIRCULAR,
        "gaussian": HeatSourceType.GAUSSIAN,
    }
    
    # Size can be scalar or tuple depending on type
    if source_type.lower() == "rectangular":
        size_param = (size, size)  # (width, height)
    else:
        size_param = size  # radius for circular/gaussian
    
    source = HeatSource(
        source_type=type_map.get(source_type.lower(), HeatSourceType.POINT),
        position=position,
        power=power,
        size=size_param,
    )
    
    return HeatSourceConfiguration(sources=[source])


def create_boundary_conditions(
    config: dict,
    domain_info: DomainInfo,
) -> BoundaryConditionSet:
    """Create boundary conditions from config."""
    
    bc_set = BoundaryConditionSet()
    bc_set.conditions = {}
    bc_set.domain_bounds = (
        0, 0, 
        domain_info.physical_size[1], 
        domain_info.physical_size[0]
    )
    
    # Default: all boundaries at ambient temperature
    ambient = config.get('ambient_temperature', 25.0)
    h_conv = config.get('convection_coefficient', 10.0)
    
    bc_config = config.get('boundary_conditions', {})
    
    location_map = {
        'left': BoundaryLocation.LEFT,
        'right': BoundaryLocation.RIGHT,
        'top': BoundaryLocation.TOP,
        'bottom': BoundaryLocation.BOTTOM,
    }
    
    for loc_name, loc_enum in location_map.items():
        bc_type = bc_config.get(loc_name, 'convective')
        
        if bc_type == 'fixed' or bc_type == 'dirichlet':
            temp = bc_config.get(f'{loc_name}_temp', ambient)
            bc_set.set_dirichlet(loc_enum, temp)
        elif bc_type == 'convective' or bc_type == 'robin':
            bc_set.set_convective(loc_enum, h_conv, ambient)
        elif bc_type == 'adiabatic' or bc_type == 'neumann':
            bc_set.set_neumann(loc_enum, 0.0)
    
    return bc_set


def run_analysis(config: dict):
    """Run complete thermal analysis pipeline."""
    
    print("=" * 60)
    print("PINN THERMAL ANALYSIS")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. Setup materials
    print("[1/7] Setting up materials...")
    
    material_str = config.get('material', 'aluminum')
    composition = parse_material_string(material_str)
    
    db = MaterialDatabase()
    calc = MixtureCalculator(db)
    
    try:
        props = calc.calculate(composition)
        print(f"  Material: {material_str}")
        print(f"  Thermal conductivity: {props.thermal_conductivity:.1f} W/(m·K)")
        print(f"  Density: {props.density:.0f} kg/m³")
        print(f"  Specific heat: {props.specific_heat:.0f} J/(kg·K)")
        print(f"  Max operating temp: {props.max_operating_temp:.0f}°C")
    except Exception as e:
        print(f"  Error: {e}")
        return None
    
    # 2. Setup geometry
    print("\n[2/7] Setting up geometry...")
    
    geometry_config = config.get('geometry', {})
    geometry_type = geometry_config.get('type', 'rectangle')
    
    if geometry_type == 'image':
        image_path = geometry_config.get('path')
        processor = ImageToGeometry()
        
        from PIL import Image
        img = np.array(Image.open(image_path).convert('L'))
        
        physical_size = tuple(geometry_config.get('physical_size', [0.1, 0.1]))
        domain_info = processor.process(img, physical_size=physical_size)
        
    elif geometry_type == 'heatsink':
        generator = HeatSinkGenerator()
        
        domain_info = generator.generate_straight_fin_heatsink(
            base_height=geometry_config.get('base_height', 0.003),
            fin_height=geometry_config.get('fin_height', 0.02),
            fin_width=geometry_config.get('fin_width', 0.002),
            num_fins=geometry_config.get('num_fins', 7),
            domain_width=geometry_config.get('domain_width', 0.1),
            resolution=geometry_config.get('resolution', 100),
        )
        
    else:  # rectangle
        width = geometry_config.get('width', 0.1)
        height = geometry_config.get('height', 0.1)
        nx = geometry_config.get('nx', 100)
        ny = geometry_config.get('ny', 100)
        
        mask = np.ones((ny, nx), dtype=np.float32)
        
        # Create minimal boundary points (corners)
        boundary_points = np.array([[0, 0], [width, 0], [width, height], [0, height]])
        
        domain_info = DomainInfo(
            mask=mask,
            sdf=np.zeros_like(mask),
            boundary_points=boundary_points,
            area=width * height,
            perimeter=2 * (width + height),
            centroid=(width/2, height/2),
            bounding_box=(0, 0, width, height),
            resolution=(ny, nx),
            physical_size=(height, width),
        )
    
    print(f"  Geometry type: {geometry_type}")
    print(f"  Domain size: {domain_info.physical_size[1]*100:.1f} × {domain_info.physical_size[0]*100:.1f} cm (W × H)")
    print(f"  Grid resolution: {domain_info.resolution[1]} × {domain_info.resolution[0]} (W × H)")
    
    # Create coordinate grids for later use
    x_coords = np.linspace(0, domain_info.physical_size[1], domain_info.resolution[1])
    y_coords = np.linspace(0, domain_info.physical_size[0], domain_info.resolution[0])
    
    # 3. Setup heat sources
    print("\n[3/7] Setting up heat sources...")
    
    power = config.get('power', 100.0)
    source_config = config.get('heat_source', {})
    
    heat_config = create_heat_source_config(
        power=power,
        source_type=source_config.get('type', 'point'),
        position=tuple(source_config.get('position', [0.5, 0.5])),
        size=source_config.get('size', 0.01),
    )
    
    print(f"  Heat source power: {power}W")
    print(f"  Source type: {source_config.get('type', 'point')}")
    
    # 4. Setup boundary conditions
    print("\n[4/7] Setting up boundary conditions...")
    
    bc_set = create_boundary_conditions(config, domain_info)
    
    for loc, bc in bc_set.conditions.items():
        print(f"  {loc.name}: {bc.bc_type.name}")
    
    # 5. Run PINN simulation
    print("\n[5/7] Running PINN simulation...")
    
    solver_config = config.get('solver', {})
    
    # For demonstration, use FDM instead of full PINN training
    print("  Using FDM solver for quick demonstration...")
    
    fdm_config = FDMConfig(
        nx=domain_info.resolution[1],
        ny=domain_info.resolution[0],
        Lx=domain_info.physical_size[1],
        Ly=domain_info.physical_size[0],
        t_max=config.get('simulation_time', 1.0),
        dt=0.001,
    )
    
    fdm = FDMSolver(config=fdm_config)
    
    initial_temp = config.get('initial_temperature', 25.0)
    
    T_history, times = fdm.solve(
        alpha=props.thermal_diffusivity,
        initial_temperature=initial_temp,
        boundary_temperatures={'bottom': config.get('ambient_temperature', 25.0)},
        show_progress=True,
    )
    
    T_final = T_history[-1]
    
    # Apply heat source effect (simplified)
    X, Y = np.meshgrid(x_coords, y_coords)
    for source in heat_config.sources:
        dist = np.sqrt((X - source.position[0])**2 + (Y - source.position[1])**2)
        T_final += source.power * 0.5 * np.exp(-dist**2 / 0.005) / props.thermal_conductivity
    
    print(f"  Simulation complete!")
    print(f"  Max temperature: {np.max(T_final):.1f}°C")
    print(f"  Mean temperature: {np.mean(T_final):.1f}°C")
    
    # 6. Thermal analysis
    print("\n[6/7] Performing thermal analysis...")
    
    # Hotspot detection
    hotspot_detector = HotspotDetector(threshold_percentile=95)
    hotspots = hotspot_detector.detect(
        T_final,
        max_operating_temp=props.max_operating_temp,
    )
    
    print(f"  Detected {len(hotspots)} hotspot(s)")
    
    for hs in hotspots:
        print(f"    - {hs.temperature:.1f}°C at ({hs.x:.3f}, {hs.y:.3f}m)")
    
    # Thermal limit analysis
    limit_analyzer = ThermalLimitAnalyzer()
    limit_result = limit_analyzer.analyze(
        material_props=props,
        heat_source_power=power,
        domain_area=domain_info.area,
        domain_thickness=0.005,  # 5mm thickness assumption
        ambient_temp=config.get('ambient_temperature', 25.0),
        predicted_temps=T_final.flatten(),
        predicted_times=times,
    )
    
    print(f"  Safety margin: {limit_result.temp_margin:.1f}°C")
    print(f"  Risk level: {limit_result.risk_level.name}")
    
    if limit_result.time_to_max_operating and limit_result.time_to_max_operating < float('inf'):
        print(f"  Estimated time to max operating temp: {limit_result.time_to_max_operating:.1f}s")
    else:
        print(f"  System is thermally stable")
    
    # Performance metrics
    # perf_analyzer = PerformanceAnalyzer()
    # metrics = perf_analyzer.analyze(...)
    
    # 7. Generate recommendations
    print("\n[7/7] Generating recommendations...")
    
    # Simple recommendations based on results
    recommendations = []
    
    max_T = np.max(T_final)
    temp_range = np.max(T_final) - np.min(T_final)
    
    if max_T > props.max_operating_temp * 0.8:
        recommendations.append({
            'priority': 'high',
            'title': 'High Temperature Warning',
            'description': f'Max temp ({max_T:.1f}°C) is close to material limit ({props.max_operating_temp:.0f}°C)'
        })
    
    if temp_range > 50:
        recommendations.append({
            'priority': 'medium',
            'title': 'Large Temperature Gradient',
            'description': f'Temperature range is {temp_range:.1f}°C. Consider improving heat spreading.'
        })
    
    if props.thermal_conductivity < 100:
        recommendations.append({
            'priority': 'medium',
            'title': 'Material Conductivity',
            'description': 'Consider higher conductivity materials like copper for better cooling.'
        })
    
    if not recommendations:
        recommendations.append({
            'priority': 'low',
            'title': 'Good Thermal Design',
            'description': 'System is operating well within thermal limits.'
        })
    
    print(f"  Generated {len(recommendations)} recommendation(s)")
    
    for rec in recommendations[:3]:
        print(f"    [{rec['priority'].upper()}] {rec['title']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    results = {
        'temperature_field': T_final,
        'temperature_history': T_history,
        'times': times,
        'max_temp': np.max(T_final),
        'mean_temp': np.mean(T_final),
        'min_temp': np.min(T_final),
        'hotspots': hotspots,
        'thermal_limits': limit_result,
        'material_properties': {
            'thermal_conductivity': props.thermal_conductivity,
            'density': props.density,
            'specific_heat': props.specific_heat,
            'max_operating_temp': props.max_operating_temp,
        },
        'recommendations': recommendations,
    }
    
    # Generate report if requested
    if config.get('generate_report', False):
        output_dir = Path(config.get('output_dir', 'outputs'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_gen = ReportGenerator()
        
        pdf_path = output_dir / f"thermal_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        report_gen.generate_pdf_report(results, str(pdf_path))
        
        print(f"\nReport saved to: {pdf_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="PINN Thermal Analysis Tool"
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--material', '-m',
        type=str,
        default='aluminum',
        help='Material specification (e.g., "70%% aluminum, 30%% copper")'
    )
    
    parser.add_argument(
        '--power', '-p',
        type=float,
        default=100.0,
        help='Heat source power in Watts'
    )
    
    parser.add_argument(
        '--initial-temp',
        type=float,
        default=25.0,
        help='Initial temperature in Celsius'
    )
    
    parser.add_argument(
        '--ambient-temp',
        type=float,
        default=25.0,
        help='Ambient temperature in Celsius'
    )
    
    parser.add_argument(
        '--time',
        type=float,
        default=1.0,
        help='Simulation time in seconds'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Load config from file or command line
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'material': args.material,
            'power': args.power,
            'initial_temperature': args.initial_temp,
            'ambient_temperature': args.ambient_temp,
            'simulation_time': args.time,
            'output_dir': args.output,
            'geometry': {
                'type': 'rectangle',
                'width': 0.1,
                'height': 0.1,
                'nx': 100,
                'ny': 100,
            },
            'solver': {
                'num_epochs': 1000,
                'fdm_method': 'implicit',
            },
            'boundary_conditions': {
                'top': 'convective',
                'bottom': 'fixed',
                'left': 'adiabatic',
                'right': 'adiabatic',
            },
        }
    
    # Run analysis
    results = run_analysis(config)
    
    if results:
        print(f"\nMax temperature: {results['max_temp']:.1f}°C")
        print(f"Risk level: {results['thermal_limits'].risk_level.name}")


if __name__ == '__main__':
    main()
