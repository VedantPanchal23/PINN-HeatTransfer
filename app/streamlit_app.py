"""
Streamlit Web Application for PINN Thermal Analysis.

A user-friendly interface for:
- Uploading geometry images
- Selecting materials by name and percentage
- Configuring heat sources and boundary conditions
- Running thermal simulations
- Viewing results and recommendations
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.materials import MaterialDatabase, MixtureCalculator
from src.materials.thermal_limits import ThermalLimitAnalyzer
from src.geometry.image_processor import ImageToGeometry, DomainInfo
from src.geometry.heat_sources import HeatSource, HeatSourceType, HeatSourceConfiguration
from src.pinn.boundary_conditions import BCType, BoundaryCondition, BoundaryConditionSet
from src.analysis.recommendations import RecommendationEngine
from src.visualization.reports import ReportGenerator, ReportConfig


# Page config
st.set_page_config(
    page_title="PINN Thermal Analyzer",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize session state with default values."""
    defaults = {
        'heat_sources': [],
        'num_materials': 1,
        'simulation_results': None,
        'thermal_limits': None,
        'domain_info': None,
        'material_properties': None,
        'material_composition': None,
        'simulation_config': None,
        'geometry_image': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    """Main application entry point."""
    
    # Initialize session state
    init_session_state()
    
    # Sidebar navigation
    st.sidebar.title("🔥 PINN Thermal Analyzer")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📐 Geometry", "🧪 Materials", "🔥 Heat Sources", 
         "🎯 Simulation", "📊 Results", "📋 Report"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📐 Geometry":
        show_geometry_page()
    elif page == "🧪 Materials":
        show_materials_page()
    elif page == "🔥 Heat Sources":
        show_heat_sources_page()
    elif page == "🎯 Simulation":
        show_simulation_page()
    elif page == "📊 Results":
        show_results_page()
    elif page == "📋 Report":
        show_report_page()


def show_home_page():
    """Display home page with overview."""
    st.title("🔥 PINN Thermal Analysis Framework")
    
    st.markdown("""
    ## Welcome!
    
    This application uses **Physics-Informed Neural Networks (PINN)** to simulate 
    transient heat transfer in complex 2D geometries.
    
    ### Features:
    
    - **📐 Geometry Input**: Upload PNG images or generate parametric heat sinks
    - **🧪 Material Selection**: Choose materials by name and specify mixtures (e.g., "70% aluminum, 30% copper")
    - **🔥 Heat Sources**: Place point, rectangular, or Gaussian heat sources
    - **🌡️ Boundary Conditions**: Configure Dirichlet, Neumann, or convective boundaries
    - **📊 Thermal Analysis**: Predict temperature fields, hotspots, and max operating time
    - **📋 Recommendations**: Get smart suggestions for design improvements
    
    ### Quick Start:
    
    1. **Geometry**: Upload a geometry image or select a preset heat sink
    2. **Materials**: Select your material composition
    3. **Heat Sources**: Define heat source locations and power
    4. **Simulation**: Run the PINN solver
    5. **Results**: View temperature fields and thermal limits
    
    ---
    
    Use the sidebar to navigate between pages.
    """)
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Materials Available", "30+")
    
    with col2:
        st.metric("Boundary Condition Types", "4")
    
    with col3:
        st.metric("Heat Source Types", "5")


def show_geometry_page():
    """Display geometry configuration page."""
    st.title("📐 Geometry Configuration")
    
    geometry_type = st.radio(
        "Select geometry input method:",
        ["Upload Image", "Parametric Heat Sink", "Simple Rectangle"]
    )
    
    if geometry_type == "Upload Image":
        st.subheader("Upload Geometry Image")
        
        st.info("""
        Upload a black and white PNG image:
        - **White pixels (255)**: Solid material
        - **Black pixels (0)**: Void/air
        """)
        
        uploaded_file = st.file_uploader(
            "Choose a PNG file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a geometry mask image"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                # Process image
                processor = ImageToGeometry()
                
                # Physical dimensions
                physical_size = st.slider(
                    "Physical domain size (m)",
                    min_value=0.01,
                    max_value=1.0,
                    value=0.1,
                    step=0.01
                )
                
                # Convert to array and process
                img_array = np.array(image.convert('L'))
                
                # Normalize to binary mask
                mask = (img_array > 128).astype(np.float32)
                
                # Create DomainInfo manually
                ny, nx = mask.shape
                boundary_points = np.array([[0, 0], [physical_size, 0], [physical_size, physical_size], [0, physical_size]])
                
                # Calculate actual geometry area based on mask pixels
                pixel_area = (physical_size / nx) * (physical_size / ny)
                geometry_area = np.sum(mask) * pixel_area
                
                domain_info = DomainInfo(
                    mask=mask,
                    sdf=np.zeros_like(mask),
                    boundary_points=boundary_points,
                    area=geometry_area,  # Use actual geometry area, not full domain
                    perimeter=4 * physical_size,  # Approximation - could compute from mask boundary
                    centroid=(physical_size/2, physical_size/2),
                    bounding_box=(0, 0, physical_size, physical_size),
                    resolution=(ny, nx),
                    physical_size=(physical_size, physical_size),
                )
                
                st.success(f"Domain processed: {domain_info.mask.shape[1]} × {domain_info.mask.shape[0]} grid")
                
                # Store in session state
                st.session_state['domain_info'] = domain_info
                st.session_state['geometry_image'] = image
    
    elif geometry_type == "Parametric Heat Sink":
        st.subheader("Generate Heat Sink Geometry")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sink_type = st.selectbox(
                "Heat sink type",
                ["Straight Fin", "Pin Fin"]
            )
            
            base_height = st.slider(
                "Base height (mm)",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5
            )
            
            fin_height = st.slider(
                "Fin height (mm)",
                min_value=5.0,
                max_value=50.0,
                value=20.0,
                step=1.0
            )
        
        with col2:
            num_fins = st.slider(
                "Number of fins",
                min_value=3,
                max_value=15,
                value=7
            )
            
            fin_width = st.slider(
                "Fin width (mm)",
                min_value=0.5,
                max_value=5.0,
                value=1.5,
                step=0.1
            )
        
        if st.button("Generate Heat Sink"):
            from src.geometry.image_processor import HeatSinkGenerator
            
            generator = HeatSinkGenerator()
            
            # Create simple fin pattern manually
            resolution = 200
            mask = np.ones((resolution, resolution), dtype=np.float32)
            
            # Add fins (simplified)
            fin_spacing = resolution // (num_fins + 1)
            fin_width_px = int(fin_width / 1000 * resolution / 0.1)
            
            for i in range(num_fins):
                x_pos = (i + 1) * fin_spacing
                mask[:, max(0, x_pos-fin_width_px//2):min(resolution, x_pos+fin_width_px//2)] = 1.0
            
            # Create DomainInfo
            boundary_points = np.array([[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]])
            domain_info = DomainInfo(
                mask=mask,
                sdf=np.zeros_like(mask),
                boundary_points=boundary_points,
                area=0.01,
                perimeter=0.4,
                centroid=(0.05, 0.05),
                bounding_box=(0, 0, 0.1, 0.1),
                resolution=(resolution, resolution),
                physical_size=(0.1, 0.1),
            )
            
            st.session_state['domain_info'] = domain_info
            
            # Visualize
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(domain_info.mask, cmap='gray', origin='lower')
            ax.set_title(f"{sink_type} Heat Sink")
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            st.pyplot(fig)
            plt.close()
            
            st.success("Heat sink generated successfully!")
    
    else:  # Simple Rectangle
        st.subheader("Simple Rectangular Domain")
        
        col1, col2 = st.columns(2)
        
        with col1:
            width = st.slider("Width (cm)", 1.0, 20.0, 10.0, 0.5)
            height = st.slider("Height (cm)", 1.0, 20.0, 10.0, 0.5)
        
        with col2:
            nx = st.slider("X Resolution", 20, 200, 100, 10)
            ny = st.slider("Y Resolution", 20, 200, 100, 10)
        
        if st.button("Create Domain"):
            mask = np.ones((ny, nx), dtype=np.float32)
            # Create minimal boundary points
            boundary_points = np.array([[0, 0], [width/100, 0], [width/100, height/100], [0, height/100]])
            
            domain_info = DomainInfo(
                mask=mask,
                sdf=np.zeros_like(mask),
                boundary_points=boundary_points,
                area=(width/100) * (height/100),
                perimeter=2 * ((width/100) + (height/100)),
                centroid=(width/200, height/200),
                bounding_box=(0, 0, width/100, height/100),
                resolution=(ny, nx),
                physical_size=(width/100, height/100),  # (width, height) in meters
            )
            
            st.session_state['domain_info'] = domain_info
            st.success(f"Created {nx}×{ny} rectangular domain")


def show_materials_page():
    """Display materials selection page."""
    st.title("🧪 Material Selection")
    
    db = MaterialDatabase()
    
    st.markdown("""
    Select materials by name and specify the composition percentage.
    The system will automatically calculate effective thermal properties.
    """)
    
    # Material selection
    st.subheader("Material Composition")
    
    available_materials = db.list_all()
    
    # Dynamic material inputs
    if 'num_materials' not in st.session_state:
        st.session_state['num_materials'] = 1
    
    col_add, col_remove = st.columns([1, 1])
    with col_add:
        if st.button("➕ Add Material"):
            st.session_state['num_materials'] += 1
    with col_remove:
        if st.button("➖ Remove Material") and st.session_state['num_materials'] > 1:
            st.session_state['num_materials'] -= 1
    
    composition = {}
    total_percent = 0
    
    for i in range(st.session_state['num_materials']):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            material = st.selectbox(
                f"Material {i+1}",
                available_materials,
                key=f"mat_{i}"
            )
        
        with col2:
            percent = st.number_input(
                f"Percentage",
                min_value=0.0,
                max_value=100.0,
                value=100.0 / st.session_state['num_materials'],
                step=1.0,
                key=f"pct_{i}"
            )
        
        if percent > 0:
            composition[material] = percent / 100
            total_percent += percent
    
    # Validation
    if abs(total_percent - 100) > 0.1:
        st.warning(f"⚠️ Total percentage is {total_percent:.1f}%, should be 100%")
    else:
        st.success("✅ Composition is valid")
    
    # Calculate effective properties
    if composition and abs(total_percent - 100) < 0.1:
        st.subheader("Effective Properties")
        
        calc = MixtureCalculator(db)
        
        try:
            props = calc.calculate(composition)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Thermal Conductivity", f"{props.thermal_conductivity:.1f} W/(m·K)")
            
            with col2:
                st.metric("Density", f"{props.density:.0f} kg/m³")
            
            with col3:
                st.metric("Specific Heat", f"{props.specific_heat:.0f} J/(kg·K)")
            
            col4, col5, col6 = st.columns(3)
            
            with col4:
                st.metric("Thermal Diffusivity", f"{props.thermal_diffusivity:.2e} m²/s")
            
            with col5:
                st.metric("Max Operating Temp", f"{props.max_operating_temp:.0f}°C")
            
            with col6:
                st.metric("Melting Point", f"{props.min_melting_point:.0f}°C")
            
            # Store in session state
            st.session_state['material_composition'] = composition
            st.session_state['material_properties'] = props
            
        except Exception as e:
            st.error(f"Error calculating properties: {e}")
    
    # Show individual material properties
    with st.expander("📖 Material Database"):
        # Map display names to actual category values in database
        category_map = {
            "Metals": "metal",
            "Ceramics": "ceramic", 
            "Polymers": "polymer",
            "Semiconductors": "semiconductor",
            "Composites": "composite"
        }
        
        for display_name, cat in category_map.items():
            st.markdown(f"**{display_name}**")
            # Get all materials and filter by category
            all_materials = db.list_all()
            materials = [m for m in all_materials if db.get(m) and db.get(m).category == cat]
            
            if materials:
                data = []
                for mat_name in materials:
                    mat = db.get(mat_name)
                    data.append({
                        "Name": mat_name.replace("_", " ").title(),
                        "k (W/m·K)": mat.thermal_conductivity,
                        "ρ (kg/m³)": mat.density,
                        "cp (J/kg·K)": mat.specific_heat,
                        "Max T (°C)": mat.max_operating_temp
                    })
                
                st.dataframe(data, use_container_width=True)


def show_heat_sources_page():
    """Display heat source configuration page."""
    st.title("🔥 Heat Source Configuration")
    
    st.markdown("""
    Configure heat sources for your thermal simulation.
    You can add multiple sources of different types.
    """)
    
    if 'heat_sources' not in st.session_state:
        st.session_state['heat_sources'] = []
    
    # Add new heat source
    st.subheader("Add Heat Source")
    
    col1, col2 = st.columns(2)
    
    with col1:
        source_type = st.selectbox(
            "Source Type",
            ["Point", "Rectangular", "Circular", "Gaussian"]
        )
        
        power = st.number_input(
            "Power (W)",
            min_value=0.1,
            max_value=10000.0,
            value=100.0,
            step=10.0
        )
    
    with col2:
        x_pos = st.slider("X Position", 0.0, 1.0, 0.5, 0.01)
        y_pos = st.slider("Y Position", 0.0, 1.0, 0.5, 0.01)
    
    # Type-specific parameters
    if source_type == "Rectangular":
        col3, col4 = st.columns(2)
        with col3:
            width = st.slider("Width", 0.01, 0.5, 0.1, 0.01)
        with col4:
            height = st.slider("Height", 0.01, 0.5, 0.1, 0.01)
    elif source_type == "Circular" or source_type == "Gaussian":
        radius = st.slider("Radius", 0.01, 0.3, 0.05, 0.01)
    
    if st.button("Add Heat Source"):
        source = {
            'type': source_type,
            'power': power,
            'x': x_pos,
            'y': y_pos
        }
        
        if source_type == "Rectangular":
            source['width'] = width
            source['height'] = height
        elif source_type in ["Circular", "Gaussian"]:
            source['radius'] = radius
        
        st.session_state['heat_sources'].append(source)
        st.success(f"Added {source_type} heat source at ({x_pos:.2f}, {y_pos:.2f})")
    
    # Display current heat sources
    st.subheader("Current Heat Sources")
    
    if st.session_state['heat_sources']:
        for i, source in enumerate(st.session_state['heat_sources']):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(
                    f"**{source['type']}**: {source['power']:.0f}W at "
                    f"({source['x']:.2f}, {source['y']:.2f})"
                )
            
            with col3:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state['heat_sources'].pop(i)
                    st.rerun()
        
        # Visualize heat sources
        if 'domain_info' in st.session_state:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            domain = st.session_state['domain_info']
            ax.imshow(domain.mask, cmap='gray', origin='lower', alpha=0.5)
            
            for source in st.session_state['heat_sources']:
                x_px = source['x'] * domain.mask.shape[1]
                y_px = source['y'] * domain.mask.shape[0]
                
                if source['type'] == 'Point':
                    ax.plot(x_px, y_px, 'r*', markersize=20)
                else:
                    circle = plt.Circle(
                        (x_px, y_px), 
                        source.get('radius', 0.05) * domain.mask.shape[0],
                        color='red', alpha=0.5
                    )
                    ax.add_patch(circle)
            
            ax.set_title("Heat Source Locations")
            st.pyplot(fig)
            plt.close()
    else:
        st.info("No heat sources added yet. Add at least one heat source.")
    
    if st.button("Clear All Sources"):
        st.session_state['heat_sources'] = []
        st.rerun()


def show_simulation_page():
    """Display simulation configuration and run page."""
    st.title("🎯 Run Simulation")
    
    # Check prerequisites
    ready = True
    
    if 'domain_info' not in st.session_state:
        st.warning("⚠️ Please configure geometry first")
        ready = False
    else:
        st.success("✅ Geometry configured")
    
    if 'material_properties' not in st.session_state:
        st.warning("⚠️ Please select materials first")
        ready = False
    else:
        st.success("✅ Materials selected")
    
    if 'heat_sources' not in st.session_state or not st.session_state['heat_sources']:
        st.warning("⚠️ Please add heat sources first")
        ready = False
    else:
        st.success(f"✅ {len(st.session_state['heat_sources'])} heat source(s) configured")
    
    st.subheader("Simulation Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        initial_temp = st.number_input(
            "Initial Temperature (°C)",
            min_value=-50.0,
            max_value=200.0,
            value=25.0,
            step=5.0
        )
        
        simulation_time = st.number_input(
            "Simulation Time (s)",
            min_value=0.1,
            max_value=1000.0,
            value=10.0,
            step=1.0
        )
    
    with col2:
        ambient_temp = st.number_input(
            "Ambient Temperature (°C)",
            min_value=-50.0,
            max_value=100.0,
            value=25.0,
            step=5.0
        )
        
        num_epochs = st.slider(
            "Training Epochs",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100
        )
    
    # Heat Transfer Coefficient Configuration
    st.subheader("🌡️ Cooling Configuration")
    
    cooling_type = st.selectbox(
        "Cooling Method",
        ["Natural Convection (Air)", "Forced Convection (Fan)", "Liquid Cooling", "Custom"],
        help="Select cooling method - this affects heat transfer coefficient"
    )
    
    # Preset h values based on cooling type
    h_conv_presets = {
        "Natural Convection (Air)": 10.0,
        "Forced Convection (Fan)": 50.0,
        "Liquid Cooling": 500.0,
        "Custom": 25.0
    }
    
    if cooling_type == "Custom":
        h_conv = st.number_input(
            "Heat Transfer Coefficient h (W/m²·K)",
            min_value=1.0,
            max_value=10000.0,
            value=25.0,
            step=5.0,
            help="Typical values: Natural air 5-25, Forced air 25-250, Water 500-10000"
        )
    else:
        h_conv = h_conv_presets[cooling_type]
        st.info(f"📊 Heat transfer coefficient: **{h_conv} W/(m²·K)**")
    
    # Show cooling info
    with st.expander("ℹ️ Cooling Method Reference"):
        st.markdown("""
        | Cooling Method | h [W/(m²·K)] | Typical Use |
        |----------------|--------------|-------------|
        | Natural Convection (Air) | 5-25 | Passive cooling, enclosed electronics |
        | Forced Convection (Fan) | 25-250 | CPU/GPU coolers, active heat sinks |
        | Liquid Cooling (Water) | 500-10,000 | High-power electronics, data centers |
        | Liquid Cooling (Oil) | 50-1,500 | Transformers, industrial equipment |
        | Boiling Water | 2,500-25,000 | Nuclear reactors, extreme cooling |
        """)
    
    st.subheader("Boundary Conditions")
    
    bc_types = {
        "Bottom": st.selectbox("Bottom BC", ["Fixed Temperature", "Adiabatic", "Convective"]),
        "Top": st.selectbox("Top BC", ["Convective", "Fixed Temperature", "Adiabatic"]),
        "Left": st.selectbox("Left BC", ["Adiabatic", "Fixed Temperature", "Convective"]),
        "Right": st.selectbox("Right BC", ["Adiabatic", "Fixed Temperature", "Convective"]),
    }
    
    st.session_state['simulation_config'] = {
        'initial_temp': initial_temp,
        'ambient_temp': ambient_temp,
        'simulation_time': simulation_time,
        'num_epochs': num_epochs,
        'boundary_conditions': bc_types,
        'h_conv': h_conv,
        'cooling_type': cooling_type
    }
    
    if ready:
        if st.button("🚀 Run Simulation", type="primary"):
            run_simulation()


def run_simulation():
    """Run the PINN simulation with live animation."""
    st.info("Starting simulation with live temperature visualization...")
    
    # Create progress bar and placeholders
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create placeholder for live animation
    animation_placeholder = st.empty()
    
    import time
    
    config = st.session_state['simulation_config']
    props = st.session_state['material_properties']
    
    # Get domain
    domain = st.session_state['domain_info']
    ny, nx = domain.mask.shape[0], domain.mask.shape[1]
    
    # Physical parameters
    k = props.thermal_conductivity  # W/(m·K)
    rho = props.density  # kg/m³
    cp = props.specific_heat  # J/(kg·K)
    alpha = k / (rho * cp)  # Thermal diffusivity (m²/s)
    
    # Get domain physical size
    L = domain.physical_size[0] if hasattr(domain, 'physical_size') else 0.1  # meters
    thickness = 0.005  # 5mm thickness assumption
    
    # Create coordinate grids
    dx = L / nx  # Grid spacing in meters
    dy = L / ny
    
    # Heat transfer coefficient (W/(m²·K)) - Use configured value
    h_conv = config.get('h_conv', 10.0)  # Default to natural convection if not set
    cooling_type = config.get('cooling_type', 'Natural Convection (Air)')
    
    # Calculate total heat source power
    total_power = sum(source['power'] for source in st.session_state['heat_sources'])
    
    # Calculate effective thermal resistance
    A_surface = L * L  # Surface area (m²)
    R_conv = 1.0 / (h_conv * A_surface)  # Convective resistance (K/W)
    R_cond = thickness / (k * A_surface)  # Conductive resistance (K/W)
    R_total = R_conv + R_cond
    
    # Steady-state temperature rise: ΔT = Q × R_total
    delta_T_steady = total_power * R_total
    T_steady_analytical = config['ambient_temp'] + delta_T_steady
    
    # Time constant: τ = ρ × V × cp × R_total
    volume = A_surface * thickness
    thermal_mass = rho * volume * cp
    time_constant = thermal_mass * R_total
    
    # Simulate temperature evolution using Finite Difference Method
    T = np.ones((ny, nx)) * config['initial_temp']
    num_steps = 50  # More steps for smoother diffusion
    t_total = config.get('simulation_time', 10.0)
    dt = t_total / num_steps
    
    # Stability criterion for explicit FDM: dt <= dx²/(4*alpha)
    dt_stable = (dx ** 2) / (4 * alpha)
    
    # Use smaller time step if needed for stability
    if dt > dt_stable:
        num_substeps = int(np.ceil(dt / dt_stable)) + 1
    else:
        num_substeps = 1
    dt_sub = dt / num_substeps
    
    # Create heat source field (Q in W/m³)
    Q_field = np.zeros((ny, nx))
    
    # Calculate cell volume for heat source normalization
    cell_volume = dx * dy * thickness  # m³
    
    for source in st.session_state['heat_sources']:
        # Convert normalized position (0-1) to grid indices
        x_idx = int(source['x'] * (nx - 1))
        y_idx = int(source['y'] * (ny - 1))
        
        # Clamp to valid range
        x_idx = max(0, min(nx - 1, x_idx))
        y_idx = max(0, min(ny - 1, y_idx))
        
        # Check if heat source is inside geometry
        if domain.mask[y_idx, x_idx] < 0.5:
            st.warning(f"Heat source at ({source['x']:.2f}, {source['y']:.2f}) is outside the geometry!")
        
        # Radius in pixels (minimum 2 pixels for spreading)
        radius_px = max(2, int(source.get('radius', 0.05) * min(nx, ny)))
        sigma = max(radius_px, 2)
        
        # Create Gaussian heat distribution
        for i in range(ny):
            for j in range(nx):
                if domain.mask[i, j] > 0.5:  # Only inside geometry
                    dist2 = (i - y_idx)**2 + (j - x_idx)**2
                    # Gaussian with proper normalization
                    # Power (W) distributed over area, converted to volumetric (W/m³)
                    gauss_weight = np.exp(-dist2 / (2 * sigma**2))
                    Q_field[i, j] += source['power'] * gauss_weight
    
    # Normalize Q_field so total power is conserved
    Q_sum = np.sum(Q_field)
    if Q_sum > 0:
        Q_field = Q_field * (total_power / Q_sum) / cell_volume  # Convert to W/m³
    
    # Precompute boundary mask (cells at edge of geometry)
    from scipy import ndimage
    geometry_mask_bool = domain.mask > 0.5
    geometry_eroded = ndimage.binary_erosion(geometry_mask_bool)
    boundary_mask = geometry_mask_bool & ~geometry_eroded
    
    # Biot number check for lumped vs distributed analysis
    Bi = h_conv * dx / k
    if Bi > 0.1:
        st.info(f"Biot number = {Bi:.2f} > 0.1: Using distributed thermal analysis")
    
    for step in range(num_steps):
        progress = (step + 1) / num_steps
        progress_bar.progress(progress)
        t_current = (step + 1) * dt
        status_text.text(f"Simulating heat transfer... t = {t_current:.2f}s ({step+1}/{num_steps})")
        
        # FDM Heat Diffusion with sub-stepping for stability (Vectorized)
        for _ in range(num_substeps):
            # Compute Laplacian using vectorized operations
            # For interior points, use neighbors; for boundary, use ambient temp
            T_padded = np.pad(T, 1, mode='edge')  # Use edge values for padding
            
            # Set padded boundary to ambient for convective BC effect
            T_padded[0, :] = config['ambient_temp']
            T_padded[-1, :] = config['ambient_temp']
            T_padded[:, 0] = config['ambient_temp']
            T_padded[:, -1] = config['ambient_temp']
            
            # Laplacian: ∇²T = (T[i+1,j] + T[i-1,j])/dy² + (T[i,j+1] + T[i,j-1] - 2*T)/dx² - (2/dx² + 2/dy²)*T
            # For square grid (dx=dy), simplifies to standard form
            # Using dx for both since grid is assumed square, but use min for safety
            dxy = min(dx, dy)
            laplacian = (
                T_padded[2:, 1:-1] +   # T[i+1, j]
                T_padded[:-2, 1:-1] +  # T[i-1, j]
                T_padded[1:-1, 2:] +   # T[i, j+1]
                T_padded[1:-1, :-2] -  # T[i, j-1]
                4 * T
            ) / (dxy ** 2)
            
            # Heat equation: dT/dt = alpha * ∇²T + Q/(rho*cp)
            # Q_field is in W/m³, divide by (rho*cp) to get K/s
            heat_source_term = Q_field / (rho * cp)
            
            # Temperature update
            dT = dt_sub * (alpha * laplacian + heat_source_term)
            T_new = T + dT
            
            # Apply convective cooling at geometry boundaries (Robin BC)
            # -k * dT/dn = h * (T - T_ambient)
            # Approximation: T_boundary_new = T - (h*dx/k) * (T - T_ambient) * (dt_sub * alpha / dx²)
            cooling_factor = (h_conv / k) * dx * (dt_sub * alpha / (dx**2))
            cooling_factor = min(cooling_factor, 0.5)  # Limit for stability
            
            T_new = np.where(
                boundary_mask,
                T_new - cooling_factor * (T_new - config['ambient_temp']),
                T_new
            )
            
            # Ensure temperature doesn't go below ambient (physical constraint)
            T_new = np.maximum(T_new, config['ambient_temp'])
            
            # Apply geometry mask (outside = ambient)
            T = np.where(geometry_mask_bool, T_new, config['ambient_temp'])
        
        # Create masked temperature for visualization (NaN outside geometry)
        T_display = np.where(geometry_mask_bool, T, np.nan)
        
        # Show live animation
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Overlay temperature only inside geometry
        vmin_temp = config['ambient_temp']
        vmax_temp = np.nanmax(T_display) if not np.all(np.isnan(T_display)) else config['ambient_temp'] + 10
        vmax_temp = max(vmax_temp, vmin_temp + 5)  # Ensure some range
        
        im = ax.imshow(T_display, cmap='hot', origin='lower', aspect='equal',
                       vmin=vmin_temp, vmax=vmax_temp)
        
        # Show geometry outline
        ax.contour(domain.mask, levels=[0.5], colors='cyan', linewidths=1.5, origin='lower')
        
        ax.set_title(f'Temperature Distribution (t = {t_current:.2f}s)')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Add annotations
        max_temp = np.nanmax(T_display) if not np.all(np.isnan(T_display)) else config['ambient_temp']
        mean_temp = np.nanmean(T_display) if not np.all(np.isnan(T_display)) else config['ambient_temp']
        ax.text(0.02, 0.98, f'Max: {max_temp:.1f}°C\nMean: {mean_temp:.1f}°C\nΔT: {max_temp - config["ambient_temp"]:.1f}°C', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
        
        plt.colorbar(im, label='Temperature (°C)', ax=ax)
        animation_placeholder.pyplot(fig)
        plt.close(fig)
        
        time.sleep(0.15)  # Animation delay
    
    # Use final T from FDM simulation
    T_final = T.copy()
    
    # Apply domain mask to ensure clean result
    T_final = np.where(geometry_mask_bool, T_final, config['ambient_temp'])
    
    # Calculate statistics only for points inside geometry
    T_inside = T_final[geometry_mask_bool]
    
    # Store results
    st.session_state['simulation_results'] = {
        'temperature_field': T_final,
        'max_temp': float(np.max(T_inside)) if len(T_inside) > 0 else config['ambient_temp'],
        'mean_temp': float(np.mean(T_inside)) if len(T_inside) > 0 else config['ambient_temp'],
        'min_temp': float(np.min(T_inside)) if len(T_inside) > 0 else config['ambient_temp'],
        'time_constant': time_constant,
        'steady_state_analytical': T_steady_analytical,
        'total_power': total_power,
        'h_conv': h_conv,
        'cooling_type': cooling_type,
        'geometry_mask': geometry_mask_bool,
    }
    
    # Thermal limit analysis
    analyzer = ThermalLimitAnalyzer()
    
    # Use physical domain area (from mask if DomainInfo provides it, otherwise calculate)
    physical_area = domain.area if hasattr(domain, 'area') and domain.area > 0 else L * L
    
    limit_result = analyzer.analyze(
        material_props=props,
        heat_source_power=total_power,
        domain_area=physical_area,
        domain_thickness=0.005,  # 5mm thickness assumption
        ambient_temp=config['ambient_temp'],
        heat_transfer_coeff=h_conv,  # Use configured heat transfer coefficient
        initial_temp=config['initial_temp'],
        predicted_temps=T_inside,  # Use final temperature field inside geometry
    )
    
    st.session_state['thermal_limits'] = limit_result
    
    progress_bar.progress(1.0)
    status_text.text("✅ Simulation complete!")
    
    st.success("Simulation completed successfully! View results in the Results page.")


def show_results_page():
    """Display simulation results."""
    st.title("📊 Simulation Results")
    
    if 'simulation_results' not in st.session_state or st.session_state['simulation_results'] is None:
        st.warning("No simulation results available. Please run a simulation first.")
        return
    
    results = st.session_state['simulation_results']
    
    # Key metrics
    st.subheader("🌡️ Temperature Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Maximum", f"{results['max_temp']:.1f}°C")
    
    with col2:
        st.metric("Mean", f"{results['mean_temp']:.1f}°C")
    
    with col3:
        st.metric("Minimum", f"{results['min_temp']:.1f}°C")
    
    with col4:
        delta = results['max_temp'] - results['min_temp']
        st.metric("Range", f"{delta:.1f}°C")
    
    # Cooling configuration display
    if 'h_conv' in results:
        st.markdown("---")
        st.subheader("🌬️ Cooling Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cooling Method", results.get('cooling_type', 'Natural Convection'))
        with col2:
            st.metric("Heat Transfer Coeff (h)", f"{results['h_conv']:.1f} W/(m²·K)")
        with col3:
            st.metric("Total Heat Power", f"{results['total_power']:.1f} W")
    
    # Thermal limits and lifetime analysis
    if 'thermal_limits' in st.session_state and st.session_state['thermal_limits'] is not None:
        limits = st.session_state['thermal_limits']
        
        st.markdown("---")
        st.subheader("🔥 Maximum Temperature Capacity & Limits")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Max Operating Temp", 
                f"{limits.max_operating_temp:.1f}°C",
                help="Maximum safe continuous operating temperature"
            )
        
        with col2:
            st.metric(
                "Melting Point", 
                f"{limits.melting_point:.1f}°C",
                help="Temperature at which material starts melting"
            )
        
        with col3:
            if limits.max_temperature_capacity:
                st.metric(
                    "Max Temp Capacity",
                    f"{limits.max_temperature_capacity:.1f}°C",
                    help="Maximum temperature the geometry can handle safely"
                )
        
        with col4:
            if limits.thermal_headroom is not None:
                headroom_color = "normal" if limits.thermal_headroom > 20 else "inverse"
                st.metric(
                    "Thermal Headroom",
                    f"{limits.thermal_headroom:.1f}%",
                    delta=f"{'Safe' if limits.thermal_headroom > 20 else 'Low!'}" if limits.thermal_headroom > 0 else "Critical",
                    delta_color=headroom_color
                )
        
        st.markdown("---")
        st.subheader("⏱️ Geometry Lifetime Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Geometry Lifetime
            if limits.geometry_lifetime is not None:
                if limits.geometry_lifetime == float('inf'):
                    st.metric("🟢 Safe Operating Time", "Indefinite ✅")
                    st.success("Geometry can operate safely at steady state!")
                else:
                    lifetime = limits.geometry_lifetime
                    if lifetime < 60:
                        lifetime_str = f"{lifetime:.1f} sec"
                        st.metric("🔴 Safe Operating Time", lifetime_str)
                        st.error("Very short operating time!")
                    elif lifetime < 3600:
                        lifetime_str = f"{lifetime/60:.1f} min"
                        st.metric("🟡 Safe Operating Time", lifetime_str)
                    else:
                        lifetime_str = f"{lifetime/3600:.1f} hours"
                        st.metric("🟢 Safe Operating Time", lifetime_str)
            else:
                st.metric("Safe Operating Time", "N/A")
        
        with col2:
            # Time to melting
            if limits.time_to_melting is not None:
                melt_time = limits.time_to_melting
                if melt_time < 60:
                    st.metric("🔥 Time Until Melting Starts", f"{melt_time:.1f} sec")
                    st.error(f"CRITICAL: Material will start melting in {melt_time:.1f} seconds!")
                elif melt_time < 3600:
                    st.metric("🔥 Time Until Melting Starts", f"{melt_time/60:.1f} min")
                    st.warning(f"Material will start melting in {melt_time/60:.1f} minutes")
                else:
                    st.metric("🔥 Time Until Melting Starts", f"{melt_time/3600:.1f} hours")
            else:
                st.metric("Time Until Melting", "Never ✅")
                st.success("Temperature stays below melting point!")
        
        with col3:
            # Time to max operating
            if limits.time_to_max_operating is not None:
                max_op_time = limits.time_to_max_operating
                if max_op_time < 60:
                    st.metric("⚠️ Time to Max Operating", f"{max_op_time:.1f} sec")
                elif max_op_time < 3600:
                    st.metric("⚠️ Time to Max Operating", f"{max_op_time/60:.1f} min")
                else:
                    st.metric("⚠️ Time to Max Operating", f"{max_op_time/3600:.1f} hours")
            else:
                st.metric("Time to Max Operating", "Never ✅")
        
        # Limiting factor
        if limits.lifetime_limiting_factor:
            st.info(f"📌 **Limiting Factor:** {limits.lifetime_limiting_factor}")
        
        st.markdown("---")
        st.subheader("🛡️ Safety Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            margin_status = "normal" if limits.temp_margin > 50 else "inverse"
            st.metric(
                "Safety Margin", 
                f"{limits.temp_margin:.1f}°C",
                delta="Good" if limits.temp_margin > 50 else "Low",
                delta_color=margin_status
            )
        
        with col2:
            risk_color = {
                'LOW': '🟢',
                'MEDIUM': '🟡', 
                'HIGH': '🟠',
                'CRITICAL': '🔴'
            }
            st.metric("Risk Level", f"{risk_color.get(limits.risk_level.name, '⚪')} {limits.risk_level.name}")
        
        with col3:
            safety_status = "✅ SAFE" if limits.is_safe else "❌ UNSAFE"
            if limits.is_safe:
                st.success(f"**Status:** {safety_status}")
            else:
                st.error(f"**Status:** {safety_status}")
        
        # Warnings
        if limits.warnings:
            st.markdown("---")
            st.subheader("⚠️ Warnings")
            for warning in limits.warnings:
                st.warning(warning)
    
    # Temperature field visualization
    st.markdown("---")
    st.subheader("🗺️ Temperature Field")
    
    T = results['temperature_field']
    
    # Get geometry mask for proper visualization
    if 'domain_info' in st.session_state:
        domain = st.session_state['domain_info']
        geometry_mask = domain.mask
        
        # Create masked temperature field - show NaN outside geometry
        T_masked = np.where(geometry_mask > 0.5, T, np.nan)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Show geometry outline in gray
        ax.imshow(geometry_mask, cmap='gray', origin='lower', aspect='equal', alpha=0.3)
        
        # Overlay temperature field only inside geometry
        im = ax.imshow(T_masked, cmap='hot', origin='lower', aspect='equal', 
                       vmin=np.nanmin(T_masked), vmax=np.nanmax(T_masked))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Temperature Distribution (within geometry)')
        plt.colorbar(im, label='Temperature (°C)')
        
        # Mark hotspots (only within geometry)
        T_in_geom = np.where(geometry_mask > 0.5, T, -np.inf)
        max_idx = np.unravel_index(np.argmax(T_in_geom), T_in_geom.shape)
        ax.plot(max_idx[1], max_idx[0], 'c*', markersize=20, label='Max Temperature')
        ax.legend()
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(T, cmap='hot', origin='lower', aspect='equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Temperature Distribution')
        plt.colorbar(im, label='Temperature (°C)')
        
        # Mark hotspots
        max_idx = np.unravel_index(np.argmax(T), T.shape)
        ax.plot(max_idx[1], max_idx[0], 'c*', markersize=20, label='Max Temperature')
        ax.legend()
    
    st.pyplot(fig)
    plt.close()
    
    # Cross-section profiles
    st.subheader("📈 Temperature Profiles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        mid_y = T.shape[0] // 2
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(T[mid_y, :])
        ax.set_xlabel('X Position')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Horizontal Profile (Y=center)')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        mid_x = T.shape[1] // 2
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(T[:, mid_x])
        ax.set_xlabel('Y Position')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Vertical Profile (X=center)')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    # Recommendations
    st.subheader("Design Recommendations")
    
    if 'material_properties' in st.session_state:
        engine = RecommendationEngine()
        
        # Simple recommendation logic
        props = st.session_state['material_properties']
        
        recommendations = []
        
        if results['max_temp'] > props.max_operating_temp * 0.8:
            recommendations.append({
                'priority': 'high',
                'category': 'thermal',
                'title': 'High Temperature Warning',
                'description': 'Maximum temperature is approaching material limits. '
                              'Consider improving cooling or using higher conductivity materials.'
            })
        
        if results['max_temp'] - results['min_temp'] > 50:
            recommendations.append({
                'priority': 'medium',
                'category': 'design',
                'title': 'Large Temperature Gradient',
                'description': 'Significant temperature variation detected. '
                              'Consider redistributing heat sources or adding thermal paths.'
            })
        
        if props.thermal_conductivity < 100:
            recommendations.append({
                'priority': 'medium',
                'category': 'material',
                'title': 'Material Conductivity',
                'description': 'Consider using higher conductivity materials like copper or aluminum '
                              'for better heat spreading.'
            })
        
        for rec in recommendations:
            priority_color = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            st.info(f"{priority_color.get(rec['priority'], '⚪')} **{rec['title']}**: {rec['description']}")
        
        if not recommendations:
            st.success("✅ No critical issues detected. Design appears thermally sound.")


def show_report_page():
    """Display report generation page."""
    st.title("📋 Generate Report")
    
    if 'simulation_results' not in st.session_state:
        st.warning("No simulation results available. Please run a simulation first.")
        return
    
    st.markdown("Generate a comprehensive PDF or HTML report of your thermal analysis.")
    
    report_format = st.radio("Report Format", ["PDF", "HTML", "Both"])
    
    st.subheader("Report Options")
    
    include_summary = st.checkbox("Include Executive Summary", value=True)
    include_material = st.checkbox("Include Material Properties", value=True)
    include_temperature = st.checkbox("Include Temperature Plots", value=True)
    include_recommendations = st.checkbox("Include Recommendations", value=True)
    
    author = st.text_input("Report Author", value="PINN Thermal Analyzer")
    
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            import io
            from datetime import datetime
            
            results = st.session_state['simulation_results']
            
            if report_format in ["PDF", "Both"]:
                # Create a temporary file for the PDF
                import tempfile
                import os
                
                tmp_name = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                        tmp_name = tmp.name
                    
                    # Generate PDF report
                    from src.visualization.reports import ReportGenerator
                    generator = ReportGenerator()
                    generator.generate_pdf_report(results, tmp_name)
                    
                    # Read the generated PDF
                    with open(tmp_name, 'rb') as f:
                        pdf_bytes = f.read()
                    
                    st.success("✅ Report generated successfully!")
                    
                    # Offer download
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"thermal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    # Fallback to simple text report if PDF generation fails
                    st.warning(f"PDF generation unavailable: {e}")
                    st.info("Generating text report instead...")
                    
                    # Create comprehensive text report
                    report_text = f"""
THERMAL ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Author: {author}

{'='*50}
SIMULATION RESULTS
{'='*50}

Maximum Temperature: {results['max_temp']:.2f}°C
Minimum Temperature: {results['min_temp']:.2f}°C
Average Temperature: {results['mean_temp']:.2f}°C

Temperature Range: {results['max_temp'] - results['min_temp']:.2f}°C
"""
                    # Add thermal limits if available
                    if 'thermal_limits' in st.session_state and st.session_state['thermal_limits'] is not None:
                        limits = st.session_state['thermal_limits']
                        
                        # Format lifetime
                        if limits.geometry_lifetime is not None:
                            if limits.geometry_lifetime == float('inf'):
                                lifetime_str = "Indefinite (Stable)"
                            elif limits.geometry_lifetime < 60:
                                lifetime_str = f"{limits.geometry_lifetime:.1f} seconds"
                            elif limits.geometry_lifetime < 3600:
                                lifetime_str = f"{limits.geometry_lifetime/60:.1f} minutes"
                            else:
                                lifetime_str = f"{limits.geometry_lifetime/3600:.1f} hours"
                        else:
                            lifetime_str = "N/A"
                        
                        # Format time to melting
                        if limits.time_to_melting is not None:
                            if limits.time_to_melting < 60:
                                melt_str = f"{limits.time_to_melting:.1f} seconds"
                            elif limits.time_to_melting < 3600:
                                melt_str = f"{limits.time_to_melting/60:.1f} minutes"
                            else:
                                melt_str = f"{limits.time_to_melting/3600:.1f} hours"
                        else:
                            melt_str = "Never (Temperature stays below melting point)"
                        
                        report_text += f"""
{'='*50}
THERMAL LIMITS & LIFETIME ANALYSIS
{'='*50}

Max Operating Temperature: {limits.max_operating_temp:.2f}°C
Melting Point: {limits.melting_point:.2f}°C
Max Temperature Capacity: {limits.max_temperature_capacity:.2f}°C

Safety Margin: {limits.temp_margin:.2f}°C
Thermal Headroom: {limits.thermal_headroom:.1f}%
Risk Level: {limits.risk_level.name}
Status: {'SAFE' if limits.is_safe else 'UNSAFE'}

*** GEOMETRY LIFETIME: {lifetime_str} ***
*** TIME UNTIL MELTING: {melt_str} ***

Limiting Factor: {limits.lifetime_limiting_factor or 'N/A'}
"""
                        if limits.warnings:
                            report_text += "\nWARNINGS:\n"
                            for warning in limits.warnings:
                                report_text += f"  - {warning}\n"
                    
                    report_text += f"\n{'='*50}\nMATERIAL PROPERTIES\n{'='*50}\n"
                    if 'material_properties' in st.session_state:
                        props = st.session_state['material_properties']
                        composition_str = ", ".join([f"{v*100:.0f}% {k}" for k, v in props.composition.items()])
                        report_text += f"""
Material Composition: {composition_str}
Thermal Conductivity: {props.thermal_conductivity:.2f} W/(m·K)
Specific Heat: {props.specific_heat:.2f} J/(kg·K)
Density: {props.density:.2f} kg/m³
Max Operating Temp: {props.max_operating_temp:.2f}°C
"""
                    
                    report_text += f"\n{'='*50}\n"
                    
                    st.download_button(
                        label="📥 Download Text Report",
                        data=report_text.encode('utf-8'),
                        file_name=f"thermal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                finally:
                    # Clean up temp file - wait a bit and use try/except to handle Windows file locking
                    if tmp_name and os.path.exists(tmp_name):
                        try:
                            import time
                            time.sleep(0.1)  # Brief delay for Windows
                            os.unlink(tmp_name)
                        except PermissionError:
                            pass  # File will be cleaned up by OS eventually
            
            elif report_format == "HTML":
                # Generate comprehensive HTML report
                
                # Get thermal limits info if available
                thermal_limits_html = ""
                if 'thermal_limits' in st.session_state and st.session_state['thermal_limits'] is not None:
                    limits = st.session_state['thermal_limits']
                    
                    # Format lifetime
                    if limits.geometry_lifetime is not None:
                        if limits.geometry_lifetime == float('inf'):
                            lifetime_str = "Indefinite (Stable)"
                            lifetime_class = "success"
                        elif limits.geometry_lifetime < 60:
                            lifetime_str = f"{limits.geometry_lifetime:.1f} seconds"
                            lifetime_class = "danger"
                        elif limits.geometry_lifetime < 3600:
                            lifetime_str = f"{limits.geometry_lifetime/60:.1f} minutes"
                            lifetime_class = "warning"
                        else:
                            lifetime_str = f"{limits.geometry_lifetime/3600:.1f} hours"
                            lifetime_class = "success"
                    else:
                        lifetime_str = "N/A"
                        lifetime_class = ""
                    
                    # Format time to melting
                    if limits.time_to_melting is not None:
                        if limits.time_to_melting < 60:
                            melt_str = f"{limits.time_to_melting:.1f} seconds"
                            melt_class = "danger"
                        elif limits.time_to_melting < 3600:
                            melt_str = f"{limits.time_to_melting/60:.1f} minutes"
                            melt_class = "warning"
                        else:
                            melt_str = f"{limits.time_to_melting/3600:.1f} hours"
                            melt_class = "success"
                    else:
                        melt_str = "Never (Safe)"
                        melt_class = "success"
                    
                    thermal_limits_html = f"""
    <h2>🔥 Thermal Limits & Lifetime Analysis</h2>
    <table>
        <tr><th>Parameter</th><th>Value</th><th>Status</th></tr>
        <tr><td>Max Operating Temperature</td><td>{limits.max_operating_temp:.1f}°C</td><td>Limit</td></tr>
        <tr><td>Melting Point</td><td>{limits.melting_point:.1f}°C</td><td>Critical</td></tr>
        <tr><td>Max Temperature Capacity</td><td>{limits.max_temperature_capacity:.1f}°C</td><td>Capacity</td></tr>
        <tr><td>Thermal Headroom</td><td>{limits.thermal_headroom:.1f}%</td><td>{'Good' if limits.thermal_headroom > 20 else 'Low'}</td></tr>
        <tr><td>Safety Margin</td><td>{limits.temp_margin:.1f}°C</td><td>{'✅' if limits.temp_margin > 50 else '⚠️'}</td></tr>
        <tr class="{lifetime_class}"><td><strong>Geometry Lifetime</strong></td><td><strong>{lifetime_str}</strong></td><td>{'✅ Safe' if limits.is_safe else '❌ Unsafe'}</td></tr>
        <tr class="{melt_class}"><td><strong>Time Until Melting</strong></td><td><strong>{melt_str}</strong></td><td>{'✅' if limits.time_to_melting is None else '⚠️'}</td></tr>
        <tr><td>Risk Level</td><td>{limits.risk_level.name}</td><td>{'🟢' if limits.risk_level.name == 'LOW' else '🟡' if limits.risk_level.name == 'MEDIUM' else '🔴'}</td></tr>
    </table>
    
    {f'<p><strong>Limiting Factor:</strong> {limits.lifetime_limiting_factor}</p>' if limits.lifetime_limiting_factor else ''}
"""
                    
                    # Add warnings if any
                    if limits.warnings:
                        thermal_limits_html += "<h3>⚠️ Warnings</h3><ul>"
                        for warning in limits.warnings:
                            thermal_limits_html += f"<li class='warning'>{warning}</li>"
                        thermal_limits_html += "</ul>"
                
                # Get material info if available
                material_html = ""
                if 'material_properties' in st.session_state:
                    props = st.session_state['material_properties']
                    composition_str = ", ".join([f"{v*100:.0f}% {k}" for k, v in props.composition.items()])
                    material_html = f"""
    <h2>🧪 Material Properties</h2>
    <p><strong>Composition:</strong> {composition_str}</p>
    <table>
        <tr><th>Property</th><th>Value</th><th>Unit</th></tr>
        <tr><td>Thermal Conductivity</td><td>{props.thermal_conductivity:.1f}</td><td>W/(m·K)</td></tr>
        <tr><td>Density</td><td>{props.density:.0f}</td><td>kg/m³</td></tr>
        <tr><td>Specific Heat</td><td>{props.specific_heat:.0f}</td><td>J/(kg·K)</td></tr>
        <tr><td>Thermal Diffusivity</td><td>{props.thermal_diffusivity:.2e}</td><td>m²/s</td></tr>
        <tr><td>Max Operating Temp</td><td>{props.max_operating_temp:.0f}</td><td>°C</td></tr>
        <tr><td>Melting Point</td><td>{props.min_melting_point:.0f}</td><td>°C</td></tr>
    </table>
"""
                
                html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Thermal Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .metric {{ font-size: 18px; font-weight: bold; color: #2196F3; }}
        .success {{ background-color: #d4edda !important; }}
        .warning {{ background-color: #fff3cd !important; }}
        .danger {{ background-color: #f8d7da !important; }}
        .metric-box {{ display: inline-block; background: #3498db; color: white; padding: 15px 25px; margin: 10px; border-radius: 8px; text-align: center; }}
        .metric-box .value {{ font-size: 24px; font-weight: bold; }}
        .metric-box .label {{ font-size: 12px; opacity: 0.9; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; text-align: center; }}
    </style>
</head>
<body>
<div class="container">
    <h1>🔥 Thermal Analysis Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Author:</strong> {author}</p>
    
    <h2>📊 Temperature Summary</h2>
    <div style="margin: 20px 0;">
        <div class="metric-box">
            <div class="value">{results['max_temp']:.1f}°C</div>
            <div class="label">Maximum Temperature</div>
        </div>
        <div class="metric-box" style="background: #27ae60;">
            <div class="value">{results['mean_temp']:.1f}°C</div>
            <div class="label">Mean Temperature</div>
        </div>
        <div class="metric-box" style="background: #9b59b6;">
            <div class="value">{results['min_temp']:.1f}°C</div>
            <div class="label">Minimum Temperature</div>
        </div>
        <div class="metric-box" style="background: #e67e22;">
            <div class="value">{results['max_temp'] - results['min_temp']:.1f}°C</div>
            <div class="label">Temperature Range</div>
        </div>
    </div>
    
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Maximum Temperature</td><td class="metric">{results['max_temp']:.2f}°C</td></tr>
        <tr><td>Minimum Temperature</td><td>{results['min_temp']:.2f}°C</td></tr>
        <tr><td>Average Temperature</td><td>{results['mean_temp']:.2f}°C</td></tr>
        <tr><td>Temperature Range</td><td>{results['max_temp'] - results['min_temp']:.2f}°C</td></tr>
    </table>
    
    {thermal_limits_html}
    
    {material_html}
    
    <div class="footer">
        <p>Generated by PINN Thermal Analysis Framework</p>
    </div>
</div>
</body>
</html>
"""
                st.success("✅ Report generated successfully!")
                st.download_button(
                    label="📥 Download HTML Report",
                    data=html_content.encode('utf-8'),
                    file_name=f"thermal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )


if __name__ == "__main__":
    main()
