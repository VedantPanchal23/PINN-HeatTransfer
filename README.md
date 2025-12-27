# 🔥 PINN Thermal Analysis Framework

A Physics-Informed Neural Network (PINN) framework for modeling transient heat transfer in complex 2D domains with automatic thermal limit analysis, geometry lifetime prediction, and design recommendations.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](https://www.docker.com/)

## 🎯 Overview

This project provides a comprehensive thermal analysis solution that:
- **Simulates** transient heat transfer in complex 2D geometries (heat sinks, irregular shapes)
- **Predicts** geometry lifetime and time-to-melting under thermal stress
- **Supports** various boundary conditions (Dirichlet, Neumann, Robin/convective)
- **Analyzes** thermal limits, maximum temperature capacity, and thermal headroom
- **Recommends** design improvements for better thermal performance
- **Validates** results against traditional numerical solvers (FDM)

## ✨ Key Features

### 🧪 Smart Material Input
Specify materials by name and percentage:
```python
# Example material specifications
"aluminum"                      # Single material
"70% aluminum, 30% copper"      # Binary mixture
"60% aluminum, 30% copper, 10% silicon"  # Multi-material
```

The system automatically calculates effective thermal properties using Hashin-Shtrikman bounds.

### 📐 Flexible Geometry Input
- **Image Upload**: PNG/JPG masks (white = solid, black = void)
- **Parametric Heat Sinks**: Auto-generated straight-fin and pin-fin designs
- **Simple Rectangles**: Quick rectangular domain setup

### 🔥 Heat Source Configuration
- Point sources (localized heating)
- Rectangular sources (chip footprints)
- Circular sources
- Gaussian sources (distributed heating)
- Pulsed/time-varying sources

### 📊 Comprehensive Analysis
- **Temperature Fields**: Full 2D transient temperature distribution
- **Hotspot Detection**: Automatic identification of thermal hotspots
- **Thermal Limits**: Safety margin and max operating time calculation
- **Geometry Lifetime**: Estimated safe operating time before thermal damage
- **Melting Analysis**: Time to reach melting point under current conditions
- **Max Temperature Capacity**: Maximum allowable temperature based on material
- **Thermal Headroom**: Percentage of remaining thermal capacity
- **Risk Assessment**: LOW/MEDIUM/HIGH/CRITICAL risk classification
- **Recommendations**: Smart suggestions for thermal improvement

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PINN Thermal Framework                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │  Materials   │   │   Geometry   │   │ Heat Sources │        │
│  │   Database   │   │  Processor   │   │    Config    │        │
│  │  (30+ mats)  │   │ (PNG/param)  │   │  (5 types)   │        │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘        │
│         │                  │                   │                │
│         └──────────────────┼───────────────────┘                │
│                            ▼                                    │
│                  ┌──────────────────┐                          │
│                  │   PINN Solver    │                          │
│                  │ (Fourier + BC)   │                          │
│                  └────────┬─────────┘                          │
│                           │                                     │
│         ┌─────────────────┼─────────────────┐                  │
│         ▼                 ▼                 ▼                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Hotspot   │  │   Thermal    │  │    Design    │          │
│  │  Detection  │  │   Limits     │  │   Recommend  │          │
│  └─────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
PINN/
├── app/                          # Web interface
│   └── streamlit_app.py          # Streamlit UI
├── configs/                      # Configuration files
│   ├── analysis.yaml             # Main analysis config
│   ├── training.yaml             # PINN training config
│   └── dataset.yaml              # Dataset generation config
├── data/                         # Data storage
├── notebooks/                    # Jupyter notebooks
├── scripts/                      # Runnable scripts
│   ├── run_analysis.py           # Main analysis script
│   ├── train.py                  # PINN training
│   └── evaluate.py               # Model evaluation
├── src/
│   ├── materials/                # Material database & mixtures
│   │   ├── database.py           # 30+ material properties
│   │   ├── mixture.py            # Hashin-Shtrikman calculator
│   │   └── thermal_limits.py     # Safety analysis
│   ├── geometry/                 # Geometry processing
│   │   ├── image_processor.py    # PNG to domain conversion
│   │   ├── heat_sources.py       # Heat source configuration
│   │   └── shapes.py             # Parametric shapes
│   ├── pinn/                     # PINN implementation
│   │   ├── network.py            # Neural network architectures
│   │   ├── loss.py               # Physics-informed loss
│   │   ├── boundary_conditions.py # All BC types
│   │   ├── enhanced_solver.py    # Integrated solver
│   │   └── solver.py             # Basic solver
│   ├── analysis/                 # Thermal analysis
│   │   ├── hotspots.py           # Hotspot detection
│   │   ├── performance.py        # Performance metrics
│   │   └── recommendations.py    # Smart recommendations
│   ├── validation/               # Validation tools
│   │   ├── fdm_solver.py         # FDM reference solver
│   │   ├── analytical.py         # Analytical solutions
│   │   └── comparison.py         # PINN vs reference
│   ├── optimization/             # Design optimization
│   │   ├── geometry_optimizer.py # Geometry optimization
│   │   └── material_optimizer.py # Material selection
│   ├── visualization/            # Plotting & reports
│   │   ├── animation.py          # Temperature animations
│   │   ├── reports.py            # PDF/HTML reports
│   │   └── dashboard.py          # Interactive dashboard
│   └── models/                   # Neural operator models
└── tests/                        # Unit tests
```

## 🚀 Installation

### Option 1: Local Installation

```bash
# Clone repository
git clone <repository-url>
cd PINN

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Docker Installation

```bash
# Build the Docker image
docker build -t pinn-thermal:latest .

# Or use docker-compose to build all services
docker-compose build
```

## 🐳 Docker Usage

### Quick Start with Docker Compose

```bash
# Start the Streamlit web interface
docker-compose up streamlit

# Access at http://localhost:8501
```

### Available Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `streamlit` | 8501 | Web interface for thermal analysis |
| `api` | 8000 | REST API server |
| `jupyter` | 8888 | Jupyter Lab for notebooks |
| `train` | - | Model training service |
| `datagen` | - | Dataset generation |
| `evaluate` | - | Model evaluation |

### Running Individual Services

```bash
# Streamlit UI
docker-compose up streamlit

# API Server
docker-compose up api

# Jupyter Lab
docker-compose up jupyter

# Training
docker-compose up train

# Generate dataset
docker-compose up datagen

# Evaluate model
docker-compose up evaluate
```

### Running without GPU

```bash
# Remove 'runtime: nvidia' and 'deploy' sections, then:
docker-compose up streamlit
```

## 📖 Usage

### Option 1: Web Interface (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

Navigate to `http://localhost:8501` in your browser.

### Option 2: Command Line

```bash
# Simple analysis with default settings
python scripts/run_analysis.py --material "aluminum" --power 100

# Custom material mixture
python scripts/run_analysis.py --material "70% aluminum, 30% copper" --power 150

# Using configuration file
python scripts/run_analysis.py --config configs/analysis.yaml
```

### Option 3: Python API

```python
from src.materials import MaterialDatabase, MixtureCalculator
from src.pinn.enhanced_solver import EnhancedPINNSolver, EnhancedPINNConfig
from src.analysis.recommendations import RecommendationEngine

# Setup materials
db = MaterialDatabase()
calc = MixtureCalculator(db)
props = calc.calculate({"aluminum": 0.7, "copper": 0.3})

print(f"Thermal conductivity: {props.thermal_conductivity:.1f} W/(m·K)")
print(f"Max operating temp: {props.max_operating_temp:.0f}°C")

# Run simulation
config = EnhancedPINNConfig(
    hidden_layers=[64, 64, 64],
    num_epochs=2000,
    use_fourier_features=True,
)

solver = EnhancedPINNSolver(config)
result = solver.solve(
    domain_info=domain,
    material_properties=props,
    heat_sources=heat_config,
    boundary_conditions=bc_set,
)

# Get recommendations
engine = RecommendationEngine()
recommendations = engine.analyze(
    temperature_field=result.temperature_field,
    material_properties=props,
    hotspots=result.hotspots,
)

for rec in recommendations:
    print(f"[{rec.priority}] {rec.title}: {rec.description}")
```

## 🧪 Available Materials

| Category | Materials |
|----------|-----------|
| **Metals** | Aluminum, Copper, Silver, Gold, Iron, Steel (Carbon, Stainless 304/316), Titanium, Magnesium, Brass, Bronze |
| **Ceramics** | Aluminum Oxide, Silicon Carbide, Aluminum Nitride, Beryllium Oxide, Boron Nitride |
| **Polymers** | ABS, PLA, PEEK, Nylon, Epoxy, Polycarbonate |
| **Semiconductors** | Silicon, Gallium Arsenide, Silicon Carbide, Germanium |
| **Composites** | Carbon Fiber, Graphite, Glass Fiber |

## 📊 Validation

The framework includes validation against:
- **FDM Solver**: Explicit, implicit (backward Euler), Crank-Nicolson methods
- **Analytical Solutions**: 1D/2D steady-state, transient, fin equations

Typical accuracy: < 5% relative L2 error compared to FDM reference.

## 🔧 Configuration Example

```yaml
# configs/analysis.yaml

material: "80% aluminum, 20% copper"
power: 100.0

geometry:
  type: "heatsink"
  base_height: 0.003
  fin_height: 0.02
  num_fins: 7

initial_temperature: 25.0
ambient_temperature: 25.0
simulation_time: 1.0

boundary_conditions:
  top: convective
  bottom: fixed
  left: adiabatic
  right: adiabatic
```

## 📈 Output Example

```
=============================================
PINN THERMAL ANALYSIS
=============================================
Material: 80% aluminum, 20% copper
  Thermal conductivity: 243.5 W/(m·K)
  Max operating temp: 276°C
  Melting point: 625°C

Simulation Results:
  Max temperature: 85.3°C
  Mean temperature: 52.1°C
  Hotspots detected: 2

Thermal Safety:
  Safety margin: 190.7°C
  Risk level: LOW
  Geometry lifetime: Unlimited ✅
  Thermal headroom: 69.1%
  Max temperature capacity: 276°C
  System is thermally stable ✅

Recommendations:
  [LOW] Good thermal design - operating well within limits
=============================================
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_pinn.py -v

# Run tests in Docker
docker-compose run --rm streamlit pytest tests/ -v
```

## 🔧 Development

### Setting up Development Environment

```bash
# Install dev dependencies
pip install -r requirements.txt
pip install pre-commit black isort flake8

# Run linting
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

### Project Configuration

Configuration files are in `configs/`:
- `analysis.yaml` - Main analysis settings
- `training.yaml` - PINN training parameters
- `dataset.yaml` - Dataset generation config
- `inference.yaml` - Inference settings

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style
- Use Black for Python formatting
- Use isort for import sorting
- Follow PEP 8 guidelines
- Add type hints where possible
- Write docstrings for public functions

## 📞 Support

- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Email**: your-email@example.com

