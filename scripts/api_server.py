"""
FastAPI Server for Thermal Surrogate Model Inference.

Provides REST API endpoints for:
- Single geometry inference
- Batch inference
- Health checks
- Model information
"""

import io
import base64
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predictor import ThermalPredictor
from src.utils.config import load_config


# ============================================================================
# Pydantic Models
# ============================================================================

class PhysicsParams(BaseModel):
    """Physics parameters for thermal simulation."""
    thermal_diffusivity: float = Field(
        default=0.1,
        ge=0.001,
        le=1.0,
        description="Thermal diffusivity (alpha)"
    )
    heat_source_x: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Heat source X position (normalized)"
    )
    heat_source_y: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Heat source Y position (normalized)"
    )
    heat_source_intensity: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Heat source intensity"
    )


class TimeParams(BaseModel):
    """Time parameters for simulation."""
    t_start: float = Field(default=0.0, ge=0.0, description="Start time")
    t_end: float = Field(default=1.0, gt=0.0, description="End time")
    num_steps: int = Field(default=10, ge=1, le=100, description="Number of time steps")


class InferenceRequest(BaseModel):
    """Request model for inference with base64 encoded image."""
    geometry_image: str = Field(
        ...,
        description="Base64 encoded geometry image (black/white PNG)"
    )
    physics: PhysicsParams = Field(default_factory=PhysicsParams)
    time: TimeParams = Field(default_factory=TimeParams)
    return_format: str = Field(
        default="json",
        description="Output format: 'json' or 'numpy'"
    )


class InferenceResponse(BaseModel):
    """Response model for inference."""
    success: bool
    inference_time_ms: float
    temperature_field: Optional[List[List[List[float]]]] = None
    shape: Optional[List[int]] = None
    time_steps: Optional[List[float]] = None
    stats: Optional[Dict[str, float]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    model_type: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_type: str
    device: str
    input_resolution: int
    embedding_dim: int
    num_parameters: int
    onnx_available: bool


# ============================================================================
# Application Setup
# ============================================================================

app = FastAPI(
    title="Thermal Surrogate Model API",
    description="Fast 2D transient heat transfer prediction using neural operators",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[ThermalPredictor] = None


# ============================================================================
# Startup / Shutdown
# ============================================================================

@app.on_event("startup")
async def load_model():
    """Load the model on startup."""
    global predictor
    
    config_path = Path(__file__).parent.parent / "configs" / "inference.yaml"
    
    try:
        if config_path.exists():
            config = load_config(str(config_path))
            checkpoint_path = config.get("inference", {}).get(
                "checkpoint_path", 
                "outputs/checkpoints/best_model.pt"
            )
        else:
            checkpoint_path = "outputs/checkpoints/best_model.pt"
        
        # Check if checkpoint exists
        if Path(checkpoint_path).exists():
            predictor = ThermalPredictor(
                checkpoint_path=checkpoint_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            print(f"Model loaded from {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("API will run but inference endpoints will return errors")
            predictor = None
            
    except Exception as e:
        print(f"Error loading model: {e}")
        predictor = None


@app.on_event("shutdown")
async def cleanup():
    """Cleanup on shutdown."""
    global predictor
    predictor = None


# ============================================================================
# Utility Functions
# ============================================================================

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array."""
    try:
        # Remove data URL prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        return np.array(image) / 255.0
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")


def encode_array_base64(array: np.ndarray) -> str:
    """Encode numpy array to base64."""
    buffer = io.BytesIO()
    np.save(buffer, array)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Thermal Surrogate Model API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        device=str(predictor.device) if predictor else "N/A",
        model_type=predictor.model_type if predictor else "N/A",
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_type=predictor.model_type,
        device=str(predictor.device),
        input_resolution=predictor.resolution,
        embedding_dim=predictor.embedding_dim,
        num_parameters=sum(
            p.numel() for p in predictor.model.parameters()
        ),
        onnx_available=Path("outputs/checkpoints/model.onnx").exists(),
    )


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """
    Run inference on a geometry image.
    
    Accepts a base64-encoded black/white geometry image and physics parameters.
    Returns the predicted temperature field T(t, x, y).
    """
    if predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please ensure checkpoint exists."
        )
    
    try:
        # Decode image
        geometry = decode_base64_image(request.geometry_image)
        
        # Create time array
        time_steps = np.linspace(
            request.time.t_start,
            request.time.t_end,
            request.time.num_steps,
        )
        
        # Prepare physics parameters
        physics_params = {
            "alpha": request.physics.thermal_diffusivity,
            "heat_source": {
                "x": request.physics.heat_source_x,
                "y": request.physics.heat_source_y,
                "intensity": request.physics.heat_source_intensity,
            },
        }
        
        # Run inference
        start_time = time.perf_counter()
        temperature_field = predictor.predict(
            geometry=geometry,
            time_steps=time_steps,
            physics_params=physics_params,
        )
        inference_time = (time.perf_counter() - start_time) * 1000
        
        # Compute statistics
        stats = {
            "min": float(temperature_field.min()),
            "max": float(temperature_field.max()),
            "mean": float(temperature_field.mean()),
            "std": float(temperature_field.std()),
        }
        
        # Format response
        if request.return_format == "numpy":
            # Return as base64 encoded numpy array
            return JSONResponse(content={
                "success": True,
                "inference_time_ms": inference_time,
                "temperature_field_b64": encode_array_base64(temperature_field),
                "shape": list(temperature_field.shape),
                "time_steps": time_steps.tolist(),
                "stats": stats,
            })
        else:
            # Return as JSON (may be large)
            return InferenceResponse(
                success=True,
                inference_time_ms=inference_time,
                temperature_field=temperature_field.tolist(),
                shape=list(temperature_field.shape),
                time_steps=time_steps.tolist(),
                stats=stats,
            )
            
    except ValueError as e:
        return InferenceResponse(
            success=False,
            inference_time_ms=0,
            error=str(e),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/file")
async def predict_from_file(
    file: UploadFile = File(..., description="Geometry image file (PNG/JPG)"),
    thermal_diffusivity: float = Form(default=0.1),
    heat_source_x: float = Form(default=0.5),
    heat_source_y: float = Form(default=0.5),
    heat_source_intensity: float = Form(default=1.0),
    t_start: float = Form(default=0.0),
    t_end: float = Form(default=1.0),
    num_steps: int = Form(default=10),
):
    """
    Run inference from an uploaded geometry image file.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure checkpoint exists."
        )
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")
        geometry = np.array(image) / 255.0
        
        # Create time array
        time_steps = np.linspace(t_start, t_end, num_steps)
        
        # Physics parameters
        physics_params = {
            "alpha": thermal_diffusivity,
            "heat_source": {
                "x": heat_source_x,
                "y": heat_source_y,
                "intensity": heat_source_intensity,
            },
        }
        
        # Run inference
        start_time = time.perf_counter()
        temperature_field = predictor.predict(
            geometry=geometry,
            time_steps=time_steps,
            physics_params=physics_params,
        )
        inference_time = (time.perf_counter() - start_time) * 1000
        
        return {
            "success": True,
            "inference_time_ms": inference_time,
            "shape": list(temperature_field.shape),
            "time_steps": time_steps.tolist(),
            "stats": {
                "min": float(temperature_field.min()),
                "max": float(temperature_field.max()),
                "mean": float(temperature_field.mean()),
            },
            "temperature_field": temperature_field.tolist(),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    thermal_diffusivity: float = Form(default=0.1),
    t_end: float = Form(default=1.0),
    num_steps: int = Form(default=10),
):
    """
    Run batch inference on multiple geometry images.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    results = []
    total_start = time.perf_counter()
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("L")
            geometry = np.array(image) / 255.0
            
            time_steps = np.linspace(0, t_end, num_steps)
            physics_params = {"alpha": thermal_diffusivity}
            
            start_time = time.perf_counter()
            temperature_field = predictor.predict(
                geometry=geometry,
                time_steps=time_steps,
                physics_params=physics_params,
            )
            inference_time = (time.perf_counter() - start_time) * 1000
            
            results.append({
                "filename": file.filename,
                "success": True,
                "inference_time_ms": inference_time,
                "shape": list(temperature_field.shape),
                "stats": {
                    "min": float(temperature_field.min()),
                    "max": float(temperature_field.max()),
                    "mean": float(temperature_field.mean()),
                },
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e),
            })
    
    total_time = (time.perf_counter() - total_start) * 1000
    
    return {
        "total_files": len(files),
        "successful": sum(1 for r in results if r.get("success")),
        "total_time_ms": total_time,
        "results": results,
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
