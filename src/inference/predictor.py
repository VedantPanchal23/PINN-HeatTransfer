"""
Fast inference for thermal surrogate model.

Provides millisecond-scale predictions for new geometries.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union, Tuple
import time
from PIL import Image
import cv2


class ThermalPredictor:
    """
    Fast inference engine for thermal surrogate model.
    
    Features:
    - Millisecond-scale predictions
    - Automatic geometry encoding
    - Batch inference support
    - Benchmarking utilities
    """
    
    def __init__(
        self,
        model_path: Path,
        encoder_path: Optional[Path] = None,
        device: str = "cuda",
        compile_model: bool = True,
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            encoder_path: Path to geometry encoder (optional if embeddings pre-computed)
            device: Device for inference
            compile_model: Whether to use torch.compile for optimization
        """
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Optionally compile for faster inference
        if compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("Model compiled with torch.compile")
            except Exception as e:
                print(f"Could not compile model: {e}")
        
        # Load encoder if provided
        self.encoder = None
        if encoder_path:
            self.encoder = self._load_encoder(encoder_path)
            self.encoder.eval()
        
        # Warmup
        self._warmup()
    
    def _load_model(self, path: Path) -> nn.Module:
        """Load trained model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Detect model type from checkpoint
        state_dict = checkpoint['model_state_dict']
        
        # Import models
        from ..models import FNO2d, DeepONet, UNetFNO
        
        # Infer model architecture from state dict keys
        if 'spectral_convs.0.weights1' in state_dict:
            # ConditionalFNO
            from ..models.fno import ConditionalFNO
            model = ConditionalFNO()
        elif 'fno_blocks.0.spectral_conv.weights1' in state_dict:
            model = FNO2d()
        elif 'branch.network.0.weight' in state_dict:
            model = DeepONet()
        elif 'down1.spectral.weights1' in state_dict:
            model = UNetFNO()
        else:
            # Default to FNO
            model = FNO2d()
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        
        return model
    
    def _load_encoder(self, path: Path) -> nn.Module:
        """Load geometry encoder."""
        from ..encoder import GeometryEncoder
        
        encoder = GeometryEncoder()
        encoder.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
        encoder.to(self.device)
        
        return encoder
    
    def _warmup(self, num_iterations: int = 10):
        """Warmup GPU for accurate benchmarking."""
        print("Warming up...")
        
        dummy_embedding = torch.randn(1, 512, device=self.device)
        dummy_physics = torch.randn(1, 4, device=self.device)
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(dummy_embedding, dummy_physics)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
    
    def encode_geometry(
        self,
        geometry: Union[np.ndarray, str, Path],
        resolution: int = 128,
    ) -> torch.Tensor:
        """
        Encode a geometry image to embedding.
        
        Args:
            geometry: Geometry as numpy array, file path, or PIL Image
            resolution: Target resolution for encoding
            
        Returns:
            Geometry embedding [1, embedding_dim]
        """
        if self.encoder is None:
            raise ValueError("Encoder not loaded. Provide encoder_path during init.")
        
        # Load and preprocess geometry
        if isinstance(geometry, (str, Path)):
            img = Image.open(geometry).convert('L')
            mask = np.array(img).astype(np.float32) / 255.0
        elif isinstance(geometry, np.ndarray):
            mask = geometry.astype(np.float32)
            if mask.max() > 1.0:
                mask = mask / 255.0
        else:
            raise ValueError(f"Unsupported geometry type: {type(geometry)}")
        
        # Resize if needed
        if mask.shape != (resolution, resolution):
            mask = cv2.resize(mask, (resolution, resolution))
        
        # Convert to tensor
        mask_tensor = torch.tensor(mask, device=self.device).unsqueeze(0)
        
        # Encode
        with torch.no_grad():
            embedding = self.encoder(mask_tensor)
        
        return embedding
    
    @torch.no_grad()
    def predict(
        self,
        geometry_embedding: torch.Tensor,
        thermal_diffusivity: float,
        source_x: float,
        source_y: float,
        source_intensity: float,
        resolution: int = 128,
    ) -> np.ndarray:
        """
        Predict temperature field for given geometry and physics.
        
        Args:
            geometry_embedding: Pre-computed geometry embedding [B, dim]
            thermal_diffusivity: Thermal diffusivity (alpha)
            source_x: Heat source X position [0, 1]
            source_y: Heat source Y position [0, 1]
            source_intensity: Heat source intensity
            resolution: Output spatial resolution
            
        Returns:
            Temperature field [T, H, W]
        """
        # Prepare physics parameters
        physics_params = torch.tensor(
            [[thermal_diffusivity, source_x, source_y, source_intensity]],
            device=self.device,
            dtype=torch.float32,
        )
        
        # Ensure embedding is on device
        if geometry_embedding.device != self.device:
            geometry_embedding = geometry_embedding.to(self.device)
        
        # Inference
        output = self.model(geometry_embedding, physics_params, resolution=resolution)
        
        return output.squeeze(0).cpu().numpy()
    
    @torch.no_grad()
    def predict_from_image(
        self,
        geometry_path: Union[str, Path],
        thermal_diffusivity: float,
        source_x: float,
        source_y: float,
        source_intensity: float,
        resolution: int = 128,
    ) -> np.ndarray:
        """
        Predict temperature field from geometry image file.
        
        Args:
            geometry_path: Path to geometry image (PNG)
            thermal_diffusivity: Thermal diffusivity
            source_x: Heat source X position
            source_y: Heat source Y position
            source_intensity: Heat source intensity
            resolution: Output resolution
            
        Returns:
            Temperature field [T, H, W]
        """
        # Encode geometry
        embedding = self.encode_geometry(geometry_path, resolution)
        
        # Predict
        return self.predict(
            embedding,
            thermal_diffusivity,
            source_x,
            source_y,
            source_intensity,
            resolution,
        )
    
    @torch.no_grad()
    def predict_batch(
        self,
        geometry_embeddings: torch.Tensor,
        physics_params: torch.Tensor,
        resolution: int = 128,
    ) -> np.ndarray:
        """
        Batch prediction for multiple samples.
        
        Args:
            geometry_embeddings: [B, embedding_dim]
            physics_params: [B, 4]
            resolution: Output resolution
            
        Returns:
            Temperature fields [B, T, H, W]
        """
        if geometry_embeddings.device != self.device:
            geometry_embeddings = geometry_embeddings.to(self.device)
        if physics_params.device != self.device:
            physics_params = physics_params.to(self.device)
        
        output = self.model(geometry_embeddings, physics_params, resolution=resolution)
        
        return output.cpu().numpy()
    
    def benchmark(
        self,
        num_warmup: int = 10,
        num_runs: int = 100,
        batch_size: int = 1,
        resolution: int = 128,
    ) -> Dict[str, float]:
        """
        Benchmark inference speed.
        
        Args:
            num_warmup: Number of warmup iterations
            num_runs: Number of timed runs
            batch_size: Batch size for inference
            resolution: Spatial resolution
            
        Returns:
            Dictionary with timing statistics
        """
        # Create dummy inputs
        dummy_embedding = torch.randn(batch_size, 512, device=self.device)
        dummy_physics = torch.randn(batch_size, 4, device=self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model(dummy_embedding, dummy_physics, resolution=resolution)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Timed runs
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = self.model(dummy_embedding, dummy_physics, resolution=resolution)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
        
        times = np.array(times)
        
        return {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'median_ms': float(np.median(times)),
            'throughput_samples_per_sec': float(batch_size / (np.mean(times) / 1000)),
        }


class ONNXPredictor:
    """
    ONNX Runtime-based predictor for deployment.
    
    Provides even faster inference using ONNX Runtime.
    """
    
    def __init__(
        self,
        onnx_path: Path,
        device: str = "cuda",
    ):
        """
        Initialize ONNX predictor.
        
        Args:
            onnx_path: Path to ONNX model
            device: Device for inference
        """
        import onnxruntime as ort
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if device == "cpu":
            providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=providers,
        )
        
        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
    
    def predict(
        self,
        geometry_embedding: np.ndarray,
        physics_params: np.ndarray,
    ) -> np.ndarray:
        """
        Run inference with ONNX Runtime.
        
        Args:
            geometry_embedding: [B, embedding_dim]
            physics_params: [B, 4]
            
        Returns:
            Temperature field [B, T, H, W]
        """
        inputs = {
            self.input_names[0]: geometry_embedding.astype(np.float32),
            self.input_names[1]: physics_params.astype(np.float32),
        }
        
        outputs = self.session.run(self.output_names, inputs)
        
        return outputs[0]
    
    def benchmark(
        self,
        num_warmup: int = 10,
        num_runs: int = 100,
        batch_size: int = 1,
    ) -> Dict[str, float]:
        """Benchmark ONNX inference speed."""
        # Dummy inputs
        embedding = np.random.randn(batch_size, 512).astype(np.float32)
        physics = np.random.randn(batch_size, 4).astype(np.float32)
        
        # Warmup
        for _ in range(num_warmup):
            _ = self.predict(embedding, physics)
        
        # Timed runs
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.predict(embedding, physics)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        times = np.array(times)
        
        return {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
        }
