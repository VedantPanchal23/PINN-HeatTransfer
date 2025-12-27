"""
Geometry Encoder - Encodes 2D geometry images to latent representations.

Key design principle: Geometry embeddings are pre-computed once and cached.
During neural operator training, we DO NOT re-encode geometries.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, List
import numpy as np
from pathlib import Path
import h5py
from tqdm import tqdm


class GeometryEncoder(nn.Module):
    """
    CNN-based geometry encoder.
    
    Encodes binary geometry masks into fixed-size latent vectors.
    Uses a pretrained backbone (ResNet) for robust feature extraction.
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        embedding_dim: int = 512,
        freeze_backbone: bool = True,
    ):
        """
        Initialize the geometry encoder.
        
        Args:
            backbone: Backbone architecture ('resnet18', 'resnet34', 'resnet50')
            pretrained: Whether to use pretrained weights
            embedding_dim: Output embedding dimension
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.freeze_backbone = freeze_backbone
        
        # Load backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            backbone_dim = 512
        elif backbone == "resnet34":
            self.backbone = models.resnet34(
                weights=models.ResNet34_Weights.DEFAULT if pretrained else None
            )
            backbone_dim = 512
        elif backbone == "resnet50":
            self.backbone = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            backbone_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Modify first conv layer to accept single-channel input
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, 64,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )
        
        # Initialize with mean of original weights
        with torch.no_grad():
            self.backbone.conv1.weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )
        
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()
        
        # Projection head to desired embedding dimension
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode geometry mask to latent vector.
        
        Args:
            x: Geometry mask [B, H, W] or [B, 1, H, W]
            
        Returns:
            Geometry embedding [B, embedding_dim]
        """
        # Ensure proper shape
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Normalize to [0, 1] if not already
        if x.max() > 1.0:
            x = x / 255.0
        
        # Extract features
        features = self.backbone(x)
        
        # Project to embedding space
        embedding = self.projector(features)
        
        return embedding
    
    def encode_batch(
        self,
        masks: np.ndarray,
        batch_size: int = 32,
        device: str = "cuda",
    ) -> np.ndarray:
        """
        Encode a batch of geometry masks.
        
        Args:
            masks: Numpy array of masks [N, H, W]
            batch_size: Batch size for encoding
            device: Device to use
            
        Returns:
            Embeddings [N, embedding_dim]
        """
        self.eval()
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        num_samples = len(masks)
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch = masks[i:i + batch_size]
                batch_tensor = torch.tensor(
                    batch, dtype=torch.float32, device=device
                )
                
                emb = self(batch_tensor)
                embeddings.append(emb.cpu().numpy())
        
        return np.concatenate(embeddings, axis=0)


class PrecomputedEncoder:
    """
    Manages pre-computed geometry embeddings.
    
    This class ensures that geometry embeddings are computed once
    and loaded during training without re-encoding.
    """
    
    def __init__(
        self,
        encoder: Optional[GeometryEncoder] = None,
        cache_path: Optional[Path] = None,
    ):
        """
        Initialize the precomputed encoder.
        
        Args:
            encoder: GeometryEncoder instance for computing embeddings
            cache_path: Path to cache file for embeddings
        """
        self.encoder = encoder
        self.cache_path = Path(cache_path) if cache_path else None
        self.embeddings: Dict[str, np.ndarray] = {}
    
    def precompute(
        self,
        geometry_samples: List[Dict],
        batch_size: int = 32,
        device: str = "cuda",
        save: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Precompute embeddings for all geometries.
        
        Args:
            geometry_samples: List of geometry samples with 'mask' and 'hash' keys
            batch_size: Batch size for encoding
            device: Device to use
            save: Whether to save to cache
            
        Returns:
            Dictionary mapping geometry hash to embedding
        """
        if self.encoder is None:
            raise ValueError("Encoder not provided")
        
        # Stack all masks
        masks = np.stack([s['mask'] for s in geometry_samples])
        hashes = [s['hash'] for s in geometry_samples]
        
        print(f"Precomputing embeddings for {len(masks)} geometries...")
        
        # Encode
        embeddings = self.encoder.encode_batch(masks, batch_size, device)
        
        # Store in dictionary
        self.embeddings = {h: embeddings[i] for i, h in enumerate(hashes)}
        
        # Save to cache
        if save and self.cache_path:
            self.save(self.cache_path)
        
        return self.embeddings
    
    def get_embedding(self, geometry_hash: str) -> np.ndarray:
        """
        Get pre-computed embedding for a geometry.
        
        Args:
            geometry_hash: Unique hash of the geometry
            
        Returns:
            Geometry embedding
        """
        if geometry_hash not in self.embeddings:
            raise KeyError(f"Embedding not found for hash: {geometry_hash}")
        
        return self.embeddings[geometry_hash]
    
    def save(self, path: Path) -> None:
        """
        Save embeddings to HDF5 file.
        
        Args:
            path: Path to save file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(path, 'w') as f:
            # Store hashes as a dataset
            hashes = list(self.embeddings.keys())
            f.create_dataset(
                'hashes',
                data=np.array(hashes, dtype='S16'),
            )
            
            # Store embeddings
            embeddings = np.stack(list(self.embeddings.values()))
            f.create_dataset('embeddings', data=embeddings)
        
        print(f"Saved {len(self.embeddings)} embeddings to {path}")
    
    def load(self, path: Path) -> None:
        """
        Load embeddings from HDF5 file.
        
        Args:
            path: Path to load from
        """
        path = Path(path)
        
        with h5py.File(path, 'r') as f:
            hashes = [h.decode() for h in f['hashes'][:]]
            embeddings = f['embeddings'][:]
            
            self.embeddings = {h: embeddings[i] for i, h in enumerate(hashes)}
        
        print(f"Loaded {len(self.embeddings)} embeddings from {path}")
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __contains__(self, geometry_hash: str) -> bool:
        return geometry_hash in self.embeddings


class MultiScaleGeometryEncoder(nn.Module):
    """
    Multi-scale geometry encoder that captures features at different resolutions.
    
    This encoder extracts features from multiple layers of the backbone
    to capture both local and global geometry information.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        pretrained: bool = True,
    ):
        """
        Initialize multi-scale encoder.
        
        Args:
            embedding_dim: Output embedding dimension
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Use ResNet18 as backbone
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        
        # Modify first conv for single channel
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight = nn.Parameter(
                resnet.conv1.weight.mean(dim=1, keepdim=True)
            )
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # Feature extraction layers
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
        # Adaptive pooling for each scale
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Fusion layer
        total_features = 64 + 128 + 256 + 512
        self.fusion = nn.Sequential(
            nn.Linear(total_features, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        # Freeze early layers
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale forward pass.
        
        Args:
            x: Geometry mask [B, H, W] or [B, 1, H, W]
            
        Returns:
            Geometry embedding [B, embedding_dim]
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Multi-scale features
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        
        # Pool and concatenate
        p1 = self.pool(f1).flatten(1)
        p2 = self.pool(f2).flatten(1)
        p3 = self.pool(f3).flatten(1)
        p4 = self.pool(f4).flatten(1)
        
        features = torch.cat([p1, p2, p3, p4], dim=1)
        
        # Fuse to final embedding
        embedding = self.fusion(features)
        
        return embedding
