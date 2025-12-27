"""
GPU-optimized training pipeline for thermal surrogate model.

Features:
- Mixed precision training (AMP)
- Gradient accumulation
- Learning rate scheduling
- Checkpointing
- TensorBoard logging
- Early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional, Callable, Union
import time
from dataclasses import dataclass
from tqdm import tqdm
import json


@dataclass
class TrainerConfig:
    """Configuration for the trainer."""
    # Optimization
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    betas: tuple = (0.9, 0.999)
    
    # Training
    epochs: int = 200
    gradient_clip: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    
    # Checkpointing
    save_every: int = 10
    keep_last: int = 5
    
    # Validation
    val_every: int = 5
    
    # Logging
    log_every: int = 100
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 30
    
    # Scheduler
    scheduler_type: str = "cosine"
    T_max: int = 200
    eta_min: float = 1e-5


class Trainer:
    """
    GPU-optimized trainer for thermal surrogate models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        output_dir: Path,
        device: str = "cuda",
    ):
        """
        Initialize trainer.
        
        Args:
            model: Neural operator model
            config: Training configuration
            output_dir: Directory for outputs
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )
        
        # Scheduler
        if config.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.T_max,
                eta_min=config.eta_min,
            )
        elif config.scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=50,
                gamma=0.5,
            )
        else:
            self.scheduler = None
        
        # Mixed precision
        self.scaler = GradScaler(device=self.device.type) if config.use_amp else None
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Logging
        self.writer = SummaryWriter(self.output_dir / "logs")
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = []
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            # Move data to device
            embeddings = batch['embedding'].to(self.device)
            physics = batch['physics'].to(self.device)
            targets = batch['temperature'].to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.config.use_amp:
                with autocast(device_type=self.device.type):
                    predictions = self.model(embeddings, physics)
                    loss = self.criterion(predictions, targets)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(embeddings, physics)
                loss = self.criterion(predictions, targets)
                
                loss.backward()
                
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            if self.global_step % self.config.log_every == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar(
                    'train/lr', 
                    self.optimizer.param_groups[0]['lr'],
                    self.global_step
                )
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / num_batches
        
        return {'train_loss': avg_loss}
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_mse = 0.0
        total_rel_l2 = 0.0
        total_max_error = 0.0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Validation"):
            embeddings = batch['embedding'].to(self.device)
            physics = batch['physics'].to(self.device)
            targets = batch['temperature'].to(self.device)
            
            predictions = self.model(embeddings, physics)
            
            # Compute metrics
            loss = self.criterion(predictions, targets)
            mse = torch.mean((predictions - targets) ** 2)
            
            # Relative L2 error
            rel_l2 = torch.norm(predictions - targets) / torch.norm(targets)
            
            # Max error
            max_err = torch.max(torch.abs(predictions - targets))
            
            total_loss += loss.item()
            total_mse += mse.item()
            total_rel_l2 += rel_l2.item()
            total_max_error += max_err.item()
            num_batches += 1
        
        metrics = {
            'val_loss': total_loss / num_batches,
            'val_mse': total_mse / num_batches,
            'val_rel_l2': total_rel_l2 / num_batches,
            'val_max_error': total_max_error / num_batches,
        }
        
        # Log to tensorboard
        for name, value in metrics.items():
            self.writer.add_scalar(f'val/{name}', value, epoch)
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, list]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Validate
            if epoch % self.config.val_every == 0:
                val_metrics = self.validate(val_loader, epoch)
                
                # Log
                print(f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.6f}, "
                      f"val_loss={val_metrics['val_loss']:.6f}, "
                      f"val_rel_l2={val_metrics['val_rel_l2']:.4f}")
                
                # Check for improvement
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.patience_counter = 0
                    self.save_checkpoint('best_model.pt', epoch, val_metrics)
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if (self.config.early_stopping and 
                    self.patience_counter >= self.config.patience):
                    print(f"Early stopping at epoch {epoch}")
                    break
                
                # Record history
                self.history.append({
                    'epoch': epoch,
                    **train_metrics,
                    **val_metrics,
                })
            
            # Periodic checkpointing
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt', epoch)
                self._cleanup_checkpoints()
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time/3600:.2f} hours")
        
        # Save training history
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.writer.close()
        
        return self.history
    
    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Optional[Dict] = None,
    ) -> None:
        """Save a checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'metrics': metrics,
            'config': self.config,
        }
        
        torch.save(checkpoint, self.output_dir / filename)
    
    def load_checkpoint(self, path: Path) -> int:
        """Load a checkpoint and return the epoch number."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        self.global_step = checkpoint['global_step']
        
        return checkpoint['epoch']
    
    def _cleanup_checkpoints(self) -> None:
        """Keep only the last N checkpoints."""
        checkpoints = sorted(
            self.output_dir.glob('checkpoint_epoch_*.pt'),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        
        while len(checkpoints) > self.config.keep_last:
            checkpoints[0].unlink()
            checkpoints.pop(0)


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss for better gradient flow.
    
    Computes loss at multiple spatial resolutions.
    """
    
    def __init__(self, scales: list = [1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute multi-scale loss.
        
        Args:
            predictions: [B, T, H, W]
            targets: [B, T, H, W]
            
        Returns:
            Scalar loss
        """
        total_loss = 0.0
        
        for scale in self.scales:
            if scale == 1:
                pred_scaled = predictions
                target_scaled = targets
            else:
                pred_scaled = torch.nn.functional.avg_pool2d(
                    predictions, scale
                )
                target_scaled = torch.nn.functional.avg_pool2d(
                    targets, scale
                )
            
            total_loss += self.mse(pred_scaled, target_scaled)
        
        return total_loss / len(self.scales)


class PhysicsInformedLoss(nn.Module):
    """
    Loss that includes physics constraints.
    
    Adds a soft constraint on the heat equation residual.
    """
    
    def __init__(
        self,
        alpha_range: tuple = (0.01, 0.5),
        physics_weight: float = 0.1,
    ):
        super().__init__()
        self.alpha_range = alpha_range
        self.physics_weight = physics_weight
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        physics_params: torch.Tensor,
    ) -> tuple:
        """
        Compute physics-informed loss.
        
        Args:
            predictions: [B, T, H, W]
            targets: [B, T, H, W]
            physics_params: [B, 4] - (alpha, source_x, source_y, intensity)
            
        Returns:
            (total_loss, data_loss, physics_loss)
        """
        # Data fitting loss
        data_loss = self.mse(predictions, targets)
        
        # Physics residual (simplified - full implementation would use autodiff)
        # Here we just penalize temporal smoothness as a proxy
        dt = predictions[:, 1:] - predictions[:, :-1]
        physics_loss = torch.mean(dt ** 2)
        
        total_loss = data_loss + self.physics_weight * physics_loss
        
        return total_loss, data_loss, physics_loss
