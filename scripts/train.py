"""
Training Script for Thermal Surrogate Model

Trains a neural operator (FNO, DeepONet, or UNet-FNO) to predict
temperature fields from geometry embeddings and physics parameters.
"""

import argparse
from pathlib import Path
import yaml
import torch
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import FNO2d, DeepONet, UNetFNO
from src.models.fno import ConditionalFNO
from src.training import ThermalDataset, create_dataloaders, Trainer
from src.training.trainer import TrainerConfig
from src.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train thermal surrogate model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training.yaml"),
        help="Path to training config"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Data directory"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=['fno', 'conditional_fno', 'deeponet', 'unet_fno'],
        help="Override model type from config"
    )
    return parser.parse_args()


def create_model(config: dict) -> torch.nn.Module:
    """Create model based on configuration."""
    model_config = config['training']['model']
    model_type = model_config.get('type', 'fno')
    
    geometry_dim = model_config['encoder']['embedding_dim']
    physics_dim = 4  # (alpha, source_x, source_y, intensity)
    
    if model_type == 'fno':
        fno_config = model_config.get('fno', {})
        model = FNO2d(
            modes=fno_config.get('modes', 16),
            width=fno_config.get('width', 64),
            num_layers=fno_config.get('num_layers', 4),
            padding=fno_config.get('padding', 9),
            geometry_dim=geometry_dim,
            physics_dim=physics_dim,
            time_steps=config['training']['data'].get('time_steps', 50),
        )
    elif model_type == 'conditional_fno':
        fno_config = model_config.get('fno', {})
        model = ConditionalFNO(
            modes=fno_config.get('modes', 16),
            width=fno_config.get('width', 64),
            num_layers=fno_config.get('num_layers', 4),
            geometry_dim=geometry_dim,
            physics_dim=physics_dim,
            time_steps=config['training']['data'].get('time_steps', 50),
        )
    elif model_type == 'deeponet':
        don_config = model_config.get('deeponet', {})
        model = DeepONet(
            geometry_dim=geometry_dim,
            physics_dim=physics_dim,
            branch_layers=don_config.get('branch_layers', [512, 256, 256, 128]),
            trunk_layers=don_config.get('trunk_layers', [128, 128, 128]),
        )
    elif model_type == 'unet_fno':
        model = UNetFNO(
            geometry_dim=geometry_dim,
            physics_dim=physics_dim,
            base_channels=64,
            time_steps=config['training']['data'].get('time_steps', 50),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: {model_type}")
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override model type if specified
    if args.model:
        config['training']['model']['type'] = args.model
    
    training_cfg = config['training']
    
    # Create output directory
    experiment_name = training_cfg.get('experiment_name', 'thermal_surrogate')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(training_cfg['output_dir']) / f"{experiment_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print("=" * 60)
    print("THERMAL SURROGATE MODEL TRAINING")
    print("=" * 60)
    print(f"Experiment: {experiment_name}")
    print(f"Output: {output_dir}")
    
    # Set seed
    seed = training_cfg.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Device
    device = training_cfg.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print(f"Device: {device}")
    
    # Create dataloaders
    data_cfg = training_cfg['data']
    data_dir = args.data_dir
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_path=data_dir / "train.h5",
        val_path=data_dir / "val.h5",
        test_path=data_dir / "test.h5",
        embeddings_path=data_dir / "embeddings.h5",
        batch_size=data_cfg['batch_size'],
        num_workers=data_cfg['num_workers'],
        pin_memory=data_cfg['pin_memory'],
    )
    
    print(f"\nData loaded:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create model
    model = create_model(config)
    
    # Create trainer config
    opt_cfg = training_cfg['optimizer']
    sched_cfg = training_cfg.get('scheduler', {})
    
    trainer_config = TrainerConfig(
        learning_rate=opt_cfg['lr'],
        weight_decay=opt_cfg['weight_decay'],
        betas=tuple(opt_cfg.get('betas', [0.9, 0.999])),
        epochs=training_cfg['epochs'],
        gradient_clip=training_cfg.get('gradient_clip', 1.0),
        use_amp=training_cfg.get('amp', {}).get('enabled', True),
        save_every=training_cfg.get('checkpoint', {}).get('save_every', 10),
        keep_last=training_cfg.get('checkpoint', {}).get('keep_last', 5),
        val_every=training_cfg.get('validation', {}).get('every', 5),
        log_every=training_cfg.get('logging', {}).get('log_every', 100),
        early_stopping=training_cfg.get('early_stopping', {}).get('enabled', True),
        patience=training_cfg.get('early_stopping', {}).get('patience', 30),
        scheduler_type=sched_cfg.get('type', 'cosine'),
        T_max=sched_cfg.get('T_max', 200),
        eta_min=sched_cfg.get('eta_min', 1e-5),
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=trainer_config,
        output_dir=output_dir,
        device=device,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    history = trainer.train(train_loader, val_loader)
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    # Load best model
    best_checkpoint = output_dir / 'best_model.pt'
    if best_checkpoint.exists():
        trainer.load_checkpoint(best_checkpoint)
    
    test_metrics = trainer.validate(test_loader, epoch=0)
    
    print(f"\nTest Results:")
    print(f"  MSE: {test_metrics['val_mse']:.6e}")
    print(f"  Relative L2: {test_metrics['val_rel_l2']:.4f}")
    print(f"  Max Error: {test_metrics['val_max_error']:.4f}")
    
    # Save test metrics
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model saved to: {best_checkpoint}")
    print(f"Logs: {output_dir / 'logs'}")


if __name__ == "__main__":
    main()
