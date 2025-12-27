"""
Configuration utilities for loading and managing YAML configs.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: Path) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save to
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base: Base configuration
        override: Override configuration (takes precedence)
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


@dataclass
class ExperimentConfig:
    """
    Comprehensive experiment configuration.
    """
    # Experiment info
    name: str = "thermal_surrogate"
    seed: int = 42
    
    # Paths
    data_dir: Path = Path("data/processed")
    output_dir: Path = Path("outputs")
    
    # Model config
    model_type: str = "fno"
    geometry_dim: int = 512
    physics_dim: int = 4
    
    # FNO specific
    fno_modes: int = 16
    fno_width: int = 64
    fno_layers: int = 4
    
    # Training
    batch_size: int = 16
    epochs: int = 200
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    
    # Misc
    device: str = "cuda"
    num_workers: int = 4
    use_amp: bool = True
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'ExperimentConfig':
        """Create config from YAML file."""
        config_dict = load_config(path)
        return cls(**flatten_config(config_dict))
    
    def to_yaml(self, path: Path) -> None:
        """Save config to YAML file."""
        save_config(self.__dict__, path)


def flatten_config(config: Dict, prefix: str = '') -> Dict:
    """
    Flatten nested configuration dictionary.
    
    Args:
        config: Nested configuration
        prefix: Prefix for nested keys
        
    Returns:
        Flattened dictionary
    """
    result = {}
    
    for key, value in config.items():
        new_key = f"{prefix}_{key}" if prefix else key
        
        if isinstance(value, dict):
            result.update(flatten_config(value, new_key))
        else:
            result[new_key] = value
    
    return result


def validate_config(config: Dict) -> bool:
    """
    Validate configuration for required fields.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If required fields are missing
    """
    required_fields = ['training', 'model']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    
    # Validate model type
    valid_models = ['fno', 'deeponet', 'unet_fno']
    model_type = config.get('training', {}).get('model', {}).get('type', '')
    
    if model_type and model_type not in valid_models:
        raise ValueError(f"Invalid model type: {model_type}. Must be one of {valid_models}")
    
    return True
