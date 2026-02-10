"""Configuration management for Energy Management RL project.

This module provides configuration management using OmegaConf for
flexible and reproducible experiments.
"""

from __future__ import annotations

from omegaconf import OmegaConf
from typing import Dict, Any, Optional
import os
from pathlib import Path


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for energy management RL.
    
    Returns:
        Default configuration dictionary
    """
    return {
        # Environment configuration
        'env': {
            'max_storage_capacity': 100.0,
            'max_demand': 50.0,
            'max_generation': 80.0,
            'price_variability': 0.3,
            'episode_length': 96,  # 24 hours * 4 (15-min intervals)
            'seed': 42,
        },
        
        # Agent configuration
        'agent': {
            'type': 'sac',  # 'dqn' or 'sac'
            'lr': 3e-4,
            'gamma': 0.99,
            'buffer_size': 100000,
            'batch_size': 256,
            'hidden_dim': 256,
        },
        
        # DQN specific
        'dqn': {
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'target_update': 100,
        },
        
        # SAC specific
        'sac': {
            'tau': 0.005,
            'alpha': None,  # Auto-tune if None
        },
        
        # Training configuration
        'training': {
            'num_episodes': 2000,
            'eval_frequency': 100,
            'save_frequency': 500,
            'max_steps_per_episode': 1000,
            'log_dir': 'logs',
        },
        
        # Evaluation configuration
        'evaluation': {
            'num_episodes': 100,
            'eval_dir': 'evaluation',
            'seeds': None,  # Will be generated if None
        },
        
        # Device configuration
        'device': {
            'use_cuda': True,
            'use_mps': True,  # Apple Silicon
            'device': None,  # Auto-detect if None
        },
        
        # Logging configuration
        'logging': {
            'level': 'INFO',
            'use_wandb': False,
            'use_tensorboard': True,
            'project_name': 'energy-management-rl',
        },
        
        # Reproducibility
        'reproducibility': {
            'seed': 42,
            'deterministic': True,
        },
    }


def load_config(config_path: Optional[str] = None) -> OmegaConf:
    """Load configuration from file or use defaults.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OmegaConf configuration object
    """
    default_config = get_default_config()
    
    if config_path and os.path.exists(config_path):
        config = OmegaConf.load(config_path)
        # Merge with defaults
        config = OmegaConf.merge(default_config, config)
    else:
        config = OmegaConf.create(default_config)
    
    return config


def save_config(config: OmegaConf, save_path: str) -> None:
    """Save configuration to file.
    
    Args:
        config: Configuration to save
        save_path: Path to save configuration
    """
    OmegaConf.save(config, save_path)


def get_device_config(config: OmegaConf) -> str:
    """Get device configuration based on available hardware.
    
    Args:
        config: Configuration object
        
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    import torch
    
    if config.device.device is not None:
        return config.device.device
    
    if config.device.use_cuda and torch.cuda.is_available():
        return 'cuda'
    elif config.device.use_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def setup_reproducibility(config: OmegaConf) -> None:
    """Setup reproducibility settings.
    
    Args:
        config: Configuration object
    """
    import torch
    import numpy as np
    import random
    
    seed = config.reproducibility.seed
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    if config.reproducibility.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
