#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from typing import List


current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.dirname(current_dir)  # scripts/
morl_modules_dir = os.path.dirname(scripts_dir)  # MORL_modules/
project_root = os.path.dirname(morl_modules_dir)  # MORL_proj/

# Add to Python path
sys.path.append(project_root)
sys.path.append(morl_modules_dir)

# ============================================================================
# CORE SYSTEM SWITCHES - MODIFY THESE TRUE/FALSE VALUES
# ============================================================================

# Algorithm Settings
USE_MOSAC = True  # Use MOSAC algorithm (False = use SAC)
USE_SCALARIZED_WRAPPER = True  # Use ScalarizedMOPCSWrapper
USE_CUDA = True  # Use CUDA if available

# Training Settings
SAVE_MODELS = True  # Save trained models
SAVE_REPLAY_BUFFER = False  # Save replay buffers (large files)
USE_TENSORBOARD = True  # Enable TensorBoard logging
USE_MONITOR_WRAPPER = True  # Use Monitor wrapper for episode tracking

# Analysis Settings
CREATE_PLOTS = True  # Generate analysis plots
SAVE_RAW_DATA = True  # Save raw numerical results
GENERATE_REPORTS = True  # Generate markdown reports
PLOT_INDIVIDUAL_CONFIGS = True  # Create per-configuration plots

# Experiment Settings
RUN_QUICK_EVAL = True  # Run quick evaluation after training
CONFIRM_BATCH_RUNS = True  # Ask confirmation for batch experiments
AUTO_CLEANUP_LOGS = False  # Automatically clean old logs


# Logging Settings
VERBOSE_LOGGING = True  # Detailed logging output

# ============================================================================
# WEIGHT CONFIGURATIONS
# ============================================================================

WEIGHT_CONFIGURATIONS = {
    'economic_only': {
        'name': 'economic_only',
        'description': 'Pure Economic Optimization',
        'weights': [1.0, 0.0, 0.0, 0.0],
        'expected_behavior': 'High profits, aggressive trading, potential battery stress'
    },
    'balanced': {
        'name': 'balanced',
        'description': 'Balanced Multi-Objective',
        'weights': [0.25, 0.25, 0.25, 0.25],
        'expected_behavior': 'Moderate performance across all objectives'
    },
    'economic_battery': {
        'name': 'economic_battery',
        'description': 'Economic + Battery Health Focus',
        'weights': [0.5, 0.5, 0.0, 0.0],
        'expected_behavior': 'Good profits while preserving battery health'
    },
    'battery_grid': {
        'name': 'battery_grid',
        'description': 'Battery Health + Grid Support',
        'weights': [0.0, 0.5, 0.5, 0.0],
        'expected_behavior': 'Focus on infrastructure health and grid stability'
    },
    'autonomous': {
        'name': 'autonomous',
        'description': 'Energy Autonomy Focus',
        'weights': [0.2, 0.2, 0.2, 0.4],
        'expected_behavior': 'Maximize energy independence'
    },
    'grid_stability': {
        'name': 'grid_stability',
        'description': 'Grid Stability Focus',
        'weights': [0.1, 0.1, 0.8, 0.0],
        'expected_behavior': 'Prioritize grid support services'
    },
    'battery_preservation': {
        'name': 'battery_preservation',
        'description': 'Battery Preservation Focus',
        'weights': [0.0, 0.8, 0.1, 0.1],
        'expected_behavior': 'Minimize battery degradation'
    }
}

# Objective names corresponding to the weight indices
OBJECTIVE_NAMES = ['Economic Profit', 'Battery Health', 'Grid Support', 'Energy Autonomy']

# ============================================================================
# DEFAULT HYPERPARAMETERS
# ============================================================================

# Training Hyperparameters
DEFAULT_TRAINING_PARAMS = {
    'timesteps': 50000,
    'learning_rate': 3e-4,
    'batch_size': 256,
    'buffer_size': 100000,
    'learning_starts': 1000,
    'tau': 0.005,
    'gamma': 0.99,
    'train_freq': 1,
    'gradient_steps': 1,
    'target_update_interval': 1,
}

# MOSAC-specific hyperparameters
MOSAC_PARAMS = {
    'num_objectives': 4,
    'policy_kwargs': {
        'net_arch': {'pi': [256, 256], 'qf': [256, 256]},
    },
    'replay_buffer_kwargs': {
        'num_objectives': 4
    }
}

# SAC-specific hyperparameters (fallback)
SAC_PARAMS = {
    'policy_kwargs': {
        'net_arch': {'pi': [256, 256], 'qf': [256, 256]},
    }
}

# Environment parameters
ENVIRONMENT_PARAMS = {
    'controller_name': 'EnergyNetController',
    'controller_module': 'energy_net.controllers',
    'env_config_path': 'energy_net/configs/environment_config.yaml',
    'iso_config_path': 'energy_net/configs/iso_config.yaml',
    'pcs_unit_config_path': 'energy_net/configs/pcs_unit_config.yaml',
    'cost_type': 'CONSTANT',  # Will be converted to CostType.CONSTANT
    'pricing_policy': 'QUADRATIC',  # Will be converted to PricingPolicy.QUADRATIC
    'demand_pattern': 'SINUSOIDAL'  # Will be converted to DemandPattern.SINUSOIDAL
}

# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================

# Default log directories
DEFAULT_LOG_DIRS = {
    'base': 'logs/morl_unified',
    'training': 'logs/morl_training',
    'batch': 'logs/batch_experiments',
    'analysis': 'logs/analysis',
    'models': 'logs/models',
    'tensorboard': 'logs/tensorboard'
}

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

# Quick evaluation parameters
QUICK_EVAL_PARAMS = {
    'n_episodes': 10,
    'deterministic': True,
    'max_episode_length': 1000
}

# Full evaluation parameters
FULL_EVAL_PARAMS = {
    'n_episodes': 100,
    'deterministic': True,
    'max_episode_length': 1000
}

# ============================================================================
# PLOTTING CONFIGURATION
# ============================================================================

# Matplotlib style settings
PLOT_STYLE = 'default'
PLOT_PALETTE = 'husl'
PLOT_DPI = 300
PLOT_FORMAT = 'png'

# Plot dimensions
PLOT_SIZES = {
    'single': (10, 6),
    'comparison': (12, 8),
    'comprehensive': (16, 12),
    'individual': (12, 8)
}

# Color schemes for different configurations
CONFIG_COLORS = {
    'economic_only': '#d62728',  # Red
    'balanced': '#2ca02c',  # Green
    'economic_battery': '#ff7f0e',  # Orange
    'battery_grid': '#1f77b4',  # Blue
    'autonomous': '#9467bd',  # Purple
    'grid_stability': '#8c564b',  # Brown
    'battery_preservation': '#e377c2'  # Pink
}

# ============================================================================
# ANALYSIS SETTINGS
# ============================================================================

# Performance metrics to track
PERFORMANCE_METRICS = [
    'mean_reward',
    'std_reward',
    'min_reward',
    'max_reward',
    'final_reward',
    'convergence_episode',
    'stability_metric'
]

# Window sizes for moving averages
ANALYSIS_WINDOWS = {
    'smoothing': 10,
    'convergence': 50,
    'stability': 100
}


# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

def get_device():
    """Get computation device based on availability and settings."""
    if not USE_CUDA:
        return 'cpu'

    try:
        import torch as th
        if th.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    except ImportError:
        return 'cpu'

DEVICE = get_device()


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_config_name(config_name: str) -> bool:
    """Validate that a configuration name exists."""
    return config_name in WEIGHT_CONFIGURATIONS


def validate_weights(weights: List[float]) -> bool:
    """Validate weight configuration."""
    if len(weights) != len(OBJECTIVE_NAMES):
        return False
    if any(w < 0 for w in weights):
        return False
    if abs(sum(weights)) < 1e-6:  # All zeros
        return False
    return True


def get_config(config_name: str) -> dict:
    """Get configuration by name with validation."""
    if not validate_config_name(config_name):
        available = list(WEIGHT_CONFIGURATIONS.keys())
        raise ValueError(f"Configuration '{config_name}' not found. Available: {available}")
    return WEIGHT_CONFIGURATIONS[config_name]


def list_configurations() -> List[str]:
    """Get list of available configuration names."""
    return list(WEIGHT_CONFIGURATIONS.keys())



# ============================================================================
# EXPERIMENT TEMPLATES
# ============================================================================

# Quick test configuration
QUICK_TEST_CONFIG = {
    'timesteps': 10000,
    'seeds': [42],
    'configs': ['balanced'],
    'save_models': False,
    'run_analysis': False
}

# Full experiment configuration
FULL_EXPERIMENT_CONFIG = {
    'timesteps': 100000,
    'seeds': [42, 123, 456],
    'configs': list(WEIGHT_CONFIGURATIONS.keys()),
    'save_models': True,
    'run_analysis': True
}

# Development configuration
DEV_CONFIG = {
    'timesteps': 5000,
    'seeds': [42],
    'configs': ['economic_only', 'balanced'],
    'save_models': False,
    'run_analysis': True,
    'verbose': True
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_log_directory(base_dir: str, experiment_name: str = None) -> Path:
    """Create logging directory structure."""
    if experiment_name:
        log_dir = Path(base_dir) / experiment_name
    else:
        log_dir = Path(base_dir)

    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_experiment_name(config_name: str = None, seed: int = None) -> str:
    """Generate experiment name from configuration and seed."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parts = []
    if config_name:
        parts.append(config_name)
    if seed:
        parts.append(f"seed{seed}")
    parts.append(timestamp)

    return "_".join(parts)


def merge_configs(base_config: dict, override_config: dict) -> dict:
    """Merge configuration dictionaries."""
    merged = base_config.copy()
    merged.update(override_config)
    return merged


# ============================================================================
# CONSTANTS
# ============================================================================

# Version information
VERSION = "1.0.0"
SCRIPT_NAME = "Unified MORL Training System"

# File extensions
LOG_EXTENSIONS = {
    'config': '.json',
    'results': '.json',
    'plots': '.png',
    'reports': '.md',
    'models': '.zip',
    'logs': '.log'
}

# Default seeds for reproducibility
DEFAULT_SEEDS = [42, 123, 456, 789, 999]

# Maximum values for safety
MAX_TIMESTEPS = 10_000_000
MAX_EPISODES = 100_000
MAX_EVAL_EPISODES = 1000

# Minimum values for meaningful training
MIN_TIMESTEPS = 1000
MIN_BUFFER_SIZE = 1000
MIN_LEARNING_STARTS = 100

if __name__ == "__main__":
    # Configuration validation and testing
    print("🔧 MORL Configuration Module")
    print("=" * 50)
    print(f"Available configurations: {len(WEIGHT_CONFIGURATIONS)}")
    for name in WEIGHT_CONFIGURATIONS:
        config = WEIGHT_CONFIGURATIONS[name]
        print(f"  {name}: {config['weights']}")

    print(f"\nDevice: {DEVICE}")
    print(f"Project root: {project_root}")
    print(f"MORL modules: {morl_modules_dir}")
    print("Configuration module loaded successfully!")