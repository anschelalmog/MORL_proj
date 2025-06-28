# MORL_modules/configs/baseline_morl_config.py

import torch.nn as nn
import numpy as np
from typing import Dict, List, Any
from pathlib import Path


class BaselineMORLConfig:
    """Configuration class for Baseline MORL experiments."""

    # Environment Configuration
    ENV_CONFIG = {
        "controller_name": "EnergyNetController",
        "controller_module": "energy_net.controllers",
        "env_config_path": "configs/environment_config.yaml",
        "iso_config_path": "configs/iso_config.yaml",
        "pcs_unit_config_path": "configs/pcs_unit_config.yaml",
        "cost_type": "CONSTANT",
        "pricing_policy": "QUADRATIC",
        "demand_pattern": "SINUSOIDAL"
    }

    # Training Configuration
    TRAINING_CONFIG = {
        "total_timesteps": 100000,
        "seed": 42,
        "verbose": 1,
        "n_eval_episodes": 20,
        "eval_freq": 10000,
        "save_freq": 25000
    }

    # Algorithm Configurations
    ALGORITHM_CONFIGS = {
        "SAC": {
            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "learning_starts": 1000,
            "target_update_interval": 1,
            "policy_kwargs": {
                "net_arch": [256, 256],
                "activation_fn": nn.ReLU
            }
        },

        "PPO": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "n_epochs": 10,
            "policy_kwargs": {
                "net_arch": [256, 256],
                "activation_fn": nn.Tanh
            }
        },

        "TD3": {
            "learning_rate": 1e-3,
            "buffer_size": 100000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": 1,
            "gradient_steps": 1,
            "learning_starts": 10000,
            "policy_delay": 2,
            "target_policy_noise": 0.2,
            "target_noise_clip": 0.5,
            "policy_kwargs": {
                "net_arch": [256, 256],
                "activation_fn": nn.ReLU
            }
        }
    }

    # Weight Vectors for Different Preferences
    WEIGHT_VECTORS = {
        # Single objective focus
        "economic_only": [1.0, 0.0, 0.0, 0.0],
        "battery_only": [0.0, 1.0, 0.0, 0.0],
        "grid_only": [0.0, 0.0, 1.0, 0.0],
        "autonomy_only": [0.0, 0.0, 0.0, 1.0],
        "balanced": [0.25, 0.25, 0.25, 0.25],
        "economic_balanced": [0.4, 0.2, 0.2, 0.2],
    }

    # Results and Logging Configuration
    RESULTS_CONFIG = {
        "base_results_dir": "MORL_modules/results",
        "models_subdir": "models",
        "logs_subdir": "logs",
        "figures_subdir": "figures",
        "data_subdir": "data",
        "save_formats": ["png", "pdf"],
        "figure_dpi": 600,
        "figure_size": (12, 8)
    }

    # Plotting Configuration
    PLOT_CONFIG = {
        "colors": {
            "economic": "#1f77b4",  # Blue
            "battery_health": "#ff7f0e",  # Orange
            "grid_support": "#2ca02c",  # Green
            "autonomy": "#d62728"  # Red
        },
        "objective_names": {
            "economic": "Economic Profit",
            "battery_health": "Battery Health",
            "grid_support": "Grid Support",
            "autonomy": "Energy Autonomy"
        },
        "linestyles": {
            "economic": "-",
            "battery_health": "--",
            "grid_support": "-.",
            "autonomy": ":"
        },
        "markers": {
            "economic": "o",
            "battery_health": "s",
            "grid_support": "^",
            "autonomy": "D"
        }
    }

    # Evaluation Configuration
    EVALUATION_CONFIG = {
        "n_eval_episodes": 20,
        "deterministic": True,
        "render_mode": None,
        "record_video": False,
        "compute_pareto_front": True,
        "compute_hypervolume": True,
        "reference_point": [-1.0, -1.0, -1.0, -1.0],  # For hypervolume calculation
        "normalize_objectives": True
    }

    @classmethod
    def get_weight_vectors_list(cls) -> List[List[float]]:
        """Get weight vectors as a list for easy iteration."""
        return list(cls.WEIGHT_VECTORS.values())

    @classmethod
    def get_weight_vector_names(cls) -> List[str]:
        """Get names of weight vectors."""
        return list(cls.WEIGHT_VECTORS.keys())

    @classmethod
    def get_algorithm_config(cls, algorithm: str) -> Dict[str, Any]:
        """Get configuration for specific algorithm."""
        if algorithm not in cls.ALGORITHM_CONFIGS:
            raise ValueError(f"Algorithm {algorithm} not found in configs")
        return cls.ALGORITHM_CONFIGS[algorithm].copy()

    @classmethod
    def create_results_structure(cls, base_dir: str = None) -> Dict[str, Path]:
        """Create results directory structure."""
        if base_dir is None:
            base_dir = cls.RESULTS_CONFIG["base_results_dir"]

        base_path = Path(base_dir)
        paths = {
            "base": base_path,
            "models": base_path / cls.RESULTS_CONFIG["models_subdir"],
            "logs": base_path / cls.RESULTS_CONFIG["logs_subdir"],
            "figures": base_path / cls.RESULTS_CONFIG["figures_subdir"],
            "data": base_path / cls.RESULTS_CONFIG["data_subdir"]
        }

        # Create directories
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)

        return paths

    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration consistency."""
        # Check weight vectors sum to 1 (approximately)
        for name, weights in cls.WEIGHT_VECTORS.items():
            if len(weights) != 4:
                raise ValueError(f"Weight vector {name} must have 4 elements")
            if not np.isclose(sum(weights), 1.0, atol=1e-6):
                print(f"Warning: Weight vector {name} does not sum to 1.0")

        # Check all algorithms have required parameters
        required_params = {
            "SAC": ["learning_rate", "buffer_size", "batch_size"],
            "PPO": ["learning_rate", "n_steps", "batch_size"],
            "TD3": ["learning_rate", "buffer_size", "batch_size"]
        }

        for alg, config in cls.ALGORITHM_CONFIGS.items():
            for param in required_params.get(alg, []):
                if param not in config:
                    raise ValueError(f"Algorithm {alg} missing required parameter: {param}")

        return True


DEFAULT_CONFIG = BaselineMORLConfig()

DEFAULT_CONFIG.validate_config()