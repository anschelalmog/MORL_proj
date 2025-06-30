# MORL_modules/agents/mosac/__init__.py

"""
Multi-Objective Soft Actor-Critic (MOSAC) Implementation

This module provides a complete implementation of Multi-Objective SAC for the EnergyNet environment.
The implementation supports both scalarized and true multi-objective training approaches.

Key Components:
- MOSAC: Main algorithm class extending Stable-Baselines3 SAC
- MOSACPolicy: Policy network with actor and multiple critics
- MOSACReplayBuffer: Replay buffer for vector rewards
- Specialized Critics: Domain-specific critic networks for each objective
- Utilities: Analysis tools, Pareto archive, and metrics

Usage Examples:

Basic MOSAC training:
```python
from MORL_modules.agents.mosac import MOSAC, MOSACPolicy
from energy_net import EnergyNetV0
from wrappers import MOPCSWrapper

# Create environment
env = EnergyNetV0()
env = MOPCSWrapper(env, num_objectives=4)

# Create model
model = MOSAC(
    policy=MOSACPolicy,
    env=env,
    num_objectives=4,
    mo_strategy="scalarized",  # or "pareto"
    preference_weights=[0.4, 0.2, 0.2, 0.2],  # Economic focus
)

# Train
model.learn(total_timesteps=100_000)
```

True Multi-Objective training:
```python
model = MOSAC(
    policy=MOSACPolicy,
    env=env,
    num_objectives=4,
    mo_strategy="pareto",
    pareto_archive_size=100,
)

# Train and analyze Pareto front
model.learn(total_timesteps=100_000)
pareto_solutions = model.get_pareto_archive()
```

Advanced critic configuration:
```python
from MORL_modules.agents.mosac.critics import create_mosac_critic

# Use specialized critics
policy_kwargs = {
    "critic_type": "specialized",  # Domain-specific critics
    "num_objectives": 4,
}

model = MOSAC(
    policy=MOSACPolicy,
    env=env,
    policy_kwargs=policy_kwargs,
)
```
"""

from .mosac import MOSAC
from .networks import (
    MOSACPolicy,
    MOSACActorNetwork,
    MOSACMultiCritic,
    MOSACCriticNetwork,
)
from .replay_buffer import MOSACReplayBuffer
from .critics import (
    create_mosac_critic,
    SpecializedMOSACMultiCritic,
    AdaptiveMOSACMultiCritic,
    HierarchicalMOSACCritic,
    CriticEnsemble,
    EconomicCritic,
    BatteryHealthCritic,
    GridSupportCritic,
    AutonomyCritic,
)
from .utils import (
    ParetoArchive,
    MOSACMetrics,
    MOSACAnalyzer,
    create_preference_weights,
    load_training_stats,
    save_training_stats,
)

# Version info
__version__ = "1.0.0"
__author__ = "MORL Research Team"

# Default configuration
DEFAULT_CONFIG = {
    "num_objectives": 4,
    "objective_names": ["economic", "battery_health", "grid_support", "autonomy"],
    "mo_strategy": "scalarized",  # "scalarized" or "pareto"
    "preference_weights": [0.25, 0.25, 0.25, 0.25],  # Equal weights
    "learning_rate": 3e-4,
    "batch_size": 256,
    "buffer_size": 1_000_000,
    "gamma": 0.99,
    "tau": 0.005,
    "actor_hidden_dims": [256, 256],
    "critic_hidden_dims": [256, 256],
    "critic_type": "standard",  # "standard", "specialized", "adaptive", "hierarchical"
    "pareto_archive_size": 100,
}

# Export all public components
__all__ = [
    # Main algorithm
    "MOSAC",

    # Networks
    "MOSACPolicy",
    "MOSACActorNetwork",
    "MOSACMultiCritic",
    "MOSACCriticNetwork",

    # Replay buffer
    "MOSACReplayBuffer",

    # Specialized critics
    "create_mosac_critic",
    "SpecializedMOSACMultiCritic",
    "AdaptiveMOSACMultiCritic",
    "HierarchicalMOSACCritic",
    "CriticEnsemble",
    "EconomicCritic",
    "BatteryHealthCritic",
    "GridSupportCritic",
    "AutonomyCritic",

    # Utilities
    "ParetoArchive",
    "MOSACMetrics",
    "MOSACAnalyzer",
    "create_preference_weights",
    "load_training_stats",
    "save_training_stats",

    # Configuration
    "DEFAULT_CONFIG",
]


def create_mosac_model(
        env,
        config: dict = None,
        **kwargs
) -> MOSAC:
    """
    Convenience function to create a MOSAC model with default configuration

    Args:
        env: Environment (should be wrapped with MOPCSWrapper)
        config: Configuration dictionary (overrides defaults)
        **kwargs: Additional arguments passed to MOSAC

    Returns:
        Configured MOSAC model
    """
    # Merge configurations
    model_config = DEFAULT_CONFIG.copy()
    if config:
        model_config.update(config)
    model_config.update(kwargs)

    # Extract policy kwargs
    policy_kwargs = {
        "num_objectives": model_config["num_objectives"],
        "actor_hidden_dims": model_config["actor_hidden_dims"],
        "critic_hidden_dims": model_config["critic_hidden_dims"],
    }

    # Create model
    model = MOSAC(
        policy=MOSACPolicy,
        env=env,
        learning_rate=model_config["learning_rate"],
        batch_size=model_config["batch_size"],
        buffer_size=model_config["buffer_size"],
        gamma=model_config["gamma"],
        tau=model_config["tau"],
        num_objectives=model_config["num_objectives"],
        preference_weights=model_config["preference_weights"],
        mo_strategy=model_config["mo_strategy"],
        pareto_archive_size=model_config["pareto_archive_size"],
        policy_kwargs=policy_kwargs,
    )

    return model


def get_objective_weights(strategy: str = "balanced", **kwargs) -> list:
    """
    Get predefined objective weight configurations

    Args:
        strategy: Weight strategy name
        **kwargs: Additional arguments for specific strategies

    Returns:
        List of preference weights [economic, battery_health, grid_support, autonomy]
    """
    strategies = {
        "balanced": [0.25, 0.25, 0.25, 0.25],
        "economic_focus": [0.5, 0.2, 0.15, 0.15],
        "battery_focus": [0.2, 0.5, 0.15, 0.15],
        "grid_focus": [0.2, 0.15, 0.5, 0.15],
        "autonomy_focus": [0.2, 0.15, 0.15, 0.5],
        "economic_battery": [0.4, 0.4, 0.1, 0.1],
        "grid_autonomy": [0.1, 0.1, 0.4, 0.4],
    }

    if strategy in strategies:
        return strategies[strategy]
    elif strategy == "custom":
        weights = kwargs.get("weights", [0.25, 0.25, 0.25, 0.25])
        return list(weights)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategies.keys())}")


# Compatibility aliases for easier migration
MOSACActor = MOSACActorNetwork
MOSACCritic = MOSACMultiCritic
MOBuffer = MOSACReplayBuffer

# Register default configurations for different use cases
PRESET_CONFIGS = {
    "quick_test": {
        "buffer_size": 10_000,
        "batch_size": 64,
        "actor_hidden_dims": [128, 128],
        "critic_hidden_dims": [128, 128],
    },

    "standard_training": DEFAULT_CONFIG,

    "high_performance": {
        "buffer_size": 2_000_000,
        "batch_size": 512,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "learning_rate": 1e-4,
    },

    "pareto_exploration": {
        "mo_strategy": "pareto",
        "pareto_archive_size": 200,
        "batch_size": 512,
    },
}


def get_preset_config(name: str) -> dict:
    """
    Get a preset configuration

    Args:
        name: Preset name

    Returns:
        Configuration dictionary
    """
    if name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESET_CONFIGS.keys())}")

    return PRESET_CONFIGS[name].copy()


# Module-level convenience functions
def quick_mosac(env, strategy: str = "balanced", **kwargs) -> MOSAC:
    """Quick setup for MOSAC with sensible defaults"""
    config = get_preset_config("standard_training")
    config["preference_weights"] = get_objective_weights(strategy)
    config.update(kwargs)

    return create_mosac_model(env, config)


def pareto_mosac(env, archive_size: int = 100, **kwargs) -> MOSAC:
    """Quick setup for Pareto-front discovery"""
    config = get_preset_config("pareto_exploration")
    config["pareto_archive_size"] = archive_size
    config.update(kwargs)

    return create_mosac_model(env, config)