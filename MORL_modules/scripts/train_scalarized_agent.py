#!/usr/bin/env python3
"""
Scalarized MOSAC Training Script

Trains MOSAC with 3 specific weight configurations:
1. economic_only: [1.0, 0.0, 0.0, 0.0] - Pure economic optimization
2. balanced: [0.25, 0.25, 0.25, 0.25] - Equal weight to all objectives
3. economic_battery: [0.5, 0.5, 0.0, 0.0] - Economic + Battery focus

Usage:
    python scripts/train_scalarize.py --config economic_only --timesteps 50000
    python scripts/train_scalarize.py --config balanced --timesteps 100000
    python scripts/train_scalarize.py --config economic_battery --timesteps 50000
"""

import os
import sys
import argparse
import numpy as np
import torch as th
from datetime import datetime
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
morl_modules_root = os.path.dirname(current_dir)

sys.path.append(project_root)
sys.path.append(morl_modules_root)

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from energy_net.envs.energy_net_v0 import EnergyNetV0
from energy_net.market.pricing.cost_types import CostType
from energy_net.market.pricing.pricing_policy import PricingPolicy
from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern

# Your existing wrappers
try:
    from wrappers.scalarized_mo_pcs_wrapper import ScalarizedMOPCSWrapper

    WRAPPER_AVAILABLE = True
    print("Using Scalarized MOPCSWrapper")
except ImportError as e:
    print(f"  Scalarized MOPCSWrapper not available: {e}")
    print("   Will use base environment (single objective)")
    WRAPPER_AVAILABLE = False

try:
    from agents.mosac import MOSAC

    MOSAC_AVAILABLE = True
    print("Using MOSAC from agents.mosac")
except ImportError:
    try:
        from agents.mosac2.mosac import MOSAC

        MOSAC_AVAILABLE = True
        print("Using MOSAC from agents.mosac2")
    except ImportError as e:
        print(f"MOSAC not available: {e}")
        print("Will use standard SAC")
        MOSAC_AVAILABLE = False

# ============================================================================
# CONFIGURATION DEFINITIONS
# ============================================================================

WEIGHT_CONFIGURATIONS = {
    'economic_only': {
        'name': 'economic_only',
        'weights': [1.0, 0.0, 0.0, 0.0],
        'expected_behavior': 'High profits, aggressive trading, potential battery stress'
    },

    'balanced': {
        'name': 'balanced',
        'weights': [0.25, 0.25, 0.25, 0.25],
        'expected_behavior': 'Moderate performance across all objectives'
    },

    'economic_battery': {
        'name': 'economic_battery',
        'weights': [0.5, 0.5, 0.0, 0.0],
        'expected_behavior': 'Good profits while preserving battery health'
    }
}


def get_config(config_name: str) -> dict:
    if config_name not in WEIGHT_CONFIGURATIONS:
        available = list(WEIGHT_CONFIGURATIONS.keys())
        raise ValueError(f"Configuration '{config_name}' not found. Available: {available}")
    return WEIGHT_CONFIGURATIONS[config_name]


def list_configurations():
    """Print all available configurations."""
    print("Available Weight Configurations:")
    print("=" * 50)

    for config_name, config in WEIGHT_CONFIGURATIONS.items():
        weights = config['weights']
        weights_str = '[' + ', '.join([f'{w:.2f}' for w in weights]) + ']'
        print(f"  {config_name:<15} {weights_str}")
        print(f"    Description: {config['description']}")
        print(f"    Expected: {config['expected_behavior']}")
        print()


# ============================================================================
# ENVIRONMENT CREATION
# ============================================================================

def create_base_environment(seed: int = 42) -> EnergyNetV0:
    try:
        env = EnergyNetV0(
            controller_name="EnergyNetController",
            controller_module="energy_net.controllers",
            env_config_path='energy_net/configs/environment_config.yaml',
            iso_config_path='energy_net/configs/iso_config.yaml',
            pcs_unit_config_path='energy_net/configs/pcs_unit_config.yaml',
            cost_type=CostType.CONSTANT,
            pricing_policy=PricingPolicy.QUADRATIC,
            demand_pattern=DemandPattern.SINUSOIDAL,
        )

        env.seed(seed)
        print(" EnergyNet environment created successfully")
        return env

    except Exception as e:
        print(f" Failed to create EnergyNet environment: {e}")
        raise


def create_scalarized_environment(weights: list, seed: int = 42):
    """Create environment with scalarized wrapper."""

    # Create base environment
    base_env = create_base_environment(seed)

    if not WRAPPER_AVAILABLE:
        print(" Using base environment without scalarization")
        return base_env

    try:
        # Apply scalarized wrapper
        scalarized_env = ScalarizedMOPCSWrapper(
            base_env,
            weights=weights,
            normalize_weights=True,
            log_level='INFO'
        )

        print(f"✅ Scalarized environment created with weights: {weights}")
        return scalarized_env

    except Exception as e:
        print(f" Failed to create scalarized wrapper: {e}")
        print("   Falling back to base environment")
        return base_env


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def create_agent(env, config: dict, learning_rate: float = 3e-4, **kwargs):
    """Create the RL agent (MOSAC or SAC)."""

    # Default parameters optimized for EnergyNet
    default_params = {
        'learning_rate': learning_rate,
        'buffer_size': 100000,
        'learning_starts': 1000,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'verbose': 1,
        'device': 'cuda' if th.cuda.is_available() else 'cpu',
        'policy_kwargs': {
            'net_arch': {'pi': [256, 256], 'qf': [256, 256]},
        }
    }

    # Update with any provided kwargs
    default_params.update(kwargs)

    if MOSAC_AVAILABLE:
        try:
            model = MOSAC(policy="MlpPolicy", env=env, **default_params)
            print(" MOSAC agent created")
            return model
        except Exception as e:
            print(f"  MOSAC creation failed: {e}")
            print("   Falling back to SAC")

    model = SAC(policy="MlpPolicy", env=env, **default_params)
    print(" SAC agent created")
    return model


def train_configuration(config_name: str,
                        timesteps: int = 50000,
                        seed: int = 42,
                        log_dir: str = 'logs/scalarized_training',
                        **training_kwargs) -> dict:
    """
    Train a single configuration.

    Args:
        config_name: Name of configuration to train
        timesteps: Number of training timesteps
        seed: Random seed
        log_dir: Base logging directory
        **training_kwargs: Additional training parameters

    Returns:
        Dictionary with training results
    """

    print("=" * 60)
    print(f" TRAINING CONFIGURATION: {config_name.upper()}")
    print("=" * 60)

    # Get configuration
    config = get_config(config_name)
    weights = config['weights']

    print(f"Configuration: {config['description']}")
    print(f"Weights: {weights}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Seed: {seed}")
    print(f"Expected behavior: {config['expected_behavior']}")
    print("=" * 60)

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_log_dir = os.path.join(log_dir, f"{config_name}_{timestamp}")
    os.makedirs(exp_log_dir, exist_ok=True)

    # Save configuration
    config_info = {
        'config_name': config_name,
        'config': config,
        'weights': weights,
        'timesteps': timesteps,
        'seed': seed,
        'timestamp': timestamp,
        'training_kwargs': training_kwargs
    }

    with open(os.path.join(exp_log_dir, 'config.json'), 'w') as f:
        json.dump(config_info, f, indent=2)

    try:
        # Create environment
        print(" Creating environment...")
        env = create_scalarized_environment(weights, seed)
        env = Monitor(env, os.path.join(exp_log_dir, 'training.monitor.csv'))

        # Create agent
        print(" Creating agent...")
        model = create_agent(
            env,
            config,
            tensorboard_log=os.path.join(exp_log_dir, 'tensorboard'),
            **training_kwargs
        )

        # Training
        print(f" Starting training for {timesteps:,} timesteps...")
        start_time = datetime.now()

        model.learn(
            total_timesteps=timesteps,
            log_interval=100,
            progress_bar=True
        )

        training_duration = (datetime.now() - start_time).total_seconds()

        # Save model
        model_path = os.path.join(exp_log_dir, 'final_model')
        model.save(model_path)

        print(f" Training completed in {training_duration:.1f} seconds")
        print(f" Results saved to: {exp_log_dir}")

        # Quick evaluation
        print(" Running quick evaluation...")
        eval_results = quick_evaluation(model, env, n_episodes=10)

        # Save evaluation results
        with open(os.path.join(exp_log_dir, 'eval_results.json'), 'w') as f:
            json.dump(eval_results, f, indent=2)

        return {
            'status': 'success',
            'config_name': config_name,
            'weights': weights,
            'timesteps': timesteps,
            'duration': training_duration,
            'log_dir': exp_log_dir,
            'eval_results': eval_results
        }

    except KeyboardInterrupt:
        print("\n Training interrupted by user")
        # Save interrupted model
        if 'model' in locals():
            model.save(os.path.join(exp_log_dir, 'interrupted_model'))

        return {
            'status': 'interrupted',
            'config_name': config_name,
            'log_dir': exp_log_dir
        }

    except Exception as e:
        print(f" Training failed: {e}")
        import traceback
        traceback.print_exc()

        # Save error info
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

        with open(os.path.join(exp_log_dir, 'error.json'), 'w') as f:
            json.dump(error_info, f, indent=2)

        return {
            'status': 'failed',
            'config_name': config_name,
            'error': str(e),
            'log_dir': exp_log_dir
        }


def quick_evaluation(model, env, n_episodes: int = 10) -> dict:
    """Run quick evaluation of trained model."""

    print(f"   Running {n_episodes} evaluation episodes...")

    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        total_reward = 0
        episode_length = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            episode_length += 1

            # Safety limit
            if episode_length > 1000:
                break

        episode_rewards.append(float(total_reward))
        episode_lengths.append(episode_length)

    eval_results = {
        'n_episodes': n_episodes,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }

    print(f"    Mean reward: {eval_results['mean_reward']:.4f} ± {eval_results['std_reward']:.4f}")
    print(f"    Mean episode length: {eval_results['mean_length']:.1f}")

    return eval_results


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main training function with command line interface."""

    parser = argparse.ArgumentParser(
        description='Train MOSAC with Scalarized Multi-Objective Rewards',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_scalarize.py --config economic_only --timesteps 50000
  python scripts/train_scalarize.py --config balanced --timesteps 100000  
  python scripts/train_scalarize.py --config economic_battery --timesteps 75000
  python scripts/train_scalarize.py --list-configs  # Show available configurations
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        choices=list(WEIGHT_CONFIGURATIONS.keys()),
        help='Configuration to train'
    )

    parser.add_argument(
        '--timesteps',
        type=int,
        default=50000,
        help='Number of training timesteps (default: 50000)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs/scalarized_training',
        help='Base logging directory (default: logs/scalarized_training)'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=3e-4,
        help='Learning rate (default: 3e-4)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size (default: 256)'
    )

    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List all available configurations and exit'
    )

    args = parser.parse_args()

    # Show configurations if requested
    if args.list_configs:
        list_configurations()
        return

    # Validate arguments
    if not args.config:
        print(" Error: --config is required")
        print("Available configurations:")
        for config_name in WEIGHT_CONFIGURATIONS.keys():
            print(f"  - {config_name}")
        print("\nUse --list-configs for detailed information")
        sys.exit(1)

    # Show system information
    print("🔧 SYSTEM INFORMATION")
    print("=" * 50)
    print(f"PyTorch version: {th.__version__}")
    print(f"CUDA available: {th.cuda.is_available()}")
    if th.cuda.is_available():
        print(f"CUDA device: {th.cuda.get_device_name()}")
    print(f"Wrapper available: {WRAPPER_AVAILABLE}")
    print(f"MOSAC available: {MOSAC_AVAILABLE}")
    print()

    # Training parameters
    training_kwargs = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
    }

    # Run training
    result = train_configuration(
        config_name=args.config,
        timesteps=args.timesteps,
        seed=args.seed,
        log_dir=args.log_dir,
        **training_kwargs
    )

    # Final summary
    print("\n" + "=" * 60)
    print(" TRAINING SUMMARY")
    print("=" * 60)
    print(f"Configuration: {result['config_name']}")
    print(f"Status: {result['status']}")

    if result['status'] == 'success':
        print(f"Duration: {result['duration']:.1f} seconds")
        print(f"Mean reward: {result['eval_results']['mean_reward']:.4f}")
        print(f"Results: {result['log_dir']}")
        print(" Training completed successfully!")
    else:
        print(f" Training {result['status']}")

    print("=" * 60)


if __name__ == "__main__":
    main()