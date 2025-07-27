#!/usr/bin/env python3
"""
MORL Core Training Module

Core functionality for Multi-Objective Reinforcement Learning training
including environment creation, agent initialization, and training logic.
"""

import os
import sys
import time
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np

# Import configuration
from .settings import (
    WEIGHT_CONFIGURATIONS, OBJECTIVE_NAMES, DEFAULT_TRAINING_PARAMS,
    MOSAC_PARAMS, SAC_PARAMS, ENVIRONMENT_PARAMS, QUICK_EVAL_PARAMS,
    USE_MOSAC, USE_SCALARIZED_WRAPPER, USE_MONITOR_WRAPPER, USE_TENSORBOARD,
    SAVE_MODELS, RUN_QUICK_EVAL, VERBOSE_LOGGING, DEVICE, GRACEFUL_FALLBACKS,
    DEPENDENCY_STATUS, get_config, validate_config_name
)

# ============================================================================
# DEPENDENCY IMPORTS WITH FALLBACKS
# ============================================================================

# Core imports
try:
    import torch as th

    DEPENDENCY_STATUS['torch'] = True
    print("✅ PyTorch available") if VERBOSE_LOGGING else None
except ImportError:
    print("❌ PyTorch not available")
    sys.exit(1)

# Stable Baselines imports
try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    DEPENDENCY_STATUS['stable_baselines3'] = True
    print("✅ Stable-Baselines3 available") if VERBOSE_LOGGING else None
except ImportError:
    print("❌ Stable-Baselines3 not available")
    sys.exit(1)

# EnergyNet imports
try:
    from energy_net.envs.energy_net_v0 import EnergyNetV0
    from energy_net.market.pricing.cost_types import CostType
    from energy_net.market.pricing.pricing_policy import PricingPolicy
    from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern

    DEPENDENCY_STATUS['energy_net'] = True
    print("✅ EnergyNet available") if VERBOSE_LOGGING else None
except ImportError as e:
    print(f"❌ EnergyNet not available: {e}")
    sys.exit(1)

# MOSAC imports (with fallback)
if USE_MOSAC:
    try:
        from agents.mosac import MOSAC, MOSACPolicy
        from agents.mobuffers import MOReplayBuffer

        DEPENDENCY_STATUS['mosac'] = True
        print("✅ MOSAC available") if VERBOSE_LOGGING else None
    except ImportError:
        try:
            from agents.mosac2.mosac import MOSAC, MOSACPolicy
            from agents.mobuffers import MOReplayBuffer

            DEPENDENCY_STATUS['mosac'] = True
            print("✅ MOSAC available (mosac2)") if VERBOSE_LOGGING else None
        except ImportError:
            print("⚠️ MOSAC not available, will fallback to SAC") if VERBOSE_LOGGING else None
            DEPENDENCY_STATUS['mosac'] = False

# Wrapper imports
if USE_SCALARIZED_WRAPPER:
    try:
        from wrappers.scalarized_mo_pcs_wrapper import ScalarizedMOPCSWrapper

        DEPENDENCY_STATUS['scalarized_wrapper'] = True
        print("✅ ScalarizedMOPCSWrapper available") if VERBOSE_LOGGING else None
    except ImportError:
        print("⚠️ ScalarizedMOPCSWrapper not available") if VERBOSE_LOGGING else None
        DEPENDENCY_STATUS['scalarized_wrapper'] = False


# ============================================================================
# ENVIRONMENT CREATION
# ============================================================================

def create_base_environment(seed: int = 42, **env_kwargs) -> EnergyNetV0:
    """
    Create base EnergyNet environment.

    Args:
        seed: Random seed for environment
        **env_kwargs: Additional environment parameters

    Returns:
        EnergyNetV0 environment instance
    """
    try:
        # Merge default parameters with overrides
        params = ENVIRONMENT_PARAMS.copy()
        params.update(env_kwargs)

        # Convert string enums to actual enum values
        cost_type = getattr(CostType, params['cost_type'])
        pricing_policy = getattr(PricingPolicy, params['pricing_policy'])
        demand_pattern = getattr(DemandPattern, params['demand_pattern'])

        env = EnergyNetV0(
            controller_name=params['controller_name'],
            controller_module=params['controller_module'],
            env_config_path=params['env_config_path'],
            iso_config_path=params['iso_config_path'],
            pcs_unit_config_path=params['pcs_unit_config_path'],
            cost_type=cost_type,
            pricing_policy=pricing_policy,
            demand_pattern=demand_pattern,
        )

        env.seed(seed)
        print(f"✅ EnergyNet environment created (seed={seed})") if VERBOSE_LOGGING else None
        return env

    except Exception as e:
        print(f"❌ Failed to create EnergyNet environment: {e}")
        if VERBOSE_LOGGING:
            traceback.print_exc()
        raise


def create_wrapped_environment(weights: List[float], seed: int = 42, **env_kwargs):
    """
    Create environment with optional scalarized wrapper.

    Args:
        weights: Objective weights for scalarization
        seed: Random seed for environment
        **env_kwargs: Additional environment parameters

    Returns:
        Wrapped environment instance
    """
    base_env = create_base_environment(seed, **env_kwargs)

    # Check if wrapper is available and enabled
    if not DEPENDENCY_STATUS['scalarized_wrapper'] or not USE_SCALARIZED_WRAPPER:
        print("⚠️ Using base environment without scalarization") if VERBOSE_LOGGING else None
        return base_env

    try:
        scalarized_env = ScalarizedMOPCSWrapper(
            base_env,
            weights=weights,
            normalize_weights=True,
            log_level='INFO' if VERBOSE_LOGGING else 'WARNING'
        )
        print(f"✅ Scalarized environment created with weights: {weights}") if VERBOSE_LOGGING else None
        return scalarized_env

    except Exception as e:
        print(f"⚠️ Failed to create scalarized wrapper: {e}")
        if GRACEFUL_FALLBACKS:
            print("   Falling back to base environment")
            return base_env
        else:
            raise


def create_monitored_environment(env, log_dir: str = None):
    """
    Wrap environment with Monitor for episode tracking.

    Args:
        env: Environment to wrap
        log_dir: Directory for monitor logs

    Returns:
        Monitored environment
    """
    if not USE_MONITOR_WRAPPER:
        return env

    if log_dir is None:
        return env

    try:
        monitor_file = os.path.join(log_dir, 'training.monitor.csv')
        monitored_env = Monitor(env, monitor_file)
        print(f"✅ Monitor wrapper added: {monitor_file}") if VERBOSE_LOGGING else None
        return monitored_env

    except Exception as e:
        print(f"⚠️ Failed to add monitor wrapper: {e}")
        if GRACEFUL_FALLBACKS:
            return env
        else:
            raise


# ============================================================================
# AGENT CREATION
# ============================================================================

def create_agent(env, config: dict, log_dir: str = None, **agent_kwargs):
    """
    Create RL agent (MOSAC or SAC with fallback).

    Args:
        env: Training environment
        config: Configuration dictionary
        log_dir: Directory for logging
        **agent_kwargs: Additional agent parameters

    Returns:
        Trained agent instance
    """
    # Merge default parameters with overrides
    params = DEFAULT_TRAINING_PARAMS.copy()
    params.update(agent_kwargs)

    # Common parameters
    common_params = {
        'learning_rate': params['learning_rate'],
        'buffer_size': params['buffer_size'],
        'learning_starts': params['learning_starts'],
        'batch_size': params['batch_size'],
        'tau': params['tau'],
        'gamma': params['gamma'],
        'train_freq': params['train_freq'],
        'gradient_steps': params['gradient_steps'],
        'verbose': 1 if VERBOSE_LOGGING else 0,
        'device': DEVICE,
        'tensorboard_log': os.path.join(log_dir, 'tensorboard') if (log_dir and USE_TENSORBOARD) else None
    }

    # Try MOSAC first if available and enabled
    if DEPENDENCY_STATUS['mosac'] and USE_MOSAC:
        try:
            mosac_params = common_params.copy()
            mosac_params.update(MOSAC_PARAMS)

            # Add MOSAC-specific parameters
            if DEPENDENCY_STATUS['scalarized_wrapper']:
                mosac_params['replay_buffer_class'] = MOReplayBuffer

            model = MOSAC(policy="MlpPolicy", env=env, **mosac_params)
            print("✅ MOSAC agent created") if VERBOSE_LOGGING else None
            return model

        except Exception as e:
            print(f"⚠️ MOSAC creation failed: {e}")
            if not GRACEFUL_FALLBACKS:
                raise
            print("   Falling back to SAC")

    # Fallback to SAC
    try:
        sac_params = common_params.copy()
        sac_params.update(SAC_PARAMS)

        model = SAC(policy="MlpPolicy", env=env, **sac_params)
        print("✅ SAC agent created") if VERBOSE_LOGGING else None
        return model

    except Exception as e:
        print(f"❌ SAC creation failed: {e}")
        if VERBOSE_LOGGING:
            traceback.print_exc()
        raise


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_single_configuration(
        config_name: str,
        timesteps: int = None,
        seed: int = 42,
        log_dir: str = 'logs/morl_training',
        **kwargs
) -> Dict[str, Any]:
    """
    Train a single configuration.

    Args:
        config_name: Name of weight configuration to train
        timesteps: Number of training timesteps
        seed: Random seed
        log_dir: Base logging directory
        **kwargs: Additional training parameters

    Returns:
        Dictionary with training results
    """
    print("=" * 60)
    print(f"🚀 TRAINING CONFIGURATION: {config_name.upper()}")
    print("=" * 60)

    # Validate configuration
    if not validate_config_name(config_name):
        available = list(WEIGHT_CONFIGURATIONS.keys())
        raise ValueError(f"Configuration '{config_name}' not found. Available: {available}")

    config = get_config(config_name)
    weights = config['weights']

    # Use default timesteps if not specified
    if timesteps is None:
        timesteps = DEFAULT_TRAINING_PARAMS['timesteps']

    print(f"Configuration: {config['description']}")
    print(f"Weights: {weights}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Seed: {seed}")
    print(f"Expected behavior: {config['expected_behavior']}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_log_dir = os.path.join(log_dir, f"{config_name}_{timestamp}")
    os.makedirs(exp_log_dir, exist_ok=True)

    # Save configuration info
    config_info = {
        'config_name': config_name,
        'config': config,
        'weights': weights,
        'timesteps': timesteps,
        'seed': seed,
        'timestamp': timestamp,
        'training_kwargs': kwargs,
        'system_info': {
            'device': DEVICE,
            'dependency_status': DEPENDENCY_STATUS.copy()
        }
    }

    with open(os.path.join(exp_log_dir, 'config.json'), 'w') as f:
        json.dump(config_info, f, indent=2)

    try:
        # Create environment
        print("🌍 Creating environment...")
        env = create_wrapped_environment(weights, seed)
        env = create_monitored_environment(env, exp_log_dir)

        # Create agent
        print("🤖 Creating agent...")
        model = create_agent(env, config, exp_log_dir, **kwargs)

        # Training
        print(f"🏋️ Starting training for {timesteps:,} timesteps...")
        start_time = time.time()

        model.learn(
            total_timesteps=timesteps,
            log_interval=100 if VERBOSE_LOGGING else None,
            progress_bar=True
        )

        training_duration = time.time() - start_time

        # Save model
        if SAVE_MODELS:
            model_path = os.path.join(exp_log_dir, 'final_model')
            model.save(model_path)
            print(f"💾 Model saved to: {model_path}") if VERBOSE_LOGGING else None

        print(f"✅ Training completed in {training_duration:.1f} seconds")
        print(f"📁 Results saved to: {exp_log_dir}")

        # Quick evaluation
        eval_results = None
        if RUN_QUICK_EVAL:
            print("📊 Running quick evaluation...")
            eval_results = quick_evaluation(model, env)

            # Save evaluation results
            with open(os.path.join(exp_log_dir, 'eval_results.json'), 'w') as f:
                json.dump(eval_results, f, indent=2)

        return {
            'status': 'success',
            'config_name': config_name,
            'weights': weights,
            'timesteps': timesteps,
            'seed': seed,
            'duration': training_duration,
            'log_dir': exp_log_dir,
            'eval_results': eval_results,
            'model_path': os.path.join(exp_log_dir, 'final_model') if SAVE_MODELS else None
        }

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")

        # Save interrupted model if possible
        if SAVE_MODELS and 'model' in locals():
            try:
                interrupted_path = os.path.join(exp_log_dir, 'interrupted_model')
                model.save(interrupted_path)
                print(f"💾 Interrupted model saved to: {interrupted_path}")
            except Exception as save_e:
                print(f"⚠️ Could not save interrupted model: {save_e}")

        return {
            'status': 'interrupted',
            'config_name': config_name,
            'log_dir': exp_log_dir,
            'seed': seed
        }

    except Exception as e:
        print(f"❌ Training failed: {e}")
        if VERBOSE_LOGGING:
            traceback.print_exc()

        # Save error info
        error_info = {
            'config_name': config_name,
            'seed': seed,
            'timesteps': timesteps,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

        try:
            with open(os.path.join(exp_log_dir, 'error.json'), 'w') as f:
                json.dump(error_info, f, indent=2)
        except Exception:
            pass

        return {
            'status': 'failed',
            'config_name': config_name,
            'seed': seed,
            'error': str(e),
            'log_dir': exp_log_dir
        }


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def quick_evaluation(model, env, n_episodes: int = None) -> Dict[str, Any]:
    """
    Run quick evaluation of trained model.

    Args:
        model: Trained RL model
        env: Environment for evaluation
        n_episodes: Number of episodes to evaluate

    Returns:
        Dictionary with evaluation results
    """
    if n_episodes is None:
        n_episodes = QUICK_EVAL_PARAMS['n_episodes']

    max_episode_length = QUICK_EVAL_PARAMS['max_episode_length']
    deterministic = QUICK_EVAL_PARAMS['deterministic']

    print(f"   Running {n_episodes} evaluation episodes...")

    episode_rewards = []
    episode_lengths = []
    episode_info = []

    for episode in range(n_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        total_reward = 0
        episode_length = 0
        done = False
        episode_data = []

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            episode_length += 1

            # Store step data
            step_data = {
                'step': episode_length,
                'reward': float(reward),
                'action': action.tolist() if hasattr(action, 'tolist') else action
            }
            episode_data.append(step_data)

            # Safety limit
            if episode_length >= max_episode_length:
                break

        episode_rewards.append(float(total_reward))
        episode_lengths.append(episode_length)
        episode_info.append({
            'episode': episode,
            'total_reward': float(total_reward),
            'length': episode_length,
            'steps': episode_data
        })

    # Calculate statistics
    eval_results = {
        'n_episodes': n_episodes,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'median_reward': float(np.median(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'detailed_episodes': episode_info[:3]  # Store first 3 episodes in detail
    }

    print(f"    Mean reward: {eval_results['mean_reward']:.4f} ± {eval_results['std_reward']:.4f}")
    print(f"    Reward range: [{eval_results['min_reward']:.4f}, {eval_results['max_reward']:.4f}]")
    print(f"    Mean episode length: {eval_results['mean_length']:.1f}")

    return eval_results


def evaluate_model_from_path(model_path: str, config_name: str, n_episodes: int = 20) -> Dict[str, Any]:
    """
    Load and evaluate a saved model.

    Args:
        model_path: Path to saved model
        config_name: Configuration name for environment creation
        n_episodes: Number of evaluation episodes

    Returns:
        Dictionary with evaluation results
    """
    try:
        # Get configuration
        config = get_config(config_name)
        weights = config['weights']

        # Create environment
        env = create_wrapped_environment(weights, seed=42)

        # Load model
        if DEPENDENCY_STATUS['mosac'] and USE_MOSAC:
            model = MOSAC.load(model_path, env=env)
        else:
            model = SAC.load(model_path, env=env)

        print(f"✅ Model loaded from: {model_path}")

        # Evaluate
        eval_results = quick_evaluation(model, env, n_episodes)
        eval_results['model_path'] = model_path
        eval_results['config_name'] = config_name
        eval_results['weights'] = weights

        return eval_results

    except Exception as e:
        print(f"❌ Model evaluation failed: {e}")
        if VERBOSE_LOGGING:
            traceback.print_exc()
        raise


# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

def save_model_with_metadata(model, save_path: str, config_info: dict):
    """
    Save model with additional metadata.

    Args:
        model: Trained model to save
        save_path: Path to save model
        config_info: Configuration information
    """
    try:
        # Save model
        model.save(save_path)

        # Save metadata
        metadata = {
            'model_type': type(model).__name__,
            'save_timestamp': datetime.now().isoformat(),
            'config_info': config_info,
            'training_stats': {
                'total_timesteps': getattr(model, '_total_timesteps', 0),
                'num_timesteps': getattr(model, 'num_timesteps', 0),
                'n_updates': getattr(model, '_n_updates', 0)
            }
        }

        metadata_path = save_path + '_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Model and metadata saved to: {save_path}")

    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        if VERBOSE_LOGGING:
            traceback.print_exc()
        raise


def load_model_with_metadata(model_path: str, env=None) -> Tuple[Any, dict]:
    """
    Load model with metadata.

    Args:
        model_path: Path to saved model
        env: Environment for model loading

    Returns:
        Tuple of (model, metadata)
    """
    try:
        # Load metadata
        metadata_path = model_path + '_metadata.json'
        metadata = {}

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        # Determine model type and load
        model_type = metadata.get('model_type', 'SAC')

        if model_type == 'MOSAC' and DEPENDENCY_STATUS['mosac']:
            model = MOSAC.load(model_path, env=env)
        else:
            model = SAC.load(model_path, env=env)

        print(f"✅ Model loaded: {model_type} from {model_path}")
        return model, metadata

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        if VERBOSE_LOGGING:
            traceback.print_exc()
        raise


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_system_info():
    """Print system and dependency information."""
    print("🔧 SYSTEM INFORMATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    if DEPENDENCY_STATUS['torch']:
        print(f"PyTorch version: {th.__version__}")
        if th.cuda.is_available():
            print(f"CUDA device: {th.cuda.get_device_name()}")

    print("\n📦 DEPENDENCY STATUS:")
    for dep, status in DEPENDENCY_STATUS.items():
        status_icon = "✅" if status else "❌"
        print(f"  {dep}: {status_icon}")

    print("\n⚙️ CONFIGURATION STATUS:")
    print(f"  MOSAC enabled: {USE_MOSAC}")
    print(f"  Scalarized wrapper: {USE_SCALARIZED_WRAPPER}")
    print(f"  Monitor wrapper: {USE_MONITOR_WRAPPER}")
    print(f"  TensorBoard: {USE_TENSORBOARD}")
    print(f"  Model saving: {SAVE_MODELS}")
    print("=" * 60)


def check_training_readiness() -> bool:
    """
    Check if system is ready for training.

    Returns:
        True if ready, False otherwise
    """
    # Check essential dependencies
    essential_deps = ['torch', 'stable_baselines3', 'energy_net']

    for dep in essential_deps:
        if not DEPENDENCY_STATUS[dep]:
            print(f"❌ Essential dependency missing: {dep}")
            return False

    # Check optional dependencies with warnings
    if USE_MOSAC and not DEPENDENCY_STATUS['mosac']:
        print("⚠️ MOSAC requested but not available, will use SAC fallback")

    if USE_SCALARIZED_WRAPPER and not DEPENDENCY_STATUS['scalarized_wrapper']:
        print("⚠️ Scalarized wrapper requested but not available, will use base environment")

    return True


def estimate_training_time(timesteps: int, configs: List[str], seeds: List[int]) -> Dict[str, float]:
    """
    Estimate training time for experiments.

    Args:
        timesteps: Timesteps per experiment
        configs: List of configurations
        seeds: List of seeds

    Returns:
        Dictionary with time estimates
    """
    # Base estimates (very rough)
    seconds_per_1k_timesteps = 5 if DEVICE == 'cuda' else 15
    setup_overhead_per_experiment = 30  # seconds

    experiments = len(configs) * len(seeds)
    training_time = (timesteps / 1000) * seconds_per_1k_timesteps * experiments
    overhead_time = setup_overhead_per_experiment * experiments
    total_time = training_time + overhead_time

    return {
        'total_experiments': experiments,
        'estimated_training_time_minutes': training_time / 60,
        'estimated_overhead_time_minutes': overhead_time / 60,
        'estimated_total_time_minutes': total_time / 60,
        'estimated_total_time_hours': total_time / 3600
    }


if __name__ == "__main__":
    # Test core functionality
    print("🔧 MORL Core Training Module")
    print_system_info()

    if check_training_readiness():
        print("✅ System ready for training!")
    else:
        print("❌ System not ready for training")

    # Test configuration loading
    print(f"\n📋 Available configurations: {len(WEIGHT_CONFIGURATIONS)}")
    for name in WEIGHT_CONFIGURATIONS:
        config = get_config(name)
        print(f"  {name}: {config['weights']}")

    print("Core module loaded successfully!")