#!/usr/bin/env python3
"""
Scalarized MORL Trainer

Training and evaluation functionality for Multi-Objective Reinforcement Learning
with scalarized rewards using three key configurations.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# RL and Environment imports
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

# EnergyNet imports
from energy_net.envs.energy_net_v0 import EnergyNetV0
from energy_net.market.pricing.cost_types import CostType
from energy_net.market.pricing.pricing_policy import PricingPolicy
from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern

from MORL_modules.agents.mosac import MOSAC
from MORL_modules.wrappers.scalarized_mo_pcs_wrapper import ScalarizedMOPCSWrapper
from MORL_modules.wrappers.mo_pcs_wrapper import MOPCSWrapper
from MORL_modules.wrappers.dict_to_box_wrapper import DictToBoxWrapper
# ============================================================================
# CONFIGURATION SETTINGS
# ============================================================================

# Three key configurations for comparison
CONFIGURATIONS = {
    'economic_only': {
        'weights': [1.0, 0.0, 0.0, 0.0],
        'description': 'Pure Economic Optimization',
        'color': '#d62728'  # Red
    },
    'economic_battery': {
        'weights': [0.5, 0.5, 0.0, 0.0],
        'description': 'Economic + Battery Health',
        'color': '#ff7f0e'  # Orange
    },
    'balanced': {
        'weights': [0.25, 0.25, 0.25, 0.25],
        'description': 'Balanced Multi-Objective',
        'color': '#2ca02c'  # Green
    }
}

OBJECTIVE_NAMES = ['Economic Profit', 'Battery Health', 'Grid Support', 'Energy Autonomy']

# Default training parameters
DEFAULT_PARAMS = {
    'timesteps': 50000,
    'learning_rate': 3e-4,
    'batch_size': 256,
    'buffer_size': 100000,
    'seed': 42,
    'device': 'cuda' if th.cuda.is_available() else 'cpu'
}

# Environment parameters
ENV_PARAMS = {
    'controller_name': 'EnergyNetController',
    'controller_module': 'energy_net.controllers',
    'env_config_path': 'energy_net/configs/environment_config.yaml',
    'iso_config_path': 'energy_net/configs/iso_config.yaml',
    'pcs_unit_config_path': 'energy_net/configs/pcs_unit_config.yaml',
    'cost_type': CostType.CONSTANT,
    'pricing_policy': PricingPolicy.QUADRATIC,
    'demand_pattern': DemandPattern.SINUSOIDAL
}

# Plotting configuration
plt.style.use('default')
sns.set_palette("husl")


# ============================================================================
# ENVIRONMENT AND AGENT CREATION
# ============================================================================

def create_environment(weights: List[float], seed: int = 42) -> ScalarizedMOPCSWrapper:
    """Create scalarized EnergyNet environment."""

    # Create base environment
    base_env = EnergyNetV0(
        controller_name=ENV_PARAMS['controller_name'],
        controller_module=ENV_PARAMS['controller_module'],
        env_config_path=ENV_PARAMS['env_config_path'],
        iso_config_path=ENV_PARAMS['iso_config_path'],
        pcs_unit_config_path=ENV_PARAMS['pcs_unit_config_path'],
        cost_type=ENV_PARAMS['cost_type'],
        pricing_policy=ENV_PARAMS['pricing_policy'],
        demand_pattern=ENV_PARAMS['demand_pattern']
    )

    base_env.reset(seed=seed)
    dict_to_box_env = DictToBoxWrapper(base_env)
    # Wrap with scalarization
    scalarized_env = ScalarizedMOPCSWrapper(
        dict_to_box_env,
        weights=weights,
        normalize_weights=True,
        log_level='WARNING'
    )

    return scalarized_env


def create_agent(env, log_dir: str, **kwargs) -> MOSAC:
    """Create MOSAC agent."""

    params = DEFAULT_PARAMS.copy()
    params.update(kwargs)

    model = MOSAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=params['learning_rate'],
        buffer_size=params['buffer_size'],
        batch_size=params['batch_size'],
        device=params['device'],
        verbose=1,
        tensorboard_log=os.path.join(log_dir, 'tensorboard')
    )

    return model


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_configuration(config_name: str, timesteps: int = None, seed: int = 42,
                        log_dir: str = 'logs') -> Dict[str, Any]:
    """Train a single configuration."""

    print(f"\n🚀 Training {config_name}")
    print("=" * 50)

    config = CONFIGURATIONS[config_name]
    weights = config['weights']
    timesteps = timesteps or DEFAULT_PARAMS['timesteps']

    print(f"Description: {config['description']}")
    print(f"Weights: {weights}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Seed: {seed}")

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(log_dir) / f"{config_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create environment and agent
        env = create_environment(weights, seed)

        # Add monitor for tracking
        monitor_file = exp_dir / "training.monitor.csv"
        env = Monitor(env, str(monitor_file))

        agent = create_agent(env, str(exp_dir))

        # Training
        print("🏋️ Starting training...")
        start_time = time.time()

        agent.learn(total_timesteps=timesteps, progress_bar=True)

        training_time = time.time() - start_time

        # Save model
        model_path = exp_dir / "final_model"
        agent.save(str(model_path))

        # Evaluation
        print("📊 Running evaluation...")
        eval_results = evaluate_model(agent, env, n_episodes=20)

        # Save results
        results = {
            'config_name': config_name,
            'weights': weights,
            'timesteps': timesteps,
            'seed': seed,
            'training_time': training_time,
            'eval_results': eval_results,
            'exp_dir': str(exp_dir),
            'model_path': str(model_path),
            'monitor_file': str(monitor_file)
        }

        with open(exp_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"✅ Training completed in {training_time:.1f}s")
        print(f"📊 Mean eval reward: {eval_results['mean_reward']:.4f}")
        print(f"📁 Results saved to: {exp_dir}")

        return results

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return {
            'config_name': config_name,
            'status': 'failed',
            'error': str(e),
            'exp_dir': str(exp_dir)
        }


def evaluate_model(model, env, n_episodes: int = 20) -> Dict[str, Any]:
    """Evaluate trained model."""

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

            if episode_length >= 1000:  # Safety limit
                break

        episode_rewards.append(float(total_reward))
        episode_lengths.append(episode_length)

    return {
        'n_episodes': n_episodes,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }


# ============================================================================
# RESULTS ANALYSIS
# ============================================================================

def load_training_data(monitor_file: str) -> pd.DataFrame:
    """Load training data from monitor CSV."""
    try:
        # Skip the header comment line
        df = pd.read_csv(monitor_file, skiprows=1)
        df['episode'] = range(len(df))
        df['reward_ma'] = df['r'].rolling(window=10, min_periods=1).mean()
        return df
    except Exception as e:
        print(f"⚠️ Could not load {monitor_file}: {e}")
        return pd.DataFrame()


def compare_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare results from multiple configurations."""

    print("\n📊 RESULTS COMPARISON")
    print("=" * 60)

    comparison = {}

    for result in results_list:
        if 'eval_results' not in result:
            continue

        config_name = result['config_name']
        eval_results = result['eval_results']

        comparison[config_name] = {
            'description': CONFIGURATIONS[config_name]['description'],
            'weights': CONFIGURATIONS[config_name]['weights'],
            'mean_reward': eval_results['mean_reward'],
            'std_reward': eval_results['std_reward'],
            'training_time': result.get('training_time', 0)
        }

        print(f"\n{config_name.upper()}")
        print(f"  Description: {CONFIGURATIONS[config_name]['description']}")
        print(f"  Weights: {CONFIGURATIONS[config_name]['weights']}")
        print(f"  Mean Reward: {eval_results['mean_reward']:.4f} ± {eval_results['std_reward']:.4f}")
        print(f"  Range: [{eval_results['min_reward']:.4f}, {eval_results['max_reward']:.4f}]")
        print(f"  Training Time: {result.get('training_time', 0):.1f}s")

    # Find best configuration
    if comparison:
        best_config = max(comparison.keys(), key=lambda k: comparison[k]['mean_reward'])
        best_reward = comparison[best_config]['mean_reward']

        print(f"\n🏆 BEST CONFIGURATION: {best_config}")
        print(f"   Reward: {best_reward:.4f}")
        print(f"   Description: {comparison[best_config]['description']}")

    print("=" * 60)

    return comparison


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_analysis_plots(results_list: List[Dict[str, Any]], output_dir: str):
    """Create comprehensive analysis plots."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n📈 Creating analysis plots in {output_dir}")

    # Load all training data
    training_data = {}
    for result in results_list:
        if 'monitor_file' in result and os.path.exists(result['monitor_file']):
            config_name = result['config_name']
            df = load_training_data(result['monitor_file'])
            if not df.empty:
                df['config_name'] = config_name
                training_data[config_name] = df

    if not training_data:
        print("⚠️ No training data found for plotting")
        return

    # Create the four main plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Learning curves comparison
    plot_learning_curves(ax1, training_data)

    # Plot 2: Final performance comparison
    plot_final_performance(ax2, results_list)

    # Plot 3: Weight configurations
    plot_weight_configurations(ax3)

    # Plot 4: Individual reward traces
    plot_reward_traces(ax4, training_data)

    plt.tight_layout()
    plot_file = output_path / "analysis_plots.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Analysis plots saved to: {plot_file}")


def plot_learning_curves(ax, training_data: Dict[str, pd.DataFrame]):
    """Plot learning curves for all configurations."""

    for config_name, df in training_data.items():
        config = CONFIGURATIONS[config_name]
        color = config['color']

        # Plot smoothed learning curve
        ax.plot(df['episode'], df['reward_ma'],
                label=config['description'],
                color=color, linewidth=2)

        # Add raw data with transparency
        ax.plot(df['episode'], df['r'],
                color=color, alpha=0.3, linewidth=0.5)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Learning Curves Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_final_performance(ax, results_list: List[Dict[str, Any]]):
    """Plot final performance comparison."""

    configs = []
    means = []
    stds = []
    colors = []

    for result in results_list:
        if 'eval_results' in result:
            config_name = result['config_name']
            eval_results = result['eval_results']

            configs.append(CONFIGURATIONS[config_name]['description'])
            means.append(eval_results['mean_reward'])
            stds.append(eval_results['std_reward'])
            colors.append(CONFIGURATIONS[config_name]['color'])

    bars = ax.bar(configs, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.7)

    # Add value labels
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom')

    ax.set_ylabel('Mean Reward')
    ax.set_title('Final Performance Comparison')
    ax.tick_params(axis='x', rotation=45)


def plot_weight_configurations(ax):
    """Plot weight configurations."""

    configs = list(CONFIGURATIONS.keys())
    x = np.arange(len(OBJECTIVE_NAMES))
    width = 0.25

    for i, config_name in enumerate(configs):
        config = CONFIGURATIONS[config_name]
        weights = config['weights']
        color = config['color']

        ax.bar(x + i * width, weights, width,
               label=config['description'],
               color=color, alpha=0.7)

    ax.set_xlabel('Objective')
    ax.set_ylabel('Weight')
    ax.set_title('Weight Configurations')
    ax.set_xticks(x + width)
    ax.set_xticklabels(OBJECTIVE_NAMES, rotation=45)
    ax.legend()


def plot_reward_traces(ax, training_data: Dict[str, pd.DataFrame]):
    """Plot individual reward traces over time."""

    for config_name, df in training_data.items():
        config = CONFIGURATIONS[config_name]
        color = config['color']

        # Plot recent episodes for clarity
        recent_episodes = min(1000, len(df))
        recent_df = df.tail(recent_episodes).copy()
        recent_df['episode_norm'] = range(len(recent_df))

        ax.plot(recent_df['episode_norm'], recent_df['r'],
                label=f"{config['description']} (last {recent_episodes})",
                color=color, alpha=0.8, linewidth=1)

    ax.set_xlabel('Episode (Recent)')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Recent Reward Traces')
    ax.legend()
    ax.grid(True, alpha=0.3)


# ============================================================================
# BATCH OPERATIONS
# ============================================================================

def train_all_configurations(timesteps: int = None, seed: int = 42,
                             log_dir: str = 'logs') -> List[Dict[str, Any]]:
    """Train all three configurations."""

    print("\n🧪 TRAINING ALL CONFIGURATIONS")
    print("=" * 60)

    results_list = []

    for config_name in CONFIGURATIONS.keys():
        result = train_configuration(
            config_name=config_name,
            timesteps=timesteps,
            seed=seed,
            log_dir=log_dir
        )
        results_list.append(result)

    # Compare results
    comparison = compare_results(results_list)

    # Create plots
    create_analysis_plots(results_list, log_dir)

    return results_list


def get_configuration_list() -> List[str]:
    """Get list of available configurations."""
    return list(CONFIGURATIONS.keys())


def validate_configuration(config_name: str) -> bool:
    """Validate configuration name."""
    return config_name in CONFIGURATIONS or config_name == 'all'