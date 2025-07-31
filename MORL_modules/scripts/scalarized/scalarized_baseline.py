#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch as th
from stable_baselines3 import PPO, SAC, A2C, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from MORL_modules.scripts.scalarized.scalarized_trainer import (create_environment,
    CONFIGURATIONS, DEFAULT_PARAMS, load_training_data)
from MORL_modules.wrappers.dict_to_box_wrapper import DictToBoxWrapper

BASELINE_ALGORITHMS = {
    'ppo': {
        'class': PPO,
        'name': 'PPO',
        'description': 'Proximal Policy Optimization',
        'params': {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.0,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'policy_kwargs': {'net_arch': [256, 256]}
        },
        'color': '#1f77b4'  # Blue
    },
    'sac': {
        'class': SAC,
        'name': 'SAC',
        'description': 'Soft Actor-Critic',
        'params': {
            'learning_rate': 3e-4,
            'buffer_size': 100000,
            'learning_starts': 1000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 1,
            'gradient_steps': 1,
            'policy_kwargs': {'net_arch': [256, 256]}
        },
        'color': '#ff7f0e'  # Orange
    },
    'a2c': {
        'class': A2C,
        'name': 'A2C',
        'description': 'Advantage Actor-Critic',
        'params': {
            'learning_rate': 7e-4,
            'n_steps': 5,
            'gamma': 0.99,
            'gae_lambda': 1.0,
            'ent_coef': 0.0,
            'vf_coef': 0.25,
            'max_grad_norm': 0.5,
            'rms_prop_eps': 1e-5,
            'policy_kwargs': {'net_arch': [256, 256]}
        },
        'color': '#2ca02c'  # Green
    },
    'td3': {
        'class': TD3,
        'name': 'TD3',
        'description': 'Twin Delayed Deep Deterministic Policy Gradient',
        'params': {
            'learning_rate': 1e-3,
            'buffer_size': 100000,
            'learning_starts': 1000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 1,
            'gradient_steps': 1,
            'policy_delay': 2,
            'target_policy_noise': 0.2,
            'target_noise_clip': 0.5,
            'policy_kwargs': {'net_arch': [256, 256]}
        },
        'color': '#d62728'  # Red
    }
}


def create_baseline_agent(algorithm: str, env, log_dir: str, **kwargs):
    if algorithm not in BASELINE_ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    alg_config = BASELINE_ALGORITHMS[algorithm]
    agent_class = alg_config['class']

    params = alg_config['params'].copy()
    params.update(kwargs)
    params['device'] = DEFAULT_PARAMS['device']
    params['verbose'] = 1
    params['tensorboard_log'] = os.path.join(log_dir, 'tensorboard')

    agent = agent_class(policy="MlpPolicy", env=env, **params)

    return agent


def train_baseline_configuration(config_name: str, algorithm: str,
                                 timesteps: int = None, seed: int = 42,
                                 log_dir: str = 'logs') -> Dict[str, Any]:

    print(f"\n🚀 Training {algorithm.upper()} on {config_name}")
    print("=" * 60)

    config = CONFIGURATIONS[config_name]
    alg_config = BASELINE_ALGORITHMS[algorithm]
    weights = config['weights']
    timesteps = timesteps or DEFAULT_PARAMS['timesteps']

    print(f"Algorithm: {alg_config['description']}")
    print(f"Configuration: {config['description']}")
    print(f"Weights: {weights}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Seed: {seed}")

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(log_dir) / f"baseline_{algorithm}_{config_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create environment
        env = create_environment(weights, seed)

        # Add monitor for tracking
        monitor_file = exp_dir / "training.monitor.csv"
        env = Monitor(env, str(monitor_file))

        # Create baseline agent
        agent = create_baseline_agent(algorithm, env, str(exp_dir))

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
        eval_results = evaluate_baseline_model(agent, env, n_episodes=20)

        # Save results
        results = {
            'algorithm': algorithm,
            'algorithm_name': alg_config['name'],
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
            'algorithm': algorithm,
            'config_name': config_name,
            'status': 'failed',
            'error': str(e),
            'exp_dir': str(exp_dir)
        }


def evaluate_baseline_model(model, env, n_episodes: int = 20) -> Dict[str, Any]:
    """Evaluate baseline model (same as MOSAC evaluation)."""

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
# BASELINE ANALYSIS AND COMPARISON
# ============================================================================

def compare_baseline_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare baseline results across algorithms and configurations."""

    print("\n📊 BASELINE RESULTS COMPARISON")
    print("=" * 70)

    # Group by configuration and algorithm
    comparison = {}

    for result in results_list:
        if 'eval_results' not in result:
            continue

        config_name = result['config_name']
        algorithm = result['algorithm']
        eval_results = result['eval_results']

        if config_name not in comparison:
            comparison[config_name] = {}

        comparison[config_name][algorithm] = {
            'algorithm_name': result['algorithm_name'],
            'mean_reward': eval_results['mean_reward'],
            'std_reward': eval_results['std_reward'],
            'training_time': result.get('training_time', 0)
        }

    # Print comparison by configuration
    for config_name, algorithms in comparison.items():
        config_desc = CONFIGURATIONS[config_name]['description']
        weights = CONFIGURATIONS[config_name]['weights']

        print(f"\n{config_name.upper()} - {config_desc}")
        print(f"Weights: {weights}")
        print("-" * 50)

        # Sort algorithms by performance
        sorted_algorithms = sorted(
            algorithms.items(),
            key=lambda x: x[1]['mean_reward'],
            reverse=True
        )

        for i, (algorithm, stats) in enumerate(sorted_algorithms):
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"{rank_emoji} {stats['algorithm_name']:<8}: "
                  f"{stats['mean_reward']:7.4f} ± {stats['std_reward']:6.4f} "
                  f"({stats['training_time']:5.1f}s)")

    # Overall best performer
    all_results = []
    for config_results in comparison.values():
        for alg, stats in config_results.items():
            all_results.append((alg, stats['algorithm_name'], stats['mean_reward']))

    if all_results:
        best_alg, best_name, best_reward = max(all_results, key=lambda x: x[2])
        print(f"\n🏆 OVERALL BEST BASELINE: {best_name}")
        print(f"   Best reward: {best_reward:.4f}")

    print("=" * 70)

    return comparison


def create_baseline_plots(results_list: List[Dict[str, Any]], output_dir: str):
    """Create comprehensive baseline analysis plots."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n📈 Creating baseline analysis plots in {output_dir}")

    # Load all training data
    training_data = {}

    for result in results_list:
        if 'monitor_file' in result and os.path.exists(result['monitor_file']):
            key = f"{result['algorithm']}_{result['config_name']}"
            df = load_training_data(result['monitor_file'])
            if not df.empty:
                df['algorithm'] = result['algorithm']
                df['config_name'] = result['config_name']
                df['key'] = key
                training_data[key] = df

    if not training_data:
        print("⚠️ No training data found for plotting")
        return

    # Create comprehensive baseline plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Algorithm comparison by configuration
    plot_algorithm_comparison(ax1, results_list)

    # Plot 2: Learning curves by algorithm
    plot_baseline_learning_curves(ax2, training_data)

    # Plot 3: Performance vs training time
    plot_performance_vs_time(ax3, results_list)

    # Plot 4: Configuration sensitivity across algorithms
    plot_configuration_sensitivity(ax4, results_list)

    plt.tight_layout()
    plot_file = output_path / "baseline_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Baseline analysis plots saved to: {plot_file}")

    # Create individual learning curves plot
    create_detailed_learning_curves(training_data, output_path)


def plot_algorithm_comparison(ax, results_list: List[Dict[str, Any]]):
    """Plot algorithm performance comparison grouped by configuration."""

    # Group results
    data = {}
    for result in results_list:
        if 'eval_results' not in result:
            continue

        config = result['config_name']
        algorithm = result['algorithm']
        reward = result['eval_results']['mean_reward']
        std = result['eval_results']['std_reward']

        if config not in data:
            data[config] = {}
        data[config][algorithm] = {'mean': reward, 'std': std}

    # Plot grouped bars
    configs = list(data.keys())
    algorithms = list(BASELINE_ALGORITHMS.keys())

    x = np.arange(len(configs))
    width = 0.8 / len(algorithms)

    for i, algorithm in enumerate(algorithms):
        means = []
        stds = []
        for config in configs:
            if algorithm in data[config]:
                means.append(data[config][algorithm]['mean'])
                stds.append(data[config][algorithm]['std'])
            else:
                means.append(0)
                stds.append(0)

        color = BASELINE_ALGORITHMS[algorithm]['color']
        ax.bar(x + i * width, means, width, yerr=stds,
               label=BASELINE_ALGORITHMS[algorithm]['name'],
               color=color, alpha=0.7, capsize=3)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Algorithm Performance by Configuration')
    ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels([CONFIGURATIONS[c]['description'] for c in configs])
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_baseline_learning_curves(ax, training_data: Dict[str, pd.DataFrame]):
    """Plot learning curves for different algorithms."""

    for key, df in training_data.items():
        algorithm = df['algorithm'].iloc[0]
        config = df['config_name'].iloc[0]

        color = BASELINE_ALGORITHMS[algorithm]['color']
        label = f"{BASELINE_ALGORITHMS[algorithm]['name']} ({config})"

        # Plot smoothed curve
        ax.plot(df['episode'], df['reward_ma'],
                label=label, color=color, linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (Moving Average)')
    ax.set_title('Learning Curves by Algorithm')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)


def plot_performance_vs_time(ax, results_list: List[Dict[str, Any]]):
    """Plot performance vs training time scatter."""

    for result in results_list:
        if 'eval_results' not in result:
            continue

        algorithm = result['algorithm']
        reward = result['eval_results']['mean_reward']
        time = result.get('training_time', 0)

        color = BASELINE_ALGORITHMS[algorithm]['color']
        ax.scatter(time, reward,
                   color=color,
                   label=BASELINE_ALGORITHMS[algorithm]['name'],
                   s=100, alpha=0.7)

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Performance vs Training Efficiency')
    ax.grid(True, alpha=0.3)


def plot_configuration_sensitivity(ax, results_list: List[Dict[str, Any]]):
    """Plot how each algorithm performs across configurations."""

    # Calculate performance variance across configurations for each algorithm
    algorithm_variance = {}

    for algorithm in BASELINE_ALGORITHMS.keys():
        rewards = []
        for result in results_list:
            if result.get('algorithm') == algorithm and 'eval_results' in result:
                rewards.append(result['eval_results']['mean_reward'])

        if rewards:
            algorithm_variance[algorithm] = {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'rewards': rewards
            }

    # Plot variance as bar chart
    algorithms = list(algorithm_variance.keys())
    means = [algorithm_variance[alg]['mean'] for alg in algorithms]
    stds = [algorithm_variance[alg]['std'] for alg in algorithms]
    colors = [BASELINE_ALGORITHMS[alg]['color'] for alg in algorithms]

    bars = ax.bar(algorithms, stds, color=colors, alpha=0.7)

    # Add mean performance as text
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                f'μ={mean_val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Performance Std Dev')
    ax.set_title('Algorithm Sensitivity Across Configurations')
    ax.tick_params(axis='x', rotation=45)


def create_detailed_learning_curves(training_data: Dict[str, pd.DataFrame], output_path: Path):
    """Create detailed learning curves plot."""

    fig, axes = plt.subplots(1, len(CONFIGURATIONS), figsize=(5 * len(CONFIGURATIONS), 6))
    if len(CONFIGURATIONS) == 1:
        axes = [axes]

    for i, (config_name, config_info) in enumerate(CONFIGURATIONS.items()):
        ax = axes[i]

        # Plot all algorithms for this configuration
        for key, df in training_data.items():
            if df['config_name'].iloc[0] == config_name:
                algorithm = df['algorithm'].iloc[0]
                color = BASELINE_ALGORITHMS[algorithm]['color']
                name = BASELINE_ALGORITHMS[algorithm]['name']

                # Raw data with transparency
                ax.plot(df['episode'], df['r'],
                        color=color, alpha=0.3, linewidth=0.5)

                # Smoothed curve
                ax.plot(df['episode'], df['reward_ma'],
                        color=color, label=name, linewidth=2)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(f'{config_info["description"]}\nWeights: {config_info["weights"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = output_path / "detailed_learning_curves.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Detailed learning curves saved to: {plot_file}")


# ============================================================================
# BATCH BASELINE OPERATIONS
# ============================================================================

def train_baseline_batch(algorithms: List[str], configs: List[str] = None,
                         timesteps: int = None, seed: int = 42,
                         log_dir: str = 'logs') -> List[Dict[str, Any]]:
    """Train multiple baseline algorithms on multiple configurations."""

    if configs is None:
        configs = list(CONFIGURATIONS.keys())

    print("\n🧪 TRAINING BASELINE BATCH")
    print("=" * 60)
    print(f"Algorithms: {[BASELINE_ALGORITHMS[a]['name'] for a in algorithms]}")
    print(f"Configurations: {configs}")
    print(f"Total experiments: {len(algorithms) * len(configs)}")

    results_list = []
    experiment_count = 0
    total_experiments = len(algorithms) * len(configs)

    for algorithm in algorithms:
        for config_name in configs:
            experiment_count += 1
            print(f"\n📊 Experiment {experiment_count}/{total_experiments}")

            result = train_baseline_configuration(
                config_name=config_name,
                algorithm=algorithm,
                timesteps=timesteps,
                seed=seed,
                log_dir=log_dir
            )
            results_list.append(result)

    # Compare results
    comparison = compare_baseline_results(results_list)

    # Create plots
    create_baseline_plots(results_list, log_dir)

    return results_list


# ============================================================================
# CLI INTERFACE
# ============================================================================

def create_baseline_parser() -> argparse.ArgumentParser:
    """Create command line argument parser for baseline experiments."""

    parser = argparse.ArgumentParser(
        description="Baseline Algorithm Comparison for Scalarized MORL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train PPO and SAC on balanced configuration
  python scalarized_baseline.py --config balanced --algorithms ppo sac

  # Train all algorithms on all configurations
  python scalarized_baseline.py --config all --algorithms all

  # Train specific algorithm with custom parameters  
  python scalarized_baseline.py --config economic_only --algorithms td3 --timesteps 100000

Available Algorithms: ppo  sac a2c td3
"""
    )

    parser.add_argument(
        '--config',
        choices=list(CONFIGURATIONS.keys()) + ['all'],
        default='balanced',
        help='Configuration(s) to train on (default: balanced)'
    )

    parser.add_argument(
        '--algorithms',
        nargs='+',
        choices=list(BASELINE_ALGORITHMS.keys()) + ['all'],
        default=['ppo', 'sac'],
        help='Algorithm(s) to train (default: ppo sac)'
    )

    parser.add_argument(
        '--timesteps',
        type=int,
        default=DEFAULT_PARAMS['timesteps'],
        help=f'Number of training timesteps (default: {DEFAULT_PARAMS["timesteps"]:,})'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_PARAMS['seed'],
        help=f'Random seed (default: {DEFAULT_PARAMS["seed"]})'
    )

    parser.add_argument(
        '--log-dir',
        default='logs',
        help='Base directory for logs and results (default: logs)'
    )

    parser.add_argument(
        '--list-algorithms',
        action='store_true',
        help='List available algorithms and exit'
    )

    return parser


def print_baseline_header():
    """Print baseline application header."""
    print("=" * 70)
    print("SCALARIZED MORL BASELINE COMPARISON")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEFAULT_PARAMS['device']}")
    print("=" * 70)


def print_algorithms():
    """Print available algorithms."""
    print("\n🤖 AVAILABLE BASELINE ALGORITHMS")
    print("=" * 50)

    for alg_name, alg_info in BASELINE_ALGORITHMS.items():
        print(f"\n{alg_name}:")
        print(f"  Name: {alg_info['name']}")
        print(f"  Description: {alg_info['description']}")


def main():
    parser = create_baseline_parser()
    args = parser.parse_args()

    print_baseline_header()

    try:
        if 'all' in args.algorithms:
            algorithms = list(BASELINE_ALGORITHMS.keys())
        else:
            algorithms = args.algorithms

        if args.config == 'all':
            configs = list(CONFIGURATIONS.keys())
        else:
            configs = [args.config]

        # Validate algorithms
        for alg in algorithms:
            if alg not in BASELINE_ALGORITHMS:
                print(f" Invalid algorithm: {alg}")
                print("Available algorithms:", list(BASELINE_ALGORITHMS.keys()))

        # Create log directory
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Log directory: {log_dir.absolute()}")
        print(f"🤖 Algorithms: {[BASELINE_ALGORITHMS[a]['name'] for a in algorithms]}")
        print(f"🎯 Configurations: {configs}")
        print(f"⏱️ Timesteps: {args.timesteps:,}")
        print(f"🎲 Seed: {args.seed}")

        total_experiments = len(algorithms) * len(configs)
        print(f"📊 Total experiments: {total_experiments}")

        # Run baseline experiments
        results = train_baseline_batch(
            algorithms=algorithms,
            configs=configs,
            timesteps=args.timesteps,
            seed=args.seed,
            log_dir=str(log_dir)
        )

        # Print final summary
        successful_runs = [r for r in results if 'eval_results' in r]
        failed_runs = [r for r in results if 'eval_results' not in r]

        print(f"\n✅ Baseline comparison completed!")
        print(f"   Successful experiments: {len(successful_runs)}")
        print(f"   Failed experiments: {len(failed_runs)}")

        if failed_runs:
            print("Failed experiments:")
            for failed in failed_runs:
                print(f"   - {failed['algorithm']} on {failed['config_name']}: {failed.get('error', 'Unknown error')}")

    except KeyboardInterrupt:
        print("\n Baseline experiments interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"\n Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\n Baseline run completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()