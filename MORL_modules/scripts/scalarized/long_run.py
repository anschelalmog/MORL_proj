#!/usr/bin/env python3
"""
10k Economic-Only Run Script

Runs 10,000 episodes on 5 algorithms (PPO, SAC, A2C, TD3, MOSAC)
with economic-only configuration [1, 0, 0, 0] across 3 different seeds.

Usage:
    python economic_10k_run.py
    python economic_10k_run.py --seeds 42 123 456
    python economic_10k_run.py --timesteps 20000
"""

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

# Import from the existing modules
from MORL_modules.scripts.scalarized.scalarized_trainer import (
    create_environment, create_agent, DEFAULT_PARAMS, evaluate_model, load_training_data
)
from MORL_modules.scripts.scalarized.scalarized_baseline import (
    BASELINE_ALGORITHMS, create_baseline_agent, evaluate_baseline_model
)

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Economic-only configuration (pure economic optimization)
ECONOMIC_CONFIG = {
    'weights': [1.0, 0.0, 0.0, 0.0],
    'description': 'Pure Economic Optimization',
    'name': 'economic_only'
}

# All algorithms to test (5 total)
ALGORITHMS = {
    'ppo': {
        'type': 'baseline',
        'name': 'PPO',
        'description': 'Proximal Policy Optimization',
        'color': '#1f77b4'
    },
    'sac': {
        'type': 'baseline',
        'name': 'SAC',
        'description': 'Soft Actor-Critic',
        'color': '#ff7f0e'
    },
    'a2c': {
        'type': 'baseline',
        'name': 'A2C',
        'description': 'Advantage Actor-Critic',
        'color': '#2ca02c'
    },
    'td3': {
        'type': 'baseline',
        'name': 'TD3',
        'description': 'Twin Delayed Deep Deterministic Policy Gradient',
        'color': '#d62728'
    },
    'mosac': {
        'type': 'morl',
        'name': 'MOSAC',
        'description': 'Multi-Objective Soft Actor-Critic',
        'color': '#9467bd'
    }
}

# Default experiment parameters
EXPERIMENT_PARAMS = {
    'timesteps': 10000,
    'seeds': [42, 123, 456],
    'n_eval_episodes': 20,
    'log_dir': 'logs/10k_run'
}


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_algorithm_seed(algorithm: str, seed: int, timesteps: int,
                         base_log_dir: str) -> Dict[str, Any]:
    """Train a single algorithm with a specific seed."""

    print(f"\n🚀 Training {ALGORITHMS[algorithm]['name']} (seed={seed})")
    print("-" * 50)

    alg_info = ALGORITHMS[algorithm]
    weights = ECONOMIC_CONFIG['weights']

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_log_dir) / f"{algorithm}_seed{seed}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create environment
        env = create_environment(weights, seed)

        # Add monitor for tracking
        monitor_file = exp_dir / "training.monitor.csv"
        env = Monitor(env, str(monitor_file))

        # Create agent based on algorithm type
        if alg_info['type'] == 'baseline':
            agent = create_baseline_agent(algorithm, env, str(exp_dir))
            eval_func = evaluate_baseline_model
        else:  # MOSAC
            agent = create_agent(env, str(exp_dir))
            eval_func = evaluate_model

        # Training
        print(f"🏋️ Starting training for {timesteps:,} timesteps...")
        start_time = time.time()

        agent.learn(total_timesteps=timesteps, progress_bar=True)

        training_time = time.time() - start_time

        # Save model
        model_path = exp_dir / "final_model"
        agent.save(str(model_path))

        # Evaluation
        print("📊 Running evaluation...")
        eval_results = eval_func(agent, env, n_episodes=EXPERIMENT_PARAMS['n_eval_episodes'])

        # Save results
        results = {
            'algorithm': algorithm,
            'algorithm_name': alg_info['name'],
            'algorithm_type': alg_info['type'],
            'seed': seed,
            'timesteps': timesteps,
            'weights': weights,
            'config_name': ECONOMIC_CONFIG['name'],
            'training_time': training_time,
            'eval_results': eval_results,
            'exp_dir': str(exp_dir),
            'model_path': str(model_path),
            'monitor_file': str(monitor_file)
        }

        # Save individual result
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
            'algorithm_name': alg_info['name'],
            'seed': seed,
            'status': 'failed',
            'error': str(e),
            'exp_dir': str(exp_dir)
        }


def run_full_experiment(algorithms: List[str], seeds: List[int],
                        timesteps: int, log_dir: str) -> List[Dict[str, Any]]:
    """Run the full experiment across all algorithms and seeds."""

    print("\n🧪 STARTING 10K ECONOMIC-ONLY EXPERIMENT")
    print("=" * 70)
    print(f"Configuration: {ECONOMIC_CONFIG['description']}")
    print(f"Weights: {ECONOMIC_CONFIG['weights']}")
    print(f"Algorithms: {[ALGORITHMS[a]['name'] for a in algorithms]}")
    print(f"Seeds: {seeds}")
    print(f"Timesteps per run: {timesteps:,}")
    print(f"Total experiments: {len(algorithms) * len(seeds)}")
    print("=" * 70)

    results_list = []
    experiment_count = 0
    total_experiments = len(algorithms) * len(seeds)

    start_time = time.time()

    for algorithm in algorithms:
        for seed in seeds:
            experiment_count += 1
            print(f"\n📊 Experiment {experiment_count}/{total_experiments}")
            print(f"Algorithm: {ALGORITHMS[algorithm]['name']}, Seed: {seed}")

            result = train_algorithm_seed(algorithm, seed, timesteps, log_dir)
            results_list.append(result)

            # Save intermediate results
            save_batch_results(results_list, log_dir)

    total_time = time.time() - start_time

    print(f"\n🏁 Full experiment completed in {total_time / 60:.1f} minutes")

    return results_list


# ============================================================================
# ANALYSIS AND PLOTTING
# ============================================================================

def analyze_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze experiment results across algorithms and seeds."""

    print("\n📊 ANALYZING RESULTS")
    print("=" * 50)

    # Group results by algorithm
    algorithm_results = {}
    successful_runs = [r for r in results_list if 'eval_results' in r]
    failed_runs = [r for r in results_list if 'eval_results' not in r]

    print(f"Total experiments: {len(results_list)}")
    print(f"Successful: {len(successful_runs)}")
    print(f"Failed: {len(failed_runs)}")

    if failed_runs:
        print("\nFailed experiments:")
        for failed in failed_runs:
            print(f"  - {failed['algorithm_name']} (seed {failed['seed']}): {failed.get('error', 'Unknown')}")

    # Group by algorithm
    for result in successful_runs:
        algorithm = result['algorithm']
        if algorithm not in algorithm_results:
            algorithm_results[algorithm] = {
                'name': result['algorithm_name'],
                'type': result['algorithm_type'],
                'rewards': [],
                'training_times': [],
                'seeds': []
            }

        algorithm_results[algorithm]['rewards'].append(result['eval_results']['mean_reward'])
        algorithm_results[algorithm]['training_times'].append(result['training_time'])
        algorithm_results[algorithm]['seeds'].append(result['seed'])

    # Calculate statistics
    statistics = {}
    print(f"\n{'Algorithm':<15} {'Mean Reward':<12} {'Std Dev':<10} {'Training Time':<12} {'Seeds'}")
    print("-" * 70)

    for algorithm, data in algorithm_results.items():
        if data['rewards']:
            mean_reward = np.mean(data['rewards'])
            std_reward = np.std(data['rewards'])
            mean_time = np.mean(data['training_times'])

            statistics[algorithm] = {
                'name': data['name'],
                'type': data['type'],
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'mean_training_time': mean_time,
                'n_seeds': len(data['rewards']),
                'all_rewards': data['rewards']
            }

            print(f"{data['name']:<15} {mean_reward:<12.4f} {std_reward:<10.4f} "
                  f"{mean_time:<12.1f}s {data['seeds']}")

    # Find best algorithm
    if statistics:
        best_algorithm = max(statistics.keys(), key=lambda k: statistics[k]['mean_reward'])
        best_reward = statistics[best_algorithm]['mean_reward']
        print(f"\n🏆 Best Algorithm: {statistics[best_algorithm]['name']}")
        print(f"   Mean Reward: {best_reward:.4f} ± {statistics[best_algorithm]['std_reward']:.4f}")

    return {
        'algorithm_results': algorithm_results,
        'statistics': statistics,
        'successful_runs': successful_runs,
        'failed_runs': failed_runs
    }


def create_experiment_plots(results_list: List[Dict[str, Any]], output_dir: str):
    """Create comprehensive plots for the experiment results."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n📈 Creating experiment plots in {output_dir}")

    # Analyze results first
    analysis = analyze_results(results_list)
    statistics = analysis['statistics']
    successful_runs = analysis['successful_runs']

    if not statistics:
        print("⚠️ No successful runs to plot")
        return

    # Create comprehensive plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Mean performance comparison
    plot_mean_performance(ax1, statistics)

    # Plot 2: Performance distribution across seeds
    plot_performance_distribution(ax2, statistics)

    # Plot 3: Training time comparison
    plot_training_times(ax3, statistics)

    # Plot 4: Learning curves comparison
    plot_learning_curves(ax4, successful_runs)

    plt.suptitle(f'10K Economic-Only Experiment Results\nConfiguration: {ECONOMIC_CONFIG["weights"]}',
                 fontsize=16, y=0.98)
    plt.tight_layout()

    plot_file = output_path / "experiment_results.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Experiment plots saved to: {plot_file}")

    # Create individual learning curves plot
    create_detailed_learning_curves(successful_runs, output_path)


def plot_mean_performance(ax, statistics: Dict):
    """Plot mean performance with error bars."""

    algorithms = list(statistics.keys())
    means = [statistics[alg]['mean_reward'] for alg in algorithms]
    stds = [statistics[alg]['std_reward'] for alg in algorithms]
    names = [statistics[alg]['name'] for alg in algorithms]
    colors = [ALGORITHMS[alg]['color'] for alg in algorithms]

    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, alpha=0.7)

    # Add value labels
    for bar, mean_val, std_val in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + std_val + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom')

    ax.set_ylabel('Mean Reward')
    ax.set_title('Mean Performance Across Seeds')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)


def plot_performance_distribution(ax, statistics: Dict):
    """Plot performance distribution across seeds."""

    data_for_violin = []
    labels = []
    colors = []

    for algorithm, stats in statistics.items():
        data_for_violin.append(stats['all_rewards'])
        labels.append(stats['name'])
        colors.append(ALGORITHMS[algorithm]['color'])

    positions = range(1, len(data_for_violin) + 1)
    violin_parts = ax.violinplot(data_for_violin, positions=positions, showmeans=True)

    # Color the violins
    for pc, color in zip(violin_parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Reward')
    ax.set_title('Performance Distribution Across Seeds')
    ax.grid(True, alpha=0.3)


def plot_training_times(ax, statistics: Dict):
    """Plot training time comparison."""

    algorithms = list(statistics.keys())
    times = [statistics[alg]['mean_training_time'] for alg in algorithms]
    names = [statistics[alg]['name'] for alg in algorithms]
    colors = [ALGORITHMS[alg]['color'] for alg in algorithms]

    bars = ax.bar(names, times, color=colors, alpha=0.7)

    # Add value labels
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 5,
                f'{time_val:.0f}s', ha='center', va='bottom')

    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison')
    ax.tick_params(axis='x', rotation=45)


def plot_learning_curves(ax, successful_runs: List[Dict]):
    """Plot learning curves for all successful runs."""

    # Load and plot training data
    for result in successful_runs:
        if 'monitor_file' in result and os.path.exists(result['monitor_file']):
            df = load_training_data(result['monitor_file'])
            if not df.empty:
                algorithm = result['algorithm']
                seed = result['seed']
                color = ALGORITHMS[algorithm]['color']
                name = ALGORITHMS[algorithm]['name']

                # Plot with transparency to show individual runs
                ax.plot(df['episode'], df['reward_ma'],
                        color=color, alpha=0.6, linewidth=1,
                        label=f"{name} (seed {seed})")

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (Moving Average)')
    ax.set_title('Learning Curves')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)


def create_detailed_learning_curves(successful_runs: List[Dict], output_path: Path):
    """Create detailed learning curves plot by algorithm."""

    # Group by algorithm
    algorithm_data = {}
    for result in successful_runs:
        if 'monitor_file' in result and os.path.exists(result['monitor_file']):
            algorithm = result['algorithm']
            if algorithm not in algorithm_data:
                algorithm_data[algorithm] = []

            df = load_training_data(result['monitor_file'])
            if not df.empty:
                df['seed'] = result['seed']
                algorithm_data[algorithm].append(df)

    if not algorithm_data:
        return

    n_algorithms = len(algorithm_data)
    fig, axes = plt.subplots(1, n_algorithms, figsize=(5 * n_algorithms, 6))
    if n_algorithms == 1:
        axes = [axes]

    for i, (algorithm, dfs) in enumerate(algorithm_data.items()):
        ax = axes[i]
        color = ALGORITHMS[algorithm]['color']
        name = ALGORITHMS[algorithm]['name']

        for df in dfs:
            seed = df['seed'].iloc[0]
            # Raw data with high transparency
            ax.plot(df['episode'], df['r'], color=color, alpha=0.2, linewidth=0.5)
            # Smoothed curve
            ax.plot(df['episode'], df['reward_ma'], color=color, alpha=0.8,
                    linewidth=1.5, label=f'Seed {seed}')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(f'{name} Learning Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = output_path / "detailed_learning_curves.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Detailed learning curves saved to: {plot_file}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_batch_results(results_list: List[Dict[str, Any]], log_dir: str):
    """Save batch results to JSON file."""

    results_file = Path(log_dir) / "batch_results.json"

    # Create summary
    summary = {
        'experiment_config': {
            'name': 'Economic-Only 10K Run',
            'description': ECONOMIC_CONFIG['description'],
            'weights': ECONOMIC_CONFIG['weights'],
            'timesteps': EXPERIMENT_PARAMS['timesteps'],
            'seeds': EXPERIMENT_PARAMS['seeds'],
            'algorithms': list(ALGORITHMS.keys())
        },
        'timestamp': datetime.now().isoformat(),
        'total_experiments': len(results_list),
        'completed_experiments': len([r for r in results_list if 'eval_results' in r]),
        'failed_experiments': len([r for r in results_list if 'eval_results' not in r]),
        'results': results_list
    }

    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"📁 Batch results saved to: {results_file}")


def print_header():
    """Print experiment header."""
    print("=" * 70)
    print("🤖 10K ECONOMIC-ONLY EXPERIMENT")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEFAULT_PARAMS['device']}")
    print(f"Configuration: {ECONOMIC_CONFIG['description']}")
    print(f"Weights: {ECONOMIC_CONFIG['weights']}")
    print("=" * 70)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""

    parser = argparse.ArgumentParser(
        description="10K Economic-Only MORL Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (10k timesteps, seeds 42,123,456)
  python economic_10k_run.py

  # Custom timesteps and seeds
  python economic_10k_run.py --timesteps 20000 --seeds 42 100 200

  # Custom log directory
  python economic_10k_run.py --log-dir logs/custom_10k_run

Available Algorithms: ppo, sac, a2c, td3, mosac
Configuration: Economic-Only [1, 0, 0, 0]
        """
    )

    parser.add_argument(
        '--timesteps',
        type=int,
        default=EXPERIMENT_PARAMS['timesteps'],
        help=f'Number of training timesteps (default: {EXPERIMENT_PARAMS["timesteps"]:,})'
    )

    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=EXPERIMENT_PARAMS['seeds'],
        help=f'Random seeds to use (default: {EXPERIMENT_PARAMS["seeds"]})'
    )

    parser.add_argument(
        '--algorithms',
        nargs='+',
        choices=list(ALGORITHMS.keys()) + ['all'],
        default=list(ALGORITHMS.keys()),
        help='Algorithms to train (default: all)'
    )

    parser.add_argument(
        '--log-dir',
        default=EXPERIMENT_PARAMS['log_dir'],
        help=f'Base directory for logs and results (default: {EXPERIMENT_PARAMS["log_dir"]})'
    )

    parser.add_argument(
        '--eval-episodes',
        type=int,
        default=EXPERIMENT_PARAMS['n_eval_episodes'],
        help=f'Number of evaluation episodes (default: {EXPERIMENT_PARAMS["n_eval_episodes"]})'
    )

    return parser


def main():
    """Main entry point."""

    parser = create_parser()
    args = parser.parse_args()

    print_header()

    try:
        # Handle 'all' algorithms
        if 'all' in args.algorithms:
            algorithms = list(ALGORITHMS.keys())
        else:
            algorithms = args.algorithms

        # Validate algorithms
        for alg in algorithms:
            if alg not in ALGORITHMS:
                print(f"❌ Invalid algorithm: {alg}")
                print("Available algorithms:", list(ALGORITHMS.keys()))
                sys.exit(1)

        # Update global experiment parameters
        EXPERIMENT_PARAMS['timesteps'] = args.timesteps
        EXPERIMENT_PARAMS['seeds'] = args.seeds
        EXPERIMENT_PARAMS['n_eval_episodes'] = args.eval_episodes

        # Create log directory
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Log directory: {log_dir.absolute()}")
        print(f"🤖 Algorithms: {[ALGORITHMS[a]['name'] for a in algorithms]}")
        print(f"🎲 Seeds: {args.seeds}")
        print(f"⏱️ Timesteps per run: {args.timesteps:,}")
        print(f"📊 Evaluation episodes: {args.eval_episodes}")

        # Run the full experiment
        results = run_full_experiment(
            algorithms=algorithms,
            seeds=args.seeds,
            timesteps=args.timesteps,
            log_dir=str(log_dir)
        )

        # Analyze and plot results
        print("\n📊 Analyzing results...")
        analysis = analyze_results(results)

        print("\n📈 Creating plots...")
        create_experiment_plots(results, str(log_dir))

        # Final summary
        successful_runs = len([r for r in results if 'eval_results' in r])
        failed_runs = len([r for r in results if 'eval_results' not in r])

        print(f"\n✅ 10K Economic-Only Experiment Completed!")
        print(f"   Total experiments: {len(results)}")
        print(f"   Successful: {successful_runs}")
        print(f"   Failed: {failed_runs}")
        print(f"   Results saved to: {log_dir.absolute()}")

        if analysis['statistics']:
            best_alg = max(analysis['statistics'].keys(),
                           key=lambda k: analysis['statistics'][k]['mean_reward'])
            best_name = analysis['statistics'][best_alg]['name']
            best_reward = analysis['statistics'][best_alg]['mean_reward']
            print(f"\n🏆 Best Algorithm: {best_name}")
            print(f"   Mean Reward: {best_reward:.4f}")

    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\n🏁 Experiment completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()