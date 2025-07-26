#!/usr/bin/env python3
"""
MORL CLI Module

Command-line interface and experiment orchestration for Multi-Objective
Reinforcement Learning training and analysis.
"""

import os
import sys
import argparse
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np

# Import other modules
from .settings import (
    WEIGHT_CONFIGURATIONS, OBJECTIVE_NAMES, DEFAULT_TRAINING_PARAMS,
    QUICK_TEST_CONFIG, FULL_EXPERIMENT_CONFIG, DEV_CONFIG,
    CONFIRM_BATCH_RUNS, VERBOSE_LOGGING, VERSION, SCRIPT_NAME,
    list_configurations, validate_config_name, get_config,
    create_log_directory, get_experiment_name
)

from .trainer import (
    train_single_configuration, print_system_info, check_training_readiness,
)

from .evaluate import (
    ResultsAnalyzer, analyze_results_directory
)


# ============================================================================
# BATCH EXPERIMENT RUNNER
# ============================================================================

class ExperimentRunner:
    """Manages and executes batch experiments."""

    def __init__(self,
                 base_log_dir: str = "logs/batch_experiments",
                 timesteps: int = None,
                 seeds: List[int] = None):
        """
        Initialize experiment runner.

        Args:
            base_log_dir: Base directory for experiment logs
            timesteps: Timesteps per experiment
            seeds: List of random seeds to use
        """
        self.base_log_dir = Path(base_log_dir)
        self.timesteps = timesteps or DEFAULT_TRAINING_PARAMS['timesteps']
        self.seeds = seeds or [42]

        # Experiment tracking
        self.results = {}
        self.start_time = None
        self.experiment_metadata = {
            'start_time': None,
            'end_time': None,
            'total_experiments': 0,
            'successful_experiments': 0,
            'failed_experiments': 0
        }

        # Create directories
        self.base_log_dir.mkdir(parents=True, exist_ok=True)

        print(f"🧪 Experiment Runner Initialized")
        print(f"📁 Log directory: {self.base_log_dir.absolute()}")
        print(f"⏱️ Timesteps per experiment: {self.timesteps:,}")
        print(f"🎲 Seeds: {self.seeds}")

    def run_single_experiment(self, config_name: str, seed: int, **kwargs) -> Dict[str, Any]:
        """
        Run a single experiment with error handling.

        Args:
            config_name: Configuration name
            seed: Random seed
            **kwargs: Additional training parameters

        Returns:
            Experiment result dictionary
        """
        print(f"\n🔬 Running experiment: {config_name} (seed={seed})")

        try:
            result = train_single_configuration(
                config_name=config_name,
                timesteps=self.timesteps,
                seed=seed,
                log_dir=str(self.base_log_dir / "individual_runs"),
                **kwargs
            )

            if result['status'] == 'success':
                self.experiment_metadata['successful_experiments'] += 1
            else:
                self.experiment_metadata['failed_experiments'] += 1

            return result

        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            if VERBOSE_LOGGING:
                traceback.print_exc()

            self.experiment_metadata['failed_experiments'] += 1

            return {
                'status': 'failed',
                'config_name': config_name,
                'seed': seed,
                'error': str(e)
            }

    def run_all_experiments(self,
                            configs: List[str] = None,
                            **training_kwargs) -> Dict[str, Any]:
        """
        Run all experiments sequentially.

        Args:
            configs: List of configuration names (None = all)
            **training_kwargs: Additional training parameters

        Returns:
            Summary of all experiments
        """
        if configs is None:
            configs = list(WEIGHT_CONFIGURATIONS.keys())

        print("\n" + "=" * 60)
        print("🧪 RUNNING BATCH EXPERIMENTS")
        print("=" * 60)

        total_experiments = len(configs) * len(self.seeds)
        self.experiment_metadata['total_experiments'] = total_experiments

        print(f"Configurations: {configs}")
        print(f"Seeds: {self.seeds}")
        print(f"Total experiments: {total_experiments}")

        # Ask for confirmation if enabled
        if CONFIRM_BATCH_RUNS and total_experiments > 1:
            response = input(f"\n🎯 About to run {total_experiments} experiments. Proceed? [Y/n]: ").lower().strip()
            if response in ['n', 'no']:
                print("Experiments cancelled.")
                return {}

        print("=" * 60)

        # Initialize tracking
        self.start_time = time.time()
        self.experiment_metadata['start_time'] = datetime.now().isoformat()
        experiment_count = 0

        # Run all combinations
        for config_name in configs:
            for seed in self.seeds:
                experiment_count += 1
                print(f"\n📊 Experiment {experiment_count}/{total_experiments}")

                # Run single experiment
                result = self.run_single_experiment(config_name, seed, **training_kwargs)

                # Store result
                if config_name not in self.results:
                    self.results[config_name] = []
                self.results[config_name].append(result)

        # Finalize metadata
        total_time = time.time() - self.start_time
        self.experiment_metadata['end_time'] = datetime.now().isoformat()
        self.experiment_metadata['total_time_seconds'] = total_time

        # Generate summary
        summary = self._generate_summary()

        # Save results
        self._save_results(summary)

        return summary

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate experiment summary."""

        summary = {
            'experiment_info': self.experiment_metadata.copy(),
            'results_by_config': {},
            'status_summary': {'success': 0, 'failed': 0, 'interrupted': 0}
        }

        # Analyze results by configuration
        for config_name, config_results in self.results.items():
            successful_results = [r for r in config_results if r['status'] == 'success']

            config_summary = {
                'total_runs': len(config_results),
                'successful_runs': len(successful_results),
                'failed_runs': len(config_results) - len(successful_results),
                'runs': config_results
            }

            # Calculate performance statistics for successful runs
            if successful_results:
                durations = [r['duration'] for r in successful_results if 'duration' in r]
                if durations:
                    config_summary['avg_duration'] = float(np.mean(durations))
                    config_summary['std_duration'] = float(np.std(durations))

                # Evaluation statistics
                eval_rewards = []
                for result in successful_results:
                    if result.get('eval_results') and 'mean_reward' in result['eval_results']:
                        eval_rewards.append(result['eval_results']['mean_reward'])

                if eval_rewards:
                    config_summary['eval_stats'] = {
                        'mean_reward': float(np.mean(eval_rewards)),
                        'std_reward': float(np.std(eval_rewards)),
                        'min_reward': float(np.min(eval_rewards)),
                        'max_reward': float(np.max(eval_rewards))
                    }

            summary['results_by_config'][config_name] = config_summary

            # Update status summary
            for result in config_results:
                status = result['status']
                if status in summary['status_summary']:
                    summary['status_summary'][status] += 1

        return summary

    def _save_results(self, summary: Dict[str, Any]):
        """Save experiment results and summary."""

        # Save detailed results
        results_file = self.base_log_dir / "experiment_results.json"

        try:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            import json

            # Recursively convert numpy types
            def recursive_convert(data):
                if isinstance(data, dict):
                    return {k: recursive_convert(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [recursive_convert(item) for item in data]
                else:
                    return convert_numpy(data)

            clean_summary = recursive_convert(summary)

            with open(results_file, 'w') as f:
                json.dump(clean_summary, f, indent=2)

            print(f"\n📄 Results saved to: {results_file}")

        except Exception as e:
            print(f"⚠️ Could not save results: {e}")

    def print_summary(self, summary: Dict[str, Any]):
        """Print experiment summary to console."""

        info = summary['experiment_info']
        status = summary['status_summary']

        print("\n" + "=" * 60)
        print("🏁 EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Total experiments: {info['total_experiments']}")
        print(f"Successful: {info['successful_experiments']}")
        print(f"Failed: {info['failed_experiments']}")

        if 'total_time_seconds' in info:
            print(f"Total time: {info['total_time_seconds'] / 60:.1f} minutes")

        success_rate = (info['successful_experiments'] / info['total_experiments'] * 100
                        if info['total_experiments'] > 0 else 0)
        print(f"Success rate: {success_rate:.1f}%")

        print("\nResults by configuration:")
        for config_name, config_summary in summary['results_by_config'].items():
            success_rate = (config_summary['successful_runs'] / config_summary['total_runs'] * 100
                            if config_summary['total_runs'] > 0 else 0)
            print(f"  {config_name}:")
            print(f"    Success rate: {success_rate:.1f}%")
            print(f"    Runs: {config_summary['successful_runs']}/{config_summary['total_runs']}")

            if 'eval_stats' in config_summary:
                eval_stats = config_summary['eval_stats']
                print(f"    Eval reward: {eval_stats['mean_reward']:.4f} ± {eval_stats['std_reward']:.4f}")
                print(f"    Range: [{eval_stats['min_reward']:.4f}, {eval_stats['max_reward']:.4f}]")

            if 'avg_duration' in config_summary:
                print(f"    Avg duration: {config_summary['avg_duration']:.1f}s")

        print("=" * 60)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description=f"{SCRIPT_NAME} v{VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single configuration
  python -m morl_modules.scripts.main train balanced --timesteps 50000

  # Train all configurations with multiple seeds
  python -m morl_modules.scripts.main batch --configs all --seeds 42 123 456

  # Analyze existing results
  python -m morl_modules.scripts.main analyze logs/batch_experiments

  # Quick test run
  python -m morl_modules.scripts.main quick-test

  # List available configurations
  python -m morl_modules.scripts.main list-configs
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train single configuration
    train_parser = subparsers.add_parser('train', help='Train single configuration')
    train_parser.add_argument('config', choices=list_configurations(),
                              help='Configuration to train')
    train_parser.add_argument('--timesteps', type=int, default=DEFAULT_TRAINING_PARAMS['timesteps'],
                              help='Number of training timesteps')
    train_parser.add_argument('--seed', type=int, default=42,
                              help='Random seed')
    train_parser.add_argument('--log-dir', default='logs/morl_training',
                              help='Base logging directory')
    train_parser.add_argument('--no-eval', action='store_true',
                              help='Skip evaluation after training')

    # Batch experiments
    batch_parser = subparsers.add_parser('batch', help='Run batch experiments')
    batch_parser.add_argument('--configs', nargs='+',
                              choices=list_configurations() + ['all'],
                              default=['all'],
                              help='Configurations to train')
    batch_parser.add_argument('--seeds', nargs='+', type=int, default=[42],
                              help='Random seeds to use')
    batch_parser.add_argument('--timesteps', type=int, default=DEFAULT_TRAINING_PARAMS['timesteps'],
                              help='Number of training timesteps')
    batch_parser.add_argument('--log-dir', default='logs/batch_experiments',
                              help='Base logging directory')
    batch_parser.add_argument('--analyze', action='store_true',
                              help='Run analysis after completion')

    # Analysis
    analyze_parser = subparsers.add_parser('analyze', help='Analyze experiment results')
    analyze_parser.add_argument('results_dir',
                                help='Directory containing experiment results')
    analyze_parser.add_argument('--output-dir',
                                help='Output directory for analysis (default: results_dir/analysis)')

    # Evaluation
    eval_parser = subparsers.add_parser('eval', help='Evaluate saved model')
    eval_parser.add_argument('model_path', help='Path to saved model')
    eval_parser.add_argument('config', choices=list_configurations(),
                             help='Configuration name for environment')
    eval_parser.add_argument('--episodes', type=int, default=20,
                             help='Number of evaluation episodes')

    # Quick test
    quick_parser = subparsers.add_parser('quick-test', help='Run quick test')
    quick_parser.add_argument('--config', choices=list_configurations(), default='balanced',
                              help='Configuration to test')

    # List configurations
    list_parser = subparsers.add_parser('list-configs', help='List available configurations')

    # System info
    info_parser = subparsers.add_parser('info', help='Show system information')

    return parser


def main():
    """Main CLI entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Print header
    print("=" * 80)
    print(f"🤖 {SCRIPT_NAME} v{VERSION}")
    print("=" * 80)

    # Check system readiness
    if not check_training_readiness():
        print("❌ System not ready. Please check dependencies.")
        sys.exit(1)

    try:
        if args.command == 'train':
            # Single configuration training
            print(f"🚀 Training single configuration: {args.config}")

            result = train_single_configuration(
                config_name=args.config,
                timesteps=args.timesteps,
                seed=args.seed,
                log_dir=args.log_dir
            )

            print(f"\n✅ Training completed with status: {result['status']}")
            if result['status'] == 'success':
                print(f"📁 Results saved to: {result['log_dir']}")
                if result['eval_results']:
                    eval_reward = result['eval_results']['mean_reward']
                    print(f"📊 Evaluation reward: {eval_reward:.4f}")

        elif args.command == 'batch':
            # Batch experiments
            print("🧪 Running batch experiments")

            # Handle 'all' configurations
            if 'all' in args.configs:
                configs = list_configurations()
            else:
                configs = args.configs

            print(f"Configurations: {configs}")
            print(f"Seeds: {args.seeds}")
            print(f"Timesteps: {args.timesteps:,}")

            # Create experiment runner
            runner = ExperimentRunner(
                base_log_dir=args.log_dir,
                timesteps=args.timesteps,
                seeds=args.seeds
            )

            # Run experiments
            summary = runner.run_all_experiments(configs=configs)

            # Print summary
            if summary:
                runner.print_summary(summary)

                # Run analysis if requested
                if args.analyze:
                    print("\n📊 Running post-experiment analysis...")
                    analyze_results_directory(args.log_dir)

        elif args.command == 'analyze':
            # Results analysis
            print(f"📊 Analyzing results in: {args.results_dir}")

            success = analyze_results_directory(
                results_dir=args.results_dir,
                output_dir=args.output_dir
            )

            if success:
                print("✅ Analysis completed successfully")
            else:
                print("❌ Analysis failed")
                sys.exit(1)

            result = train_single_configuration(
                config_name=args.config,
                timesteps=QUICK_TEST_CONFIG['timesteps'],
                seed=QUICK_TEST_CONFIG['seeds'][0],
                log_dir='logs/quick_test'
            )

            if result['status'] == 'success':
                print("✅ Quick test completed successfully")
                if result['eval_results']:
                    print(f"📊 Test reward: {result['eval_results']['mean_reward']:.4f}")
            else:
                print(f"❌ Quick test failed: {result.get('error', 'Unknown error')}")

        elif args.command == 'list-configs':
            # List configurations
            print("📋 Available Configurations:")
            print("=" * 50)

            for config_name in list_configurations():
                config = get_config(config_name)
                print(f"\n{config_name}:")
                print(f"  Description: {config['description']}")
                print(f"  Weights: {config['weights']}")
                print(f"  Expected: {config['expected_behavior']}")

        elif args.command == 'info':
            # System information
            print_system_info()

        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\n⚠️ Operation interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if VERBOSE_LOGGING:
            traceback.print_exc()
        sys.exit(1)


# ============================================================================
# UTILITY FUNCTIONS FOR CLI
# ============================================================================

def interactive_config_selection() -> str:
    """Interactive configuration selection for users."""
    configs = list_configurations()

    print("\n📋 Select a configuration:")
    for i, config_name in enumerate(configs, 1):
        config = get_config(config_name)
        print(f"  {i}. {config_name}: {config['description']}")

    while True:
        try:
            choice = input(f"\nEnter choice (1-{len(configs)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(configs):
                return configs[idx]
            else:
                print(f"Please enter a number between 1 and {len(configs)}")
        except (ValueError, KeyboardInterrupt):
            print("Invalid input. Please enter a number.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()