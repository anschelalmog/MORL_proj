#!/usr/bin/env python3
"""
Scalarized MORL Runner

Command-line interface for running Multi-Objective Reinforcement Learning
experiments with scalarized rewards.

Usage:
    python scalarized_run.py --config economic_only --timesteps 50000
    python scalarized_run.py --config all --seed 42
    python scalarized_run.py --help
"""

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

from MORL_modules.scripts.scalarized.scalarized_trainer import (
    train_configuration, train_all_configurations,
    get_configuration_list, validate_configuration, CONFIGURATIONS, DEFAULT_PARAMS)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""

    parser = argparse.ArgumentParser(
        description="Scalarized MORL Training Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single configuration
  python scalarized_run.py --config economic_only

  # Train all configurations  
  python scalarized_run.py --config all

  # Custom parameters
  python scalarized_run.py --config balanced --timesteps 100000 --seed 123

  # List available configurations
  python scalarized_run.py --list-configs

Available Configurations:
  economic_only    - Pure Economic Optimization [1, 0, 0, 0]
  economic_battery - Economic + Battery Health [0.5, 0.5, 0, 0]  
  balanced         - Balanced Multi-Objective [0.25, 0.25, 0.25, 0.25]
  all              - Run all three configurations
        """
    )

    parser.add_argument(
        '--config',
        choices=get_configuration_list() + ['all'],
        default='balanced',
        help='Configuration to train (default: balanced)'
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
        '--list-configs',
        action='store_true',
        help='List available configurations and exit'
    )

    return parser


def print_header():
    """Print application header."""
    print("=" * 70)
    print("🤖 SCALARIZED MORL TRAINING SYSTEM")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEFAULT_PARAMS['device']}")
    print("=" * 70)


def print_configurations():
    """Print available configurations."""
    print("\n📋 AVAILABLE CONFIGURATIONS")
    print("=" * 50)

    for config_name, config_info in CONFIGURATIONS.items():
        weights = config_info['weights']
        description = config_info['description']

        print(f"\n{config_name}:")
        print(f"  Description: {description}")
        print(f"  Weights: {weights}")
        print(f"  Focus: {_describe_focus(weights)}")


def _describe_focus(weights):
    """Describe the focus based on weights."""
    if weights == [1.0, 0.0, 0.0, 0.0]:
        return "Pure economic optimization"
    elif weights == [0.5, 0.5, 0.0, 0.0]:
        return "Economic performance with battery preservation"
    elif weights == [0.25, 0.25, 0.25, 0.25]:
        return "Balanced across all objectives"
    else:
        return "Custom weight distribution"


def main():
    """Main entry point."""

    parser = create_parser()
    args = parser.parse_args()

    # Handle list configurations
    if args.list_configs:
        print_configurations()
        sys.exit(0)

    print_header()

    try:
        # Validate configuration
        if not validate_configuration(args.config):
            print(f"❌ Invalid configuration: {args.config}")
            print("Available configurations:", get_configuration_list() + ['all'])
            sys.exit(1)

        # Create log directory
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Log directory: {log_dir.absolute()}")
        print(f"🎯 Configuration: {args.config}")
        print(f"⏱️ Timesteps: {args.timesteps:,}")
        print(f"🎲 Seed: {args.seed}")

        # Run training
        if args.config == 'all':
            # Train all configurations
            print("\n🚀 RUNNING ALL CONFIGURATIONS")
            print("This will train all three configurations sequentially...")

            # Ask for confirmation
            response = input("\nProceed with training all configurations? [Y/n]: ").lower().strip()
            if response in ['n', 'no']:
                print("Training cancelled.")
                sys.exit(0)

            results = train_all_configurations(
                timesteps=args.timesteps,
                seed=args.seed,
                log_dir=str(log_dir)
            )

            # Print summary
            successful_runs = [r for r in results if 'eval_results' in r]
            failed_runs = [r for r in results if 'eval_results' not in r]

            print(f"\n✅ Batch training completed!")
            print(f"   Successful: {len(successful_runs)}")
            print(f"   Failed: {len(failed_runs)}")

            if failed_runs:
                print("Failed configurations:")
                for failed in failed_runs:
                    print(f"   - {failed['config_name']}: {failed.get('error', 'Unknown error')}")

        else:
            # Train single configuration
            print(f"\n🚀 TRAINING SINGLE CONFIGURATION: {args.config}")

            config_info = CONFIGURATIONS[args.config]
            print(f"Description: {config_info['description']}")
            print(f"Weights: {config_info['weights']}")

            result = train_configuration(
                config_name=args.config,
                timesteps=args.timesteps,
                seed=args.seed,
                log_dir=str(log_dir)
            )

            if 'eval_results' in result:
                eval_results = result['eval_results']
                print(f"\n✅ Training completed successfully!")
                print(f"📊 Final performance: {eval_results['mean_reward']:.4f} ± {eval_results['std_reward']:.4f}")
                print(f"📁 Results saved to: {result['exp_dir']}")
            else:
                print(f"\n❌ Training failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\n🏁 Run completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()