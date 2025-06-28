#!/usr/bin/env python3
# MORL_modules/agents/mosac/train_mosac.py

"""
Multi-Objective SAC Training Script for EnergyNet Environment

This script provides a complete training pipeline for MOSAC on the EnergyNet environment
with comprehensive logging, evaluation, and analysis capabilities.

Usage:
    python train_mosac.py --config configs/mosac_config.yaml
    python train_mosac.py --strategy economic_focus --total-timesteps 100000
    python train_mosac.py --pareto --archive-size 200
"""

import argparse
import os
import sys
import yaml
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import torch
import gymnasium as gym

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import MOSAC components
from MORL_modules.agents.mosac import (
    MOSAC, MOSACPolicy, create_mosac_model,
    get_objective_weights, get_preset_config,
    MOSACAnalyzer, save_training_stats
)

# Import environment and wrappers
try:
    from energy_net import EnergyNetV0
    from wrappers.mo_pcs_wrapper import MOPCSWrapper

    ENERGY_NET_AVAILABLE = True
except ImportError:
    print("Warning: EnergyNet not available, using dummy environment")
    ENERGY_NET_AVAILABLE = False


class MOSACTrainer:
    """
    Comprehensive trainer for Multi-Objective SAC
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer with configuration

        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.setup_logging()
        self.setup_directories()

        # Initialize tracking
        self.training_stats = {
            'timesteps': [],
            'objective_rewards': {obj: [] for obj in config['objective_names']},
            'episode_lengths': [],
            'evaluation_rewards': [],
            'pareto_archive': [],
        }

        self.logger.info("MOSAC Trainer initialized")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    os.path.join(self.config['log_dir'], 'training.log')
                )
            ]
        )

        self.logger = logging.getLogger("MOSACTrainer")

    def setup_directories(self):
        """Create necessary directories"""
        dirs_to_create = [
            self.config['log_dir'],
            self.config['model_dir'],
            self.config['plot_dir'],
            self.config['results_dir'],
        ]

        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)

    def create_environment(self, for_eval: bool = False) -> gym.Env:
        """
        Create and configure environment

        Args:
            for_eval: Whether environment is for evaluation

        Returns:
            Configured environment
        """
        if ENERGY_NET_AVAILABLE:
            # Create EnergyNet environment
            env_kwargs = self.config.get('env_kwargs', {})
            env = EnergyNetV0(**env_kwargs)

            # Wrap with multi-objective wrapper
            wrapper_kwargs = self.config.get('wrapper_kwargs', {})
            env = MOPCSWrapper(
                env,
                num_objectives=self.config['num_objectives'],
                **wrapper_kwargs
            )

            self.logger.info("Created EnergyNet environment with MOPCSWrapper")
        else:
            # Fallback to dummy environment for testing
            self.logger.warning("Using dummy environment (Pendulum-v1)")
            env = gym.make("Pendulum-v1")

            # Simple wrapper to provide multi-objective rewards
            class DummyMOWrapper(gym.Wrapper):
                def __init__(self, env, num_objectives=4):
                    super().__init__(env)
                    self.num_objectives = num_objectives

                def step(self, action):
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    # Convert scalar reward to vector
                    mo_reward = np.random.randn(self.num_objectives) * 0.1 + reward
                    return obs, mo_reward, terminated, truncated, info

            env = DummyMOWrapper(env, self.config['num_objectives'])

        # Set seed
        if not for_eval:
            env.reset(seed=self.config.get('seed', 42))

        return env

    def create_model(self, env: gym.Env) -> MOSAC:
        """
        Create MOSAC model

        Args:
            env: Training environment

        Returns:
            MOSAC model
        """
        model_config = {
            'num_objectives': self.config['num_objectives'],
            'mo_strategy': self.config.get('mo_strategy', 'scalarized'),
            'preference_weights': self.config.get('preference_weights'),
            'pareto_archive_size': self.config.get('pareto_archive_size', 100),
            'learning_rate': self.config.get('learning_rate', 3e-4),
            'batch_size': self.config.get('batch_size', 256),
            'buffer_size': self.config.get('buffer_size', 1_000_000),
            'gamma': self.config.get('gamma', 0.99),
            'tau': self.config.get('tau', 0.005),
            'learning_starts': self.config.get('learning_starts', 1000),
            'train_freq': self.config.get('train_freq', 1),
            'gradient_steps': self.config.get('gradient_steps', 1),
            'target_update_interval': self.config.get('target_update_interval', 1),
            'verbose': self.config.get('verbose', 1),
            'seed': self.config.get('seed', 42),
            'device': self.config.get('device', 'auto'),
            'tensorboard_log': os.path.join(self.config['log_dir'], 'tensorboard'),
        }

        # Policy kwargs
        policy_kwargs = {
            'num_objectives': self.config['num_objectives'],
            'actor_hidden_dims': self.config.get('actor_hidden_dims', [256, 256]),
            'critic_hidden_dims': self.config.get('critic_hidden_dims', [256, 256]),
            'activation_fn': torch.nn.ReLU,
            'dropout_rate': self.config.get('dropout_rate', 0.0),
        }
        model_config['policy_kwargs'] = policy_kwargs

        model = create_mosac_model(env, model_config)

        self.logger.info("Created MOSAC model")
        self.logger.info(f"Strategy: {model_config['mo_strategy']}")
        self.logger.info(f"Preference weights: {model_config['preference_weights']}")

        return model

    def evaluate_model(self, model: MOSAC, env: gym.Env, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            model: MOSAC model to evaluate
            env: Evaluation environment
            n_episodes: Number of evaluation episodes

        Returns:
            Evaluation metrics
        """

    def evaluate_model(self, model: MOSAC, env: gym.Env, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            model: MOSAC model to evaluate
            env: Evaluation environment
            n_episodes: Number of evaluation episodes

        Returns:
            Evaluation metrics
        """
        episode_rewards = {obj: [] for obj in self.config['objective_names']}
        episode_lengths = []

        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_length = 0
            episode_obj_rewards = np.zeros(self.config['num_objectives'])

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_length += 1

                # Accumulate rewards
                if isinstance(rewards, np.ndarray):
                    episode_obj_rewards += rewards
                else:
                    episode_obj_rewards[0] += rewards  # Scalar fallback

            # Store results
            episode_lengths.append(episode_length)
            for i, obj_name in enumerate(self.config['objective_names']):
                if i < len(episode_obj_rewards):
                    episode_rewards[obj_name].append(episode_obj_rewards[i])

        # Compute statistics
        eval_results = {}
        for obj_name, rewards in episode_rewards.items():
            if rewards:
                eval_results[f"eval_{obj_name}_mean"] = np.mean(rewards)
                eval_results[f"eval_{obj_name}_std"] = np.std(rewards)

        eval_results["eval_episode_length_mean"] = np.mean(episode_lengths)
        eval_results["eval_episode_length_std"] = np.std(episode_lengths)

        return eval_results

    def save_model(self, model: MOSAC, timestep: int):
        """Save model checkpoint"""
        model_path = os.path.join(
            self.config['model_dir'],
            f"mosac_checkpoint_{timestep}.zip"
        )
        model.save(model_path)
        self.logger.info(f"Model saved to {model_path}")

    def plot_training_progress(self):
        """Plot and save training progress"""
        if not self.training_stats['timesteps']:
            self.logger.warning("No training data to plot")
            return

        analyzer = MOSACAnalyzer(self.config['objective_names'])

        # Plot training progress
        fig = analyzer.plot_training_progress(
            self.training_stats,
            save_path=os.path.join(self.config['plot_dir'], 'training_progress.png')
        )
        plt.close(fig)

        # Plot Pareto front if available
        if (self.config.get('mo_strategy') == 'pareto' and
                len(self.training_stats['pareto_archive']) > 0):
            pareto_fig = analyzer.plot_pareto_front(
                self.training_stats['pareto_archive'],
                save_path=os.path.join(self.config['plot_dir'], 'pareto_front.png')
            )
            plt.close(pareto_fig)

        self.logger.info("Training plots saved")

    def save_results(self):
        """Save training results and analysis"""
        # Save training statistics
        stats_path = os.path.join(self.config['results_dir'], 'training_stats.json')
        save_training_stats(self.training_stats, stats_path)

        # Perform analysis
        analyzer = MOSACAnalyzer(self.config['objective_names'])
        analysis = analyzer.analyze_training_data(self.training_stats)

        # Save analysis
        analysis_path = os.path.join(self.config['results_dir'], 'analysis_report.json')
        analyzer.export_analysis_report(analysis, analysis_path)

        # Save configuration
        config_path = os.path.join(self.config['results_dir'], 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        self.logger.info("Results and analysis saved")

    def train(self) -> MOSAC:
        """
        Main training loop

        Returns:
            Trained MOSAC model
        """
        self.logger.info("Starting MOSAC training")

        # Create environment and model
        env = self.create_environment()
        model = self.create_model(env)

        # Create evaluation environment
        eval_env = self.create_environment(for_eval=True) if self.config.get('eval_freq', 0) > 0 else None

        # Training parameters
        total_timesteps = self.config['total_timesteps']
        save_freq = self.config.get('save_freq', 10_000)
        eval_freq = self.config.get('eval_freq', 5_000)
        log_freq = self.config.get('log_freq', 1_000)

        # Training callback for tracking
        class TrackingCallback:
            def __init__(self, trainer, eval_env, save_freq, eval_freq, log_freq):
                self.trainer = trainer
                self.eval_env = eval_env
                self.save_freq = save_freq
                self.eval_freq = eval_freq
                self.log_freq = log_freq
                self.last_save = 0
                self.last_eval = 0
                self.last_log = 0

            def __call__(self, locals_dict, globals_dict):
                model_instance = locals_dict.get('self')
                timestep = model_instance.num_timesteps

                # Update training statistics
                if hasattr(model_instance, 'objective_tracking'):
                    obj_stats = model_instance.get_objective_statistics()
                    for obj_name, stats in obj_stats.items():
                        if stats.get('episodes', 0) > 0:
                            if obj_name not in self.trainer.training_stats['objective_rewards']:
                                self.trainer.training_stats['objective_rewards'][obj_name] = []
                            self.trainer.training_stats['objective_rewards'][obj_name].append(stats['mean'])

                    self.trainer.training_stats['timesteps'].append(timestep)

                # Update Pareto archive
                if hasattr(model_instance, 'get_pareto_archive'):
                    archive = model_instance.get_pareto_archive()
                    if archive:
                        self.trainer.training_stats['pareto_archive'] = archive

                # Periodic evaluation
                if self.eval_env and timestep - self.last_eval >= self.eval_freq:
                    eval_results = self.trainer.evaluate_model(model_instance, self.eval_env)
                    self.trainer.training_stats['evaluation_rewards'].append({
                        'timestep': timestep,
                        'results': eval_results
                    })
                    self.last_eval = timestep

                    # Log evaluation results
                    self.trainer.logger.info(f"Evaluation at timestep {timestep}:")
                    for key, value in eval_results.items():
                        self.trainer.logger.info(f"  {key}: {value:.3f}")

                # Periodic saving
                if timestep - self.last_save >= self.save_freq:
                    self.trainer.save_model(model_instance, timestep)
                    self.last_save = timestep

                # Periodic logging
                if timestep - self.last_log >= self.log_freq:
                    self.trainer.logger.info(f"Training progress: {timestep}/{total_timesteps} timesteps")
                    self.last_log = timestep

                return True

        # Create callback
        callback = TrackingCallback(self, eval_env, save_freq, eval_freq, log_freq)

        try:
            # Start training
            self.logger.info(f"Training for {total_timesteps} timesteps")
            model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                log_interval=10,
                progress_bar=self.config.get('progress_bar', True),
            )

            self.logger.info("Training completed successfully!")

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

        finally:
            # Save final results
            self.save_model(model, total_timesteps)
            self.plot_training_progress()
            self.save_results()

            # Print final statistics
            if hasattr(model, 'get_objective_statistics'):
                final_stats = model.get_objective_statistics()
                self.logger.info("Final objective statistics:")
                for obj_name, stats in final_stats.items():
                    self.logger.info(f"  {obj_name}: mean={stats.get('mean', 0):.3f}, "
                                     f"std={stats.get('std', 0):.3f}")

            # Close environments
            env.close()
            if eval_env:
                eval_env.close()

        return model


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_default_config() -> Dict[str, Any]:
    """Create default configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"./logs/mosac_run_{timestamp}"

    return {
        # Environment
        'num_objectives': 4,
        'objective_names': ['economic', 'battery_health', 'grid_support', 'autonomy'],

        # Training
        'total_timesteps': 100_000,
        'learning_rate': 3e-4,
        'batch_size': 256,
        'buffer_size': 1_000_000,
        'gamma': 0.99,
        'tau': 0.005,
        'learning_starts': 1000,
        'train_freq': 1,
        'gradient_steps': 1,
        'target_update_interval': 1,

        # Multi-objective
        'mo_strategy': 'scalarized',  # 'scalarized' or 'pareto'
        'preference_weights': [0.25, 0.25, 0.25, 0.25],
        'pareto_archive_size': 100,

        # Network architecture
        'actor_hidden_dims': [256, 256],
        'critic_hidden_dims': [256, 256],
        'dropout_rate': 0.0,

        # Logging and evaluation
        'log_dir': f"{base_dir}/logs",
        'model_dir': f"{base_dir}/models",
        'plot_dir': f"{base_dir}/plots",
        'results_dir': f"{base_dir}/results",
        'save_freq': 10_000,
        'eval_freq': 5_000,
        'log_freq': 1_000,
        'log_level': 'INFO',
        'progress_bar': True,

        # System
        'seed': 42,
        'device': 'auto',
        'verbose': 1,

        # Environment specific
        'env_kwargs': {},
        'wrapper_kwargs': {},
    }


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Train Multi-Objective SAC on EnergyNet")

    # Configuration
    parser.add_argument("--config", type=str, help="Path to configuration YAML file")
    parser.add_argument("--preset", type=str, choices=['quick_test', 'standard_training', 'high_performance'],
                        help="Use preset configuration")

    # Training parameters
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=1_000_000, help="Replay buffer size")

    # Multi-objective parameters
    parser.add_argument("--strategy", type=str, choices=['balanced', 'economic_focus', 'battery_focus',
                                                         'grid_focus', 'autonomy_focus'], default='balanced',
                        help="Objective weighting strategy")
    parser.add_argument("--pareto", action='store_true', help="Use Pareto-based multi-objective training")
    parser.add_argument("--archive-size", type=int, default=100, help="Pareto archive size")

    # System parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="PyTorch device")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument("--no-progress", action='store_true', help="Disable progress bar")

    # Logging
    parser.add_argument("--log-dir", type=str, help="Log directory")
    parser.add_argument("--save-freq", type=int, default=10_000, help="Model saving frequency")
    parser.add_argument("--eval-freq", type=int, default=5_000, help="Evaluation frequency")

    args = parser.parse_args()

    # Load or create configuration
    if args.config:
        config = load_config(args.config)
    elif args.preset:
        config = get_preset_config(args.preset)
        # Update with default paths
        default_config = create_default_config()
        for key in ['log_dir', 'model_dir', 'plot_dir', 'results_dir']:
            config[key] = default_config[key]
    else:
        config = create_default_config()

    # Override with command line arguments
    if args.total_timesteps != 100_000:
        config['total_timesteps'] = args.total_timesteps
    if args.learning_rate != 3e-4:
        config['learning_rate'] = args.learning_rate
    if args.batch_size != 256:
        config['batch_size'] = args.batch_size
    if args.buffer_size != 1_000_000:
        config['buffer_size'] = args.buffer_size

    # Multi-objective configuration
    if args.pareto:
        config['mo_strategy'] = 'pareto'
        config['pareto_archive_size'] = args.archive_size
    else:
        config['mo_strategy'] = 'scalarized'
        config['preference_weights'] = get_objective_weights(args.strategy)

    # System configuration
    config['seed'] = args.seed
    config['device'] = args.device
    config['verbose'] = args.verbose
    config['progress_bar'] = not args.no_progress

    # Logging configuration
    if args.log_dir:
        config['log_dir'] = args.log_dir
        config['model_dir'] = os.path.join(args.log_dir, 'models')
        config['plot_dir'] = os.path.join(args.log_dir, 'plots')
        config['results_dir'] = os.path.join(args.log_dir, 'results')

    if args.save_freq != 10_000:
        config['save_freq'] = args.save_freq
    if args.eval_freq != 5_000:
        config['eval_freq'] = args.eval_freq

    # Create and run trainer
    trainer = MOSACTrainer(config)
    model = trainer.train()

    print(f"\nTraining completed! Results saved in: {config['results_dir']}")
    print(f"Final model saved in: {config['model_dir']}")


if __name__ == "__main__":
    main()