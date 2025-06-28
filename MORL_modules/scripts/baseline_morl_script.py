# MORL_modules/scripts/baseline_morl_script.py

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
import json
import time

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
morl_modules_dir = os.path.dirname(current_dir)  # MORL_modules
project_root = os.path.dirname(morl_modules_dir)  # MORL_PROJ

sys.path.append(project_root)  # Add MORL_PROJ to path
sys.path.append(morl_modules_dir)  # Add MORL_modules to path

from energy_net.envs.energy_net_v0 import EnergyNetV0
from energy_net.market.pricing.cost_types import CostType
from energy_net.market.pricing.pricing_policy import PricingPolicy
from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern

from MORL_modules.agents.baseline_morl import BaselineMORLAgent
from MORL_modules.configs.baseline_morl_config import DEFAULT_CONFIG


class BaselineMORLExperiment:
    """Complete experiment pipeline for Baseline MORL."""

    def __init__(self, config=None, experiment_name: str = None):
        """Initialize experiment with configuration."""
        self.config = config if config is not None else DEFAULT_CONFIG
        self.experiment_name = experiment_name or f"baseline_morl_{int(time.time())}"

        # Setup directories
        self.results_paths = self.config.create_results_structure()
        self.experiment_dir = self.results_paths["base"] / self.experiment_name
        self.experiment_dir.mkdir(exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Initialize components
        self.agent = None
        self.training_results = None
        self.evaluation_results = None

    def setup_logging(self):
        """Setup experiment logging."""
        log_file = self.experiment_dir / "experiment.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(f"MORL_Experiment_{self.experiment_name}")
        self.logger.info(f"Starting experiment: {self.experiment_name}")

    def create_environment(self):
        """Create EnergyNet environment."""
        try:
            env = EnergyNetV0(
                controller_name=self.config.ENV_CONFIG["controller_name"],
                controller_module=self.config.ENV_CONFIG["controller_module"],
                env_config_path=self.config.ENV_CONFIG["env_config_path"],
                iso_config_path=self.config.ENV_CONFIG["iso_config_path"],
                pcs_unit_config_path=self.config.ENV_CONFIG["pcs_unit_config_path"],
                cost_type=getattr(CostType, self.config.ENV_CONFIG["cost_type"]),
                pricing_policy=getattr(PricingPolicy, self.config.ENV_CONFIG["pricing_policy"]),
                demand_pattern=getattr(DemandPattern, self.config.ENV_CONFIG["demand_pattern"])
            )
            self.logger.info("Environment created successfully")
            return env
        except Exception as e:
            self.logger.error(f"Failed to create environment: {e}")
            raise

    def train_agents(self, algorithm: str = "SAC"):
        """Train all baseline MORL agents."""
        self.logger.info(f"Starting training with algorithm: {algorithm}")

        # Create agent
        self.agent = BaselineMORLAgent(
            env_creator_fn=self.create_environment,
            algorithm=algorithm,
            weight_vectors=self.config.get_weight_vectors_list(),
            results_dir=str(self.experiment_dir),
            seed=self.config.TRAINING_CONFIG["seed"],
            verbose=self.config.TRAINING_CONFIG["verbose"]
        )

        # Train all agents
        algorithm_kwargs = self.config.get_algorithm_config(algorithm)
        self.training_results = self.agent.train_all_agents(
            total_timesteps=self.config.TRAINING_CONFIG["total_timesteps"],
            algorithm_kwargs=algorithm_kwargs
        )

        # Save training results
        with open(self.experiment_dir / "training_results.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for agent_name, result in self.training_results.items():
                json_results[agent_name] = {
                    'weights': result['weights'],
                    'model_path': result['model_path'],
                    'n_episodes': len(result.get('mo_episode_rewards', [])),
                    'n_steps': len(result.get('mo_step_rewards', []))
                }
            json.dump(json_results, f, indent=2)

        self.logger.info("Training completed successfully")

    def evaluate_agents(self):
        """Evaluate all trained agents."""
        self.logger.info("Starting agent evaluation")

        if self.agent is None:
            self.logger.error("No agent found. Run training first.")
            return

        self.evaluation_results = self.agent.evaluate_agents(
            n_eval_episodes=self.config.EVALUATION_CONFIG["n_eval_episodes"],
            deterministic=self.config.EVALUATION_CONFIG["deterministic"]
        )

        # Save evaluation results
        eval_file = self.experiment_dir / "evaluation_results.pkl"
        with open(eval_file, 'wb') as f:
            pickle.dump(self.evaluation_results, f)

        # Save summary statistics
        self.save_evaluation_summary()
        self.logger.info("Evaluation completed successfully")

    def save_evaluation_summary(self):
        """Save evaluation summary as CSV and JSON."""
        if self.evaluation_results is None:
            return

        # Create summary DataFrame
        summary_data = []
        for agent_name, result in self.evaluation_results.items():
            row = {
                'agent_name': agent_name,
                'weight_economic': result['weights'][0],
                'weight_battery': result['weights'][1],
                'weight_grid': result['weights'][2],
                'weight_autonomy': result['weights'][3],
                'mean_scalar_reward': result['mean_scalar_reward'],
                'std_scalar_reward': result['std_scalar_reward'],
                'mean_economic': result['mean_mo_rewards'][0],
                'mean_battery': result['mean_mo_rewards'][1],
                'mean_grid': result['mean_mo_rewards'][2],
                'mean_autonomy': result['mean_mo_rewards'][3],
                'std_economic': result['std_mo_rewards'][0],
                'std_battery': result['std_mo_rewards'][1],
                'std_grid': result['std_mo_rewards'][2],
                'std_autonomy': result['std_mo_rewards'][3]
            }
            summary_data.append(row)

        df = pd.DataFrame(summary_data)
        df.to_csv(self.experiment_dir / "evaluation_summary.csv", index=False)

        # Save as JSON too
        with open(self.experiment_dir / "evaluation_summary.json", 'w') as f:
            json.dump(summary_data, f, indent=2)

    def plot_rewards_over_time(self):
        """Plot 4 subplots showing each objective reward over time."""
        if self.training_results is None:
            self.logger.warning("No training results found for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=self.config.RESULTS_CONFIG["figure_size"])
        fig.suptitle('Multi-Objective Rewards Over Time', fontsize=16, fontweight='bold')

        objective_names = ['Economic', 'Battery Health', 'Grid Support', 'Autonomy']
        colors = [self.config.PLOT_CONFIG["colors"][obj.lower().replace(' ', '_')]
                  for obj in ['economic', 'battery_health', 'grid_support', 'autonomy']]

        axes = axes.flatten()

        for obj_idx, (obj_name, color) in enumerate(zip(objective_names, colors)):
            ax = axes[obj_idx]

            # Plot rewards for each agent
            for agent_name, result in self.training_results.items():
                if 'mo_step_rewards' in result and result['mo_step_rewards']:
                    # Extract objective-specific rewards
                    obj_rewards = [step_reward[obj_idx] for step_reward in result['mo_step_rewards']]

                    # Smooth rewards using rolling average
                    if len(obj_rewards) > 50:
                        window_size = min(100, len(obj_rewards) // 10)
                        obj_rewards_smooth = pd.Series(obj_rewards).rolling(window=window_size).mean()

                        # Get weights for legend
                        weights = result['weights']
                        weight_focus = weights[obj_idx]

                        # Only plot if this objective has significant weight
                        if weight_focus > 0.1:
                            alpha = 0.3 + 0.7 * weight_focus  # More transparent for lower weights
                            linewidth = 1 + 2 * weight_focus  # Thicker lines for higher weights

                            ax.plot(obj_rewards_smooth,
                                    alpha=alpha,
                                    linewidth=linewidth,
                                    label=f'w={weight_focus:.2f}')

            ax.set_title(f'{obj_name} Rewards', fontweight='bold')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Reward')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()

        # Save in both formats
        for fmt in self.config.RESULTS_CONFIG["save_formats"]:
            fig.savefig(
                self.experiment_dir / f"rewards_over_time.{fmt}",
                dpi=self.config.RESULTS_CONFIG["figure_dpi"],
                bbox_inches='tight'
            )
        plt.close()
        self.logger.info("Rewards over time plot saved")

    def plot_pareto_front(self):
        """Plot Pareto front analysis."""
        if self.evaluation_results is None:
            self.logger.warning("No evaluation results found for Pareto front plotting")
            return

        # Extract mean rewards for all agents
        rewards_data = []
        agent_names = []
        weights_data = []

        for agent_name, result in self.evaluation_results.items():
            rewards_data.append(result['mean_mo_rewards'])
            agent_names.append(agent_name)
            weights_data.append(result['weights'])

        rewards_array = np.array(rewards_data)
        weights_array = np.array(weights_data)

        # Create pairwise Pareto plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Pairwise Pareto Front Analysis', fontsize=16, fontweight='bold')

        objective_pairs = [
            (0, 1, 'Economic vs Battery Health'),
            (0, 2, 'Economic vs Grid Support'),
            (0, 3, 'Economic vs Autonomy'),
            (1, 2, 'Battery Health vs Grid Support'),
            (1, 3, 'Battery Health vs Autonomy'),
            (2, 3, 'Grid Support vs Autonomy')
        ]

        axes = axes.flatten()

        for pair_idx, (obj1, obj2, title) in enumerate(objective_pairs):
            ax = axes[pair_idx]

            # Plot all points
            scatter = ax.scatter(rewards_array[:, obj1], rewards_array[:, obj2],
                                 c=np.arange(len(rewards_array)),
                                 cmap='viridis',
                                 alpha=0.7,
                                 s=100)

            # Find and highlight Pareto front
            pareto_front = self.agent.get_pareto_front_approximation(self.evaluation_results)
            if len(pareto_front) > 0:
                ax.scatter(pareto_front[:, obj1], pareto_front[:, obj2],
                           c='red', s=150, marker='*',
                           label='Pareto Front', alpha=0.8)

            ax.set_xlabel(f'{["Economic", "Battery Health", "Grid Support", "Autonomy"][obj1]} Reward')
            ax.set_ylabel(f'{["Economic", "Battery Health", "Grid Support", "Autonomy"][obj2]} Reward')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()

        # Save plots
        for fmt in self.config.RESULTS_CONFIG["save_formats"]:
            fig.savefig(
                self.experiment_dir / f"pareto_front_analysis.{fmt}",
                dpi=self.config.RESULTS_CONFIG["figure_dpi"],
                bbox_inches='tight'
            )
        plt.close()
        self.logger.info("Pareto front analysis plot saved")

    def plot_weight_performance_heatmap(self):
        """Plot heatmap showing performance vs weight configurations."""
        if self.evaluation_results is None:
            return

        # Create weight matrix and performance matrix
        weights_list = []
        performances = []

        for result in self.evaluation_results.values():
            weights_list.append(result['weights'])
            performances.append(result['mean_mo_rewards'])

        weights_array = np.array(weights_list)
        perf_array = np.array(performances)

        fig, axes = plt.subplots(2, 2, figsize=self.config.RESULTS_CONFIG["figure_size"])
        fig.suptitle('Weight Configuration vs Performance', fontsize=16, fontweight='bold')

        objective_names = ['Economic', 'Battery Health', 'Grid Support', 'Autonomy']
        axes = axes.flatten()

        for obj_idx, obj_name in enumerate(objective_names):
            ax = axes[obj_idx]

            # Create scatter plot: weight vs performance for this objective
            scatter = ax.scatter(weights_array[:, obj_idx], perf_array[:, obj_idx],
                                 c=perf_array[:, obj_idx], cmap='viridis',
                                 s=100, alpha=0.7)

            ax.set_xlabel(f'{obj_name} Weight')
            ax.set_ylabel(f'{obj_name} Performance')
            ax.set_title(f'{obj_name}: Weight vs Performance')
            ax.grid(True, alpha=0.3)

            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Performance')

        plt.tight_layout()

        # Save plots
        for fmt in self.config.RESULTS_CONFIG["save_formats"]:
            fig.savefig(
                self.experiment_dir / f"weight_performance_heatmap.{fmt}",
                dpi=self.config.RESULTS_CONFIG["figure_dpi"],
                bbox_inches='tight'
            )
        plt.close()
        self.logger.info("Weight-performance heatmap saved")

    def plot_objective_correlation(self):
        """Plot correlation matrix between objectives."""
        if self.evaluation_results is None:
            return

        # Extract all MO rewards
        all_rewards = []
        for result in self.evaluation_results.values():
            all_rewards.extend(result['mo_rewards'])

        rewards_df = pd.DataFrame(
            all_rewards,
            columns=['Economic', 'Battery Health', 'Grid Support', 'Autonomy']
        )

        # Calculate correlation matrix
        corr_matrix = rewards_df.corr()

        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, cbar_kws={'label': 'Correlation'})
        plt.title('Objective Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save plots
        for fmt in self.config.RESULTS_CONFIG["save_formats"]:
            plt.savefig(
                self.experiment_dir / f"objective_correlation.{fmt}",
                dpi=self.config.RESULTS_CONFIG["figure_dpi"],
                bbox_inches='tight'
            )
        plt.close()
        self.logger.info("Objective correlation plot saved")

    def run_complete_experiment(self, algorithm: str = "SAC"):
        """Run complete experiment pipeline."""
        self.logger.info(f"Starting complete experiment with {algorithm}")

        try:
            # Training phase
            self.train_agents(algorithm)

            # Evaluation phase
            self.evaluate_agents()

            # Plotting phase
            self.plot_rewards_over_time()
            self.plot_pareto_front()
            self.plot_weight_performance_heatmap()
            self.plot_objective_correlation()

            # Generate final report
            self.generate_experiment_report()

            self.logger.info("Complete experiment finished successfully")

        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            raise

    def generate_experiment_report(self):
        """Generate comprehensive experiment report."""
        if self.evaluation_results is None:
            return

        report = {
            "experiment_name": self.experiment_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": {
                "algorithm": self.agent.algorithm if self.agent else "Unknown",
                "total_timesteps": self.config.TRAINING_CONFIG["total_timesteps"],
                "n_agents": len(self.evaluation_results),
                "n_eval_episodes": self.config.EVALUATION_CONFIG["n_eval_episodes"]
            },
            "results_summary": {}
        }

        # Calculate summary statistics
        all_scalar_rewards = [r['mean_scalar_reward'] for r in self.evaluation_results.values()]
        all_mo_rewards = np.array([r['mean_mo_rewards'] for r in self.evaluation_results.values()])

        report["results_summary"] = {
            "scalar_reward_stats": {
                "mean": float(np.mean(all_scalar_rewards)),
                "std": float(np.std(all_scalar_rewards)),
                "min": float(np.min(all_scalar_rewards)),
                "max": float(np.max(all_scalar_rewards))
            },
            "mo_reward_stats": {
                "mean": all_mo_rewards.mean(axis=0).tolist(),
                "std": all_mo_rewards.std(axis=0).tolist(),
                "min": all_mo_rewards.min(axis=0).tolist(),
                "max": all_mo_rewards.max(axis=0).tolist()
            },
            "pareto_front_size": len(self.agent.get_pareto_front_approximation(self.evaluation_results))
        }

        # Save report
        with open(self.experiment_dir / "experiment_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info("Experiment report generated")


def main():
    """Main function for running experiments."""
    parser = argparse.ArgumentParser(description="Run Baseline MORL Experiment")
    parser.add_argument("--algorithm", choices=["SAC", "PPO", "TD3"], default="SAC",
                        help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Training timesteps per agent")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Name for this experiment")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Update config with command line arguments
    config = DEFAULT_CONFIG
    config.TRAINING_CONFIG["total_timesteps"] = args.timesteps
    config.TRAINING_CONFIG["seed"] = args.seed
    config.EVALUATION_CONFIG["n_eval_episodes"] = args.eval_episodes

    # Create and run experiment
    experiment = BaselineMORLExperiment(
        config=config,
        experiment_name=args.experiment_name
    )

    experiment.run_complete_experiment(algorithm=args.algorithm)

    print(f"Experiment completed! Results saved to: {experiment.experiment_dir}")


if __name__ == "__main__":
    main()