#!/usr/bin/env python3
"""
MORL Analysis Module

Results analysis, visualization, and reporting for Multi-Objective
Reinforcement Learning experiments.
"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np

# Import configuration
from .settings import (
    WEIGHT_CONFIGURATIONS, OBJECTIVE_NAMES, PERFORMANCE_METRICS,
    ANALYSIS_WINDOWS, CREATE_PLOTS, SAVE_RAW_DATA, GENERATE_REPORTS,
    PLOT_INDIVIDUAL_CONFIGS, PLOT_STYLE, PLOT_PALETTE, PLOT_DPI,
    PLOT_FORMAT, PLOT_SIZES, CONFIG_COLORS, VERBOSE_LOGGING,
    DEPENDENCY_STATUS
)

# ============================================================================
# PLOTTING IMPORTS WITH FALLBACKS
# ============================================================================

if CREATE_PLOTS:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        # Set style
        plt.style.use(PLOT_STYLE)
        sns.set_palette(PLOT_PALETTE)

        DEPENDENCY_STATUS['matplotlib'] = True
        DEPENDENCY_STATUS['pandas'] = True
        DEPENDENCY_STATUS['seaborn'] = True
        print("✅ Plotting libraries available") if VERBOSE_LOGGING else None

    except ImportError as e:
        print(f"❌ Plotting libraries not available: {e}")
        DEPENDENCY_STATUS['matplotlib'] = False
        DEPENDENCY_STATUS['pandas'] = False
        DEPENDENCY_STATUS['seaborn'] = False


# ============================================================================
# RESULTS DISCOVERY AND LOADING
# ============================================================================

class ResultsAnalyzer:
    """Comprehensive analysis of MORL training results."""

    def __init__(self, results_dir: str, output_dir: str = None):
        """
        Initialize results analyzer.

        Args:
            results_dir: Directory containing experiment results
            output_dir: Directory for analysis outputs (default: results_dir/analysis)
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.results_dir / "analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiments = []
        self.analysis_data = {}
        self.summary_stats = {}

        print(f"📊 Results Analyzer Initialized")
        print(f"📁 Results directory: {self.results_dir}")
        print(f"💾 Output directory: {self.output_dir}")

    def discover_experiments(self) -> int:
        """
        Discover all experiment results in the directory.

        Returns:
            Number of experiments found
        """
        print("\n🔍 Discovering experiments...")

        self.experiments = []

        # Look for experiment directories with results
        for item in self.results_dir.rglob("*"):
            if item.is_dir():
                experiment_info = self._extract_experiment_info(item)
                if experiment_info:
                    self.experiments.append(experiment_info)

        # Group by configuration
        config_counts = {}
        for exp in self.experiments:
            config = exp['config_name'] or 'unknown'
            config_counts[config] = config_counts.get(config, 0) + 1

        print(f"✅ Found {len(self.experiments)} experiments")
        for config, count in config_counts.items():
            print(f"   {config}: {count} experiments")

        return len(self.experiments)

    def _extract_experiment_info(self, exp_dir: Path) -> Optional[Dict[str, Any]]:
        """Extract experiment information from directory."""
        monitor_file = exp_dir / "training.monitor.csv"
        config_file = exp_dir / "config.json"
        eval_file = exp_dir / "eval_results.json"

        # Must have at least monitor file
        if not monitor_file.exists():
            return None

        exp_info = {
            'name': exp_dir.name,
            'path': exp_dir,
            'monitor_file': monitor_file,
            'config_file': config_file if config_file.exists() else None,
            'eval_file': eval_file if eval_file.exists() else None,
            'config_name': None,
            'weights': None,
            'seed': None,
            'timesteps': None,
            'status': 'unknown'
        }

        # Try to extract config info from JSON
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)

                exp_info.update({
                    'config_name': config_data.get('config_name'),
                    'weights': config_data.get('weights'),
                    'seed': config_data.get('seed'),
                    'timesteps': config_data.get('timesteps'),
                    'status': 'completed'
                })

            except Exception as e:
                print(f"⚠️ Could not load config from {config_file}: {e}")

        # Try to infer config from directory name
        if not exp_info['config_name']:
            for config_name in WEIGHT_CONFIGURATIONS.keys():
                if config_name in exp_dir.name.lower():
                    exp_info['config_name'] = config_name
                    exp_info['weights'] = WEIGHT_CONFIGURATIONS[config_name]['weights']
                    break

        # Check for error files
        error_file = exp_dir / "error.json"
        if error_file.exists():
            exp_info['status'] = 'failed'

        return exp_info

    def load_training_data(self, experiment: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load training data from monitor CSV."""
        if not DEPENDENCY_STATUS['pandas']:
            return None

        try:
            # Load monitor CSV (skip header comment line)
            df = pd.read_csv(experiment['monitor_file'], skiprows=1)

            if df.empty:
                print(f"⚠️ Empty monitor file: {experiment['name']}")
                return None

            # Add experiment metadata
            df['experiment_name'] = experiment['name']
            df['config_name'] = experiment['config_name']
            df['episode'] = range(len(df))

            # Calculate additional metrics
            if 'r' in df.columns:
                df['cumulative_reward'] = df['r'].cumsum()
                df['reward_ma'] = df['r'].rolling(
                    window=ANALYSIS_WINDOWS['smoothing'],
                    min_periods=1
                ).mean()

            return df

        except Exception as e:
            print(f"❌ Could not load training data for {experiment['name']}: {e}")
            return None

    def load_evaluation_data(self, experiment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load evaluation data from eval results JSON."""
        if not experiment['eval_file'] or not experiment['eval_file'].exists():
            return None

        try:
            with open(experiment['eval_file'], 'r') as f:
                eval_data = json.load(f)
            return eval_data

        except Exception as e:
            print(f"❌ Could not load evaluation data for {experiment['name']}: {e}")
            return None

    # ============================================================================
    # PERFORMANCE ANALYSIS
    # ============================================================================

    def analyze_performance(self) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """
        Analyze performance across all experiments.

        Returns:
            Tuple of (combined_dataframe, summary_statistics)
        """
        print("\n📈 Analyzing performance...")

        if not DEPENDENCY_STATUS['pandas']:
            print("⚠️ Pandas not available, skipping detailed analysis")
            return None, None

        all_training_data = []
        all_eval_data = {}

        # Load all training data
        for exp in self.experiments:
            training_df = self.load_training_data(exp)
            if training_df is not None:
                all_training_data.append(training_df)

            eval_data = self.load_evaluation_data(exp)
            if eval_data is not None:
                all_eval_data[exp['name']] = eval_data

        if not all_training_data:
            print("❌ No training data found")
            return None, None

        # Combine all training data
        combined_df = pd.concat(all_training_data, ignore_index=True)

        # Calculate summary statistics
        summary = self._calculate_summary_statistics(combined_df, all_eval_data)

        # Store for later use
        self.analysis_data = {
            'combined_df': combined_df,
            'eval_data': all_eval_data
        }
        self.summary_stats = summary

        # Save summary if enabled
        if SAVE_RAW_DATA:
            summary_file = self.output_dir / "performance_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"📊 Performance summary saved to: {summary_file}")

        # Print summary to console
        self._print_performance_summary(summary)

        return combined_df, summary

    def _calculate_summary_statistics(self, combined_df: pd.DataFrame, eval_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics."""
        summary = {}

        for config_name in combined_df['config_name'].unique():
            if pd.isna(config_name):
                continue

            config_data = combined_df[combined_df['config_name'] == config_name]

            # Basic training statistics
            stats = {
                'num_experiments': len(config_data['experiment_name'].unique()),
                'total_episodes': len(config_data),
                'weights': WEIGHT_CONFIGURATIONS.get(config_name, {}).get('weights', 'unknown')
            }

            if 'r' in config_data.columns:
                # Episode reward statistics
                stats.update({
                    'mean_episode_reward': float(config_data['r'].mean()),
                    'std_episode_reward': float(config_data['r'].std()),
                    'min_episode_reward': float(config_data['r'].min()),
                    'max_episode_reward': float(config_data['r'].max()),
                    'median_episode_reward': float(config_data['r'].median())
                })

                # Final performance (last episode of each experiment)
                final_rewards = config_data.groupby('experiment_name')['r'].last()
                stats.update({
                    'mean_final_reward': float(final_rewards.mean()),
                    'std_final_reward': float(final_rewards.std()),
                    'min_final_reward': float(final_rewards.min()),
                    'max_final_reward': float(final_rewards.max())
                })

                # Convergence analysis
                stats.update(self._analyze_convergence(config_data))

            # Add evaluation statistics if available
            config_eval_data = [
                eval_data[exp] for exp in eval_data.keys()
                if config_name in exp
            ]
            if config_eval_data:
                stats.update(self._analyze_evaluation_data(config_eval_data))

            summary[config_name] = stats

        return summary

    def _analyze_convergence(self, config_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze training convergence."""
        convergence_stats = {}

        try:
            # Calculate rolling mean for convergence detection
            window = ANALYSIS_WINDOWS['convergence']
            rolling_mean = config_data.groupby('experiment_name')['r'].rolling(
                window=window, min_periods=1
            ).mean()

            # Simple convergence metric: episode where variance stabilizes
            convergence_episodes = []
            for exp_name in config_data['experiment_name'].unique():
                exp_data = config_data[config_data['experiment_name'] == exp_name]
                if len(exp_data) > window * 2:
                    # Find where rolling variance becomes stable
                    rolling_var = exp_data['r'].rolling(window=window).var()
                    stable_var = rolling_var.iloc[-window:].mean()

                    # Find first episode where variance is close to final variance
                    convergence_point = None
                    for i, var in enumerate(rolling_var.iloc[window:]):
                        if abs(var - stable_var) < stable_var * 0.1:
                            convergence_point = i + window
                            break

                    if convergence_point:
                        convergence_episodes.append(convergence_point)

            if convergence_episodes:
                convergence_stats.update({
                    'mean_convergence_episode': float(np.mean(convergence_episodes)),
                    'std_convergence_episode': float(np.std(convergence_episodes))
                })

        except Exception as e:
            print(f"⚠️ Convergence analysis failed: {e}")

        return convergence_stats

    def _analyze_evaluation_data(self, eval_data_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze evaluation data."""
        eval_stats = {}

        try:
            if eval_data_list:
                eval_rewards = [data['mean_reward'] for data in eval_data_list if 'mean_reward' in data]

                if eval_rewards:
                    eval_stats.update({
                        'eval_mean_reward': float(np.mean(eval_rewards)),
                        'eval_std_reward': float(np.std(eval_rewards)),
                        'eval_min_reward': float(np.min(eval_rewards)),
                        'eval_max_reward': float(np.max(eval_rewards))
                    })

        except Exception as e:
            print(f"⚠️ Evaluation analysis failed: {e}")

        return eval_stats

    def _print_performance_summary(self, summary: Dict[str, Any]):
        """Print performance summary to console."""
        print("\n📊 PERFORMANCE SUMMARY")
        print("=" * 50)

        for config_name, stats in summary.items():
            print(f"\n{config_name.replace('_', ' ').title()}:")
            print(f"  Weights: {stats['weights']}")
            print(f"  Experiments: {stats['num_experiments']}")

            if 'mean_final_reward' in stats:
                print(f"  Final reward: {stats['mean_final_reward']:.4f} ± {stats['std_final_reward']:.4f}")
                print(f"  Episode reward: {stats['mean_episode_reward']:.4f} ± {stats['std_episode_reward']:.4f}")
                print(f"  Range: [{stats['min_episode_reward']:.4f}, {stats['max_episode_reward']:.4f}]")

            if 'eval_mean_reward' in stats:
                print(f"  Eval reward: {stats['eval_mean_reward']:.4f} ± {stats['eval_std_reward']:.4f}")

            if 'mean_convergence_episode' in stats:
                print(f"  Convergence: ~{stats['mean_convergence_episode']:.0f} episodes")

    # ============================================================================
    # PLOTTING FUNCTIONS
    # ============================================================================

    def create_plots(self, combined_df: pd.DataFrame = None, summary: Dict[str, Any] = None):
        """Create comprehensive analysis plots."""
        if not DEPENDENCY_STATUS['matplotlib'] or not CREATE_PLOTS:
            print("⚠️ Plotting disabled or unavailable")
            return

        print("\n📈 Creating plots...")

        try:
            if combined_df is None or summary is None:
                combined_df, summary = self.analyze_performance()
                if combined_df is None:
                    return

            # Create comprehensive multi-panel plot
            self._create_comprehensive_plot(combined_df, summary)

            # Create individual plots if enabled
            if PLOT_INDIVIDUAL_CONFIGS:
                self._create_individual_plots(combined_df)

            # Create comparison plots
            self._create_comparison_plots(combined_df, summary)

        except Exception as e:
            print(f"❌ Plotting failed: {e}")
            if VERBOSE_LOGGING:
                traceback.print_exc()

    def _create_comprehensive_plot(self, combined_df: pd.DataFrame, summary: Dict[str, Any]):
        """Create comprehensive multi-panel analysis plot."""
        fig = plt.figure(figsize=PLOT_SIZES['comprehensive'])

        # Plot 1: Learning curves by configuration
        ax1 = plt.subplot(2, 3, 1)
        self._plot_learning_curves(ax1, combined_df)

        # Plot 2: Final performance comparison
        ax2 = plt.subplot(2, 3, 2)
        self._plot_final_performance(ax2, summary)

        # Plot 3: Reward distributions
        ax3 = plt.subplot(2, 3, 3)
        self._plot_reward_distributions(ax3, combined_df)

        # Plot 4: Training convergence
        ax4 = plt.subplot(2, 3, 4)
        self._plot_convergence(ax4, combined_df)

        # Plot 5: Configuration weights
        ax5 = plt.subplot(2, 3, 5)
        self._plot_weight_configurations(ax5, summary)

        # Plot 6: Performance metrics summary
        ax6 = plt.subplot(2, 3, 6)
        self._plot_performance_metrics(ax6, summary)

        plt.tight_layout()

        # Save plot
        plot_file = self.output_dir / f"comprehensive_analysis.{PLOT_FORMAT}"
        plt.savefig(plot_file, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()

        print(f"📈 Comprehensive analysis plot saved to: {plot_file}")

    def _plot_learning_curves(self, ax, combined_df: pd.DataFrame):
        """Plot learning curves for each configuration."""
        for config_name in combined_df['config_name'].unique():
            if pd.isna(config_name):
                continue

            config_data = combined_df[combined_df['config_name'] == config_name]
            color = CONFIG_COLORS.get(config_name, None)

            # Plot individual runs with transparency
            for exp_name in config_data['experiment_name'].unique():
                exp_data = config_data[config_data['experiment_name'] == exp_name]
                ax.plot(exp_data['episode'], exp_data['r'],
                        alpha=0.3, linewidth=1, color=color)

            # Plot mean learning curve
            if 'reward_ma' in config_data.columns:
                mean_curve = config_data.groupby('episode')['reward_ma'].mean()
                ax.plot(mean_curve.index, mean_curve.values,
                        label=config_name.replace('_', ' ').title(),
                        linewidth=3, color=color)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Learning Curves by Configuration')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_final_performance(self, ax, summary: Dict[str, Any]):
        """Plot final performance comparison."""
        configs = list(summary.keys())
        final_means = [summary[c].get('mean_final_reward', 0) for c in configs]
        final_stds = [summary[c].get('std_final_reward', 0) for c in configs]

        colors = [CONFIG_COLORS.get(c, 'gray') for c in configs]
        bars = ax.bar(configs, final_means, yerr=final_stds,
                      capsize=5, alpha=0.7, color=colors)

        ax.set_xlabel('Configuration')
        ax.set_ylabel('Final Reward')
        ax.set_title('Final Performance Comparison')
        ax.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, mean_val in zip(bars, final_means):
            if mean_val != 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{mean_val:.3f}', ha='center', va='bottom')

    def _plot_reward_distributions(self, ax, combined_df: pd.DataFrame):
        """Plot reward distributions as box plots."""
        config_rewards = []
        config_labels = []

        for config_name in combined_df['config_name'].unique():
            if pd.isna(config_name):
                continue
            config_data = combined_df[combined_df['config_name'] == config_name]
            config_rewards.append(config_data['r'].values)
            config_labels.append(config_name.replace('_', ' ').title())

        if config_rewards:
            bp = ax.boxplot(config_rewards, labels=config_labels, patch_artist=True)

            # Color boxes
            for patch, config_name in zip(bp['boxes'], combined_df['config_name'].unique()):
                if not pd.isna(config_name):
                    color = CONFIG_COLORS.get(config_name, 'gray')
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

        ax.set_xlabel('Configuration')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Distributions')
        ax.tick_params(axis='x', rotation=45)

    def _plot_convergence(self, ax, combined_df: pd.DataFrame):
        """Plot training convergence analysis."""
        for config_name in combined_df['config_name'].unique():
            if pd.isna(config_name):
                continue

            config_data = combined_df[combined_df['config_name'] == config_name]
            color = CONFIG_COLORS.get(config_name, None)

            # Calculate rolling mean for convergence analysis
            window = max(10, len(config_data) // 20)
            if 'reward_ma' in config_data.columns:
                rolling_mean = config_data.groupby('episode')['reward_ma'].mean()
                ax.plot(rolling_mean.index, rolling_mean.values,
                        label=f'{config_name.replace("_", " ").title()}',
                        linewidth=2, color=color)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Smoothed Reward')
        ax.set_title('Training Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_weight_configurations(self, ax, summary: Dict[str, Any]):
        """Plot weight configurations as grouped bar chart."""
        configs = list(summary.keys())

        x = np.arange(len(OBJECTIVE_NAMES))
        width = 0.8 / len(configs)

        for i, config_name in enumerate(configs):
            weights = summary[config_name]['weights']
            if weights != 'unknown':
                color = CONFIG_COLORS.get(config_name, 'gray')
                ax.bar(x + i * width, weights, width,
                       label=config_name.replace('_', ' ').title(),
                       alpha=0.7, color=color)

        ax.set_xlabel('Objective')
        ax.set_ylabel('Weight')
        ax.set_title('Weight Configurations')
        ax.set_xticks(x + width * (len(configs) - 1) / 2)
        ax.set_xticklabels(OBJECTIVE_NAMES, rotation=45)
        ax.legend()

    def _plot_performance_metrics(self, ax, summary: Dict[str, Any]):
        """Plot performance metrics comparison."""
        configs = list(summary.keys())

        metrics = ['mean_final_reward', 'mean_episode_reward']
        metric_labels = ['Final Reward', 'Mean Reward']

        x = np.arange(len(configs))
        width = 0.35

        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [summary[c].get(metric, 0) for c in configs]
            ax.bar(x + i * width, values, width, label=label, alpha=0.7)

        ax.set_xlabel('Configuration')
        ax.set_ylabel('Reward')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([c.replace('_', ' ').title() for c in configs], rotation=45)
        ax.legend()

    def _create_individual_plots(self, combined_df: pd.DataFrame):
        """Create individual plots for each configuration."""
        for config_name in combined_df['config_name'].unique():
            if pd.isna(config_name):
                continue

            try:
                config_data = combined_df[combined_df['config_name'] == config_name]

                plt.figure(figsize=PLOT_SIZES['individual'])

                # Plot each experiment run
                for exp_name in config_data['experiment_name'].unique():
                    exp_data = config_data[config_data['experiment_name'] == exp_name]
                    plt.plot(exp_data['episode'], exp_data['r'], alpha=0.6, linewidth=1)

                # Plot mean with confidence interval
                mean_curve = config_data.groupby('episode')['r'].mean()
                std_curve = config_data.groupby('episode')['r'].std()

                plt.plot(mean_curve.index, mean_curve.values,
                         'k-', linewidth=3, label='Mean')

                plt.fill_between(mean_curve.index,
                                 mean_curve.values - std_curve.values,
                                 mean_curve.values + std_curve.values,
                                 alpha=0.3, label='±1 std')

                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.title(f'{config_name.replace("_", " ").title()} - Training Progress')
                plt.legend()
                plt.grid(True, alpha=0.3)

                # Save individual plot
                plot_file = self.output_dir / f"{config_name}_learning_curve.{PLOT_FORMAT}"
                plt.savefig(plot_file, dpi=PLOT_DPI, bbox_inches='tight')
                plt.close()

            except Exception as e:
                print(f"⚠️ Could not create individual plot for {config_name}: {e}")

    def _create_comparison_plots(self, combined_df: pd.DataFrame, summary: Dict[str, Any]):
        """Create additional comparison plots."""
        try:
            # Configuration comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=PLOT_SIZES['comparison'])

            # Plot 1: Performance vs Weight Distribution
            configs = list(summary.keys())
            performances = [summary[c].get('mean_final_reward', 0) for c in configs]

            # Calculate weight diversity (entropy)
            diversities = []
            for config in configs:
                weights = summary[config]['weights']
                if weights != 'unknown':
                    # Normalize weights
                    w = np.array(weights)
                    w = w / w.sum() if w.sum() > 0 else w
                    # Calculate entropy
                    entropy = -np.sum(w * np.log(w + 1e-10))
                    diversities.append(entropy)
                else:
                    diversities.append(0)

            scatter = ax1.scatter(diversities, performances,
                                  c=[CONFIG_COLORS.get(c, 'gray') for c in configs],
                                  s=100, alpha=0.7)

            for i, config in enumerate(configs):
                ax1.annotate(config.replace('_', ' ').title(),
                             (diversities[i], performances[i]),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)

            ax1.set_xlabel('Weight Diversity (Entropy)')
            ax1.set_ylabel('Final Performance')
            ax1.set_title('Performance vs Weight Diversity')
            ax1.grid(True, alpha=0.3)

            # Plot 2: Learning Efficiency (performance per episode)
            episodes_to_converge = []
            final_performances = []

            for config in configs:
                if 'mean_convergence_episode' in summary[config]:
                    episodes_to_converge.append(summary[config]['mean_convergence_episode'])
                    final_performances.append(summary[config]['mean_final_reward'])

            if episodes_to_converge and final_performances:
                ax2.scatter(episodes_to_converge, final_performances,
                            c=[CONFIG_COLORS.get(c, 'gray') for c in configs[:len(episodes_to_converge)]],
                            s=100, alpha=0.7)

                for i, config in enumerate(configs[:len(episodes_to_converge)]):
                    ax2.annotate(config.replace('_', ' ').title(),
                                 (episodes_to_converge[i], final_performances[i]),
                                 xytext=(5, 5), textcoords='offset points', fontsize=8)

                ax2.set_xlabel('Episodes to Convergence')
                ax2.set_ylabel('Final Performance')
                ax2.set_title('Learning Efficiency')
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_file = self.output_dir / f"comparison_analysis.{PLOT_FORMAT}"
            plt.savefig(plot_file, dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()

            print(f"📈 Comparison plots saved to: {plot_file}")

        except Exception as e:
            print(f"⚠️ Could not create comparison plots: {e}")

    # ============================================================================
    # REPORT GENERATION
    # ============================================================================

    def generate_report(self, summary: Dict[str, Any] = None):
        """Generate comprehensive analysis report."""
        if not GENERATE_REPORTS:
            return

        if summary is None:
            _, summary = self.analyze_performance()
            if summary is None:
                return

        report_file = self.output_dir / "analysis_report.md"

        try:
            with open(report_file, 'w') as f:
                f.write(self._generate_report_content(summary))

            print(f"📄 Analysis report saved to: {report_file}")

        except Exception as e:
            print(f"❌ Report generation failed: {e}")
            if VERBOSE_LOGGING:
                traceback.print_exc()

    def _generate_report_content(self, summary: Dict[str, Any]) -> str:
        """Generate the content of the analysis report."""

        # Find best configuration
        best_config = None
        best_performance = -float('inf')

        for config_name, stats in summary.items():
            if 'mean_final_reward' in stats:
                if stats['mean_final_reward'] > best_performance:
                    best_performance = stats['mean_final_reward']
                    best_config = config_name

        report_content = f"""# MORL Training Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Results Directory:** {self.results_dir}  
**Total Experiments:** {len(self.experiments)}

## Executive Summary

"""

        if best_config and summary:
            report_content += f"""
- **Best Performing Configuration:** {best_config.replace('_', ' ').title()}
- **Best Final Reward:** {best_performance:.4f}
- **Configurations Tested:** {len(summary)}
- **Total Experiments:** {sum(stats['num_experiments'] for stats in summary.values())}

## Configuration Results

"""

            for config_name, stats in summary.items():
                weights = stats['weights']
                config_info = WEIGHT_CONFIGURATIONS.get(config_name, {})
                description = config_info.get('description', 'Unknown')
                expected = config_info.get('expected_behavior', 'Unknown')

                report_content += f"""
### {config_name.replace('_', ' ').title()}

- **Description:** {description}
- **Objective Weights:** {weights}
- **Expected Behavior:** {expected}
- **Number of Experiments:** {stats['num_experiments']}
"""

                if 'mean_final_reward' in stats:
                    report_content += f"""- **Final Performance:** {stats['mean_final_reward']:.4f} ± {stats['std_final_reward']:.4f}
- **Episode Performance:** {stats['mean_episode_reward']:.4f} ± {stats['std_episode_reward']:.4f}
- **Performance Range:** [{stats['min_episode_reward']:.4f}, {stats['max_episode_reward']:.4f}]
"""

                if 'eval_mean_reward' in stats:
                    report_content += f"- **Evaluation Performance:** {stats['eval_mean_reward']:.4f} ± {stats['eval_std_reward']:.4f}\n"

                if 'mean_convergence_episode' in stats:
                    report_content += f"- **Convergence:** ~{stats['mean_convergence_episode']:.0f} episodes\n"

                report_content += "\n"

        report_content += f"""
## System Configuration

- **Algorithm:** {'MOSAC' if DEPENDENCY_STATUS.get('mosac', False) else 'SAC'}
- **Environment Wrapper:** {'ScalarizedMOPCSWrapper' if DEPENDENCY_STATUS.get('scalarized_wrapper', False) else 'Base Environment'}
- **Plotting:** {'Enabled' if DEPENDENCY_STATUS.get('matplotlib', False) else 'Disabled'}

## Dependency Status

"""

        for dep, status in DEPENDENCY_STATUS.items():
            status_icon = "✅" if status else "❌"
            report_content += f"- **{dep}:** {status_icon}\n"

        report_content += f"""

## Key Findings

1. **Best Configuration:** {best_config.replace('_', ' ').title() if best_config else 'Unknown'} achieved the highest performance
2. **Training Stability:** Analysis of reward variance across episodes shows training consistency
3. **Convergence Patterns:** Different configurations show varying convergence speeds

## Recommendations

1. **For Maximum Performance:** Use {best_config.replace('_', ' ').title() if best_config else 'the best-performing'} configuration for production deployments
2. **For Stability:** Consider configurations with lower variance in performance
3. **For Future Research:** Investigate intermediate weight combinations between top performers

## Generated Files

- `performance_summary.json` - Numerical performance data
- `comprehensive_analysis.{PLOT_FORMAT}` - Multi-panel comparison plots
- `comparison_analysis.{PLOT_FORMAT}` - Additional comparison visualizations
- `{{config_name}}_learning_curve.{PLOT_FORMAT}` - Individual configuration plots
- `analysis_report.md` - This report

---
*Generated by MORL Analysis Module v1.0*
"""

        return report_content

    # ============================================================================
    # MAIN ANALYSIS PIPELINE
    # ============================================================================

    def run_full_analysis(self) -> bool:
        """
        Run complete analysis pipeline.

        Returns:
            True if analysis completed successfully
        """
        try:
            # Discover experiments
            num_experiments = self.discover_experiments()

            if num_experiments == 0:
                print("❌ No experiments found to analyze")
                return False

            # Analyze performance
            combined_df, summary = self.analyze_performance()

            if summary is None:
                print("❌ Performance analysis failed")
                return False

            # Create plots
            self.create_plots(combined_df, summary)

            # Generate report
            self.generate_report(summary)

            print(f"\n✅ Analysis complete! Results saved to: {self.output_dir}")
            return True

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            if VERBOSE_LOGGING:
                traceback.print_exc()
            return False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def analyze_results_directory(results_dir: str, output_dir: str = None) -> bool:
    """
    Convenience function to analyze results directory.

    Args:
        results_dir: Directory containing experiment results
        output_dir: Output directory for analysis

    Returns:
        True if analysis completed successfully
    """
    analyzer = ResultsAnalyzer(results_dir, output_dir)
    return analyzer.run_full_analysis()


def compare_configurations(results_dir: str, config_names: List[str] = None) -> Dict[str, Any]:
    """
    Compare specific configurations.

    Args:
        results_dir: Directory containing experiment results
        config_names: List of configuration names to compare

    Returns:
        Comparison results
    """
    analyzer = ResultsAnalyzer(results_dir)
    analyzer.discover_experiments()

    # Filter experiments by configuration names
    if config_names:
        analyzer.experiments = [
            exp for exp in analyzer.experiments
            if exp['config_name'] in config_names
        ]

    _, summary = analyzer.analyze_performance()
    return summary


if __name__ == "__main__":
    # Test analysis functionality
    print("📊 MORL Analysis Module")
    print("=" * 50)

    # Check dependencies
    print("📦 Dependency Status:")
    for dep, status in DEPENDENCY_STATUS.items():
        status_icon = "✅" if status else "❌"
        print(f"  {dep}: {status_icon}")

    print("\nAnalysis module loaded successfully!")