# MORL_modules/agents/mosac/utils.py

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import json
import os
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr


class ParetoArchive:
    """
    Maintains a Pareto archive of non-dominated solutions.
    Used for true multi-objective optimization.
    """

    def __init__(self, max_size: int = 100, objectives_dim: int = 4):
        """
        Initialize Pareto archive

        Args:
            max_size: Maximum number of solutions to keep
            objectives_dim: Number of objectives
        """
        self.max_size = max_size
        self.objectives_dim = objectives_dim
        self.archive = []  # List of objective vectors
        self.policies = []  # Corresponding policy parameters (optional)

    def add_solution(self, objectives: np.ndarray, policy_params: Optional[Dict] = None) -> bool:
        """
        Add a solution to the archive if it's non-dominated

        Args:
            objectives: Objective values [objectives_dim]
            policy_params: Optional policy parameters

        Returns:
            True if solution was added, False otherwise
        """
        objectives = np.array(objectives)

        # Check if solution is dominated by existing solutions
        if self._is_dominated(objectives):
            return False

        # Remove dominated solutions
        self._remove_dominated_by(objectives)

        # Add new solution
        self.archive.append(objectives.copy())
        if policy_params is not None:
            self.policies.append(policy_params.copy())
        else:
            self.policies.append(None)

        # Maintain size limit
        if len(self.archive) > self.max_size:
            self._prune_archive()

        return True

    def _is_dominated(self, objectives: np.ndarray) -> bool:
        """Check if objectives are dominated by any solution in archive"""
        for archive_obj in self.archive:
            if self._dominates(archive_obj, objectives):
                return True
        return False

    def _remove_dominated_by(self, objectives: np.ndarray):
        """Remove solutions dominated by the new objectives"""
        indices_to_remove = []
        for i, archive_obj in enumerate(self.archive):
            if self._dominates(objectives, archive_obj):
                indices_to_remove.append(i)

        # Remove in reverse order to maintain indices
        for i in reversed(indices_to_remove):
            self.archive.pop(i)
            self.policies.pop(i)

    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2 (assuming maximization)"""
        return np.all(obj1 >= obj2) and np.any(obj1 > obj2)

    def _prune_archive(self):
        """Prune archive to max_size using diversity preservation"""
        if len(self.archive) <= self.max_size:
            return

        # Convert to numpy array for easier manipulation
        objectives_array = np.array(self.archive)

        # Use crowding distance to maintain diversity
        distances = self._calculate_crowding_distances(objectives_array)

        # Keep solutions with highest crowding distances
        keep_indices = np.argsort(distances)[-self.max_size:]

        self.archive = [self.archive[i] for i in keep_indices]
        self.policies = [self.policies[i] for i in keep_indices]

    def _calculate_crowding_distances(self, objectives: np.ndarray) -> np.ndarray:
        """Calculate crowding distances for diversity preservation"""
        n_solutions = objectives.shape[0]
        distances = np.zeros(n_solutions)

        for obj_idx in range(self.objectives_dim):
            # Sort by objective value
            sorted_indices = np.argsort(objectives[:, obj_idx])

            # Boundary solutions get infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')

            # Calculate distances for intermediate solutions
            obj_range = objectives[sorted_indices[-1], obj_idx] - objectives[sorted_indices[0], obj_idx]
            if obj_range > 0:
                for i in range(1, n_solutions - 1):
                    distances[sorted_indices[i]] += (
                                                            objectives[sorted_indices[i + 1], obj_idx] -
                                                            objectives[sorted_indices[i - 1], obj_idx]
                                                    ) / obj_range

        return distances

    def get_archive(self) -> Tuple[List[np.ndarray], List[Optional[Dict]]]:
        """Get current archive"""
        return self.archive.copy(), self.policies.copy()

    def size(self) -> int:
        """Get archive size"""
        return len(self.archive)

    def is_empty(self) -> bool:
        """Check if archive is empty"""
        return len(self.archive) == 0


class MOSACMetrics:
    """
    Utility class for computing multi-objective metrics
    """

    @staticmethod
    def hypervolume(pareto_front: np.ndarray, reference_point: np.ndarray) -> float:
        """
        Compute hypervolume indicator for a Pareto front

        Args:
            pareto_front: Array of objective vectors [n_solutions, n_objectives]
            reference_point: Reference point for hypervolume calculation

        Returns:
            Hypervolume value
        """
        # Simple 2D hypervolume calculation
        if pareto_front.shape[1] == 2:
            return MOSACMetrics._hypervolume_2d(pareto_front, reference_point)
        else:
            # For higher dimensions, use Monte Carlo approximation
            return MOSACMetrics._hypervolume_monte_carlo(pareto_front, reference_point)

    @staticmethod
    def _hypervolume_2d(pareto_front: np.ndarray, reference_point: np.ndarray) -> float:
        """Exact hypervolume calculation for 2D"""
        if len(pareto_front) == 0:
            return 0.0

        # Sort by first objective
        sorted_front = pareto_front[np.argsort(pareto_front[:, 0])]

        volume = 0.0
        prev_x = reference_point[0]

        for point in sorted_front:
            if point[0] > prev_x and point[1] > reference_point[1]:
                volume += (point[0] - prev_x) * (point[1] - reference_point[1])
                prev_x = point[0]

        return volume

    @staticmethod
    def _hypervolume_monte_carlo(pareto_front: np.ndarray, reference_point: np.ndarray,
                                 n_samples: int = 10000) -> float:
        """Monte Carlo hypervolume approximation for higher dimensions"""
        if len(pareto_front) == 0:
            return 0.0

        # Find bounds
        max_point = np.max(pareto_front, axis=0)

        # Generate random points
        n_objectives = len(reference_point)
        random_points = np.random.uniform(
            low=reference_point,
            high=max_point,
            size=(n_samples, n_objectives)
        )

        # Count dominated points
        dominated_count = 0
        for point in random_points:
            if MOSACMetrics._is_dominated_by_front(point, pareto_front):
                dominated_count += 1

        # Calculate volume
        total_volume = np.prod(max_point - reference_point)
        return total_volume * (dominated_count / n_samples)

    @staticmethod
    def _is_dominated_by_front(point: np.ndarray, pareto_front: np.ndarray) -> bool:
        """Check if point is dominated by any solution in Pareto front"""
        for front_point in pareto_front:
            if np.all(front_point >= point):
                return True
        return False

    @staticmethod
    def spacing_metric(pareto_front: np.ndarray) -> float:
        """
        Compute spacing metric for distribution quality

        Args:
            pareto_front: Array of objective vectors

        Returns:
            Spacing metric (lower is better)
        """
        if len(pareto_front) < 2:
            return 0.0

        # Calculate distances between consecutive points
        distances = pdist(pareto_front)
        mean_distance = np.mean(distances)

        # Calculate variance in distances
        variance = np.mean((distances - mean_distance) ** 2)

        return np.sqrt(variance)

    @staticmethod
    def coverage_metric(pareto_front_a: np.ndarray, pareto_front_b: np.ndarray) -> float:
        """
        Compute coverage metric between two Pareto fronts

        Args:
            pareto_front_a: First Pareto front
            pareto_front_b: Second Pareto front

        Returns:
            Coverage of A over B (fraction of B dominated by A)
        """
        if len(pareto_front_b) == 0:
            return 1.0 if len(pareto_front_a) > 0 else 0.0

        dominated_count = 0
        for point_b in pareto_front_b:
            for point_a in pareto_front_a:
                if np.all(point_a >= point_b) and np.any(point_a > point_b):
                    dominated_count += 1
                    break

        return dominated_count / len(pareto_front_b)


class MOSACAnalyzer:
    """
    Analysis utilities for MOSAC training results
    """

    def __init__(self, objective_names: List[str] = None):
        """
        Initialize analyzer

        Args:
            objective_names: Names of objectives
        """
        self.objective_names = objective_names or ["economic", "battery_health", "grid_support", "autonomy"]

    def analyze_training_data(self, training_stats: Dict) -> Dict:
        """
        Analyze training statistics

        Args:
            training_stats: Training statistics dictionary

        Returns:
            Analysis results
        """
        analysis = {}

        # Objective correlations
        if 'objective_rewards' in training_stats:
            correlations = self._compute_objective_correlations(training_stats['objective_rewards'])
            analysis['objective_correlations'] = correlations

        # Convergence analysis
        if 'timesteps' in training_stats and 'objective_rewards' in training_stats:
            convergence = self._analyze_convergence(
                training_stats['timesteps'],
                training_stats['objective_rewards']
            )
            analysis['convergence'] = convergence

        # Performance summary
        if 'objective_rewards' in training_stats:
            performance = self._summarize_performance(training_stats['objective_rewards'])
            analysis['performance_summary'] = performance

        return analysis

    def _compute_objective_correlations(self, objective_rewards: Dict[str, List]) -> Dict:
        """Compute correlations between objectives"""
        correlations = {}

        # Get objective data
        obj_data = {}
        for obj_name in self.objective_names:
            if obj_name in objective_rewards and len(objective_rewards[obj_name]) > 1:
                obj_data[obj_name] = np.array(objective_rewards[obj_name])

        # Compute pairwise correlations
        for obj1 in obj_data:
            correlations[obj1] = {}
            for obj2 in obj_data:
                if len(obj_data[obj1]) == len(obj_data[obj2]):
                    corr, p_value = pearsonr(obj_data[obj1], obj_data[obj2])
                    correlations[obj1][obj2] = {
                        'correlation': float(corr),
                        'p_value': float(p_value)
                    }

        return correlations

    def _analyze_convergence(self, timesteps: List, objective_rewards: Dict[str, List]) -> Dict:
        """Analyze convergence properties"""
        convergence = {}

        for obj_name in self.objective_names:
            if obj_name in objective_rewards and len(objective_rewards[obj_name]) > 10:
                rewards = np.array(objective_rewards[obj_name])

                # Moving average for trend analysis
                window_size = min(10, len(rewards) // 4)
                moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

                # Compute improvement rate
                if len(moving_avg) > 1:
                    improvement_rate = (moving_avg[-1] - moving_avg[0]) / len(moving_avg)
                else:
                    improvement_rate = 0.0

                # Compute stability (variance in last 25% of training)
                last_quarter = rewards[-len(rewards) // 4:] if len(rewards) > 4 else rewards
                stability = np.std(last_quarter)

                convergence[obj_name] = {
                    'improvement_rate': float(improvement_rate),
                    'final_performance': float(rewards[-1]),
                    'stability': float(stability),
                    'best_performance': float(np.max(rewards)),
                }

        return convergence

    def _summarize_performance(self, objective_rewards: Dict[str, List]) -> Dict:
        """Summarize overall performance"""
        summary = {}

        for obj_name in self.objective_names:
            if obj_name in objective_rewards and len(objective_rewards[obj_name]) > 0:
                rewards = np.array(objective_rewards[obj_name])

                summary[obj_name] = {
                    'mean': float(np.mean(rewards)),
                    'std': float(np.std(rewards)),
                    'min': float(np.min(rewards)),
                    'max': float(np.max(rewards)),
                    'episodes': len(rewards),
                    'trend': 'improving' if len(rewards) > 1 and rewards[-1] > rewards[0] else 'declining'
                }

        return summary

    def plot_training_progress(self, training_stats: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training progress for all objectives

        Args:
            training_stats: Training statistics
            save_path: Optional path to save plot

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MOSAC Training Progress', fontsize=16)

        if 'timesteps' not in training_stats or 'objective_rewards' not in training_stats:
            plt.text(0.5, 0.5, 'Insufficient data for plotting',
                     ha='center', va='center', transform=fig.transFigure)
            return fig

        timesteps = training_stats['timesteps']
        objective_rewards = training_stats['objective_rewards']

        for i, obj_name in enumerate(self.objective_names):
            if obj_name in objective_rewards and len(objective_rewards[obj_name]) > 0:
                ax = axes[i // 2, i % 2]
                rewards = objective_rewards[obj_name]

                # Plot raw data
                x_data = timesteps[:len(rewards)] if len(timesteps) >= len(rewards) else range(len(rewards))
                ax.plot(x_data, rewards, alpha=0.6, linewidth=1, label='Raw')

                # Plot moving average
                if len(rewards) > 5:
                    window_size = max(1, len(rewards) // 20)
                    moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
                    x_smooth = x_data[:len(moving_avg)]
                    ax.plot(x_smooth, moving_avg, linewidth=2, label='Moving Avg')

                ax.set_title(f'{obj_name.replace("_", " ").title()} Objective')
                ax.set_xlabel('Episodes' if len(timesteps) < len(rewards) else 'Timesteps')
                ax.set_ylabel('Cumulative Reward')
                ax.grid(True, alpha=0.3)
                ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_pareto_front(self, pareto_archive: List[np.ndarray],
                          obj_indices: Tuple[int, int] = (0, 1),
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot 2D projection of Pareto front

        Args:
            pareto_archive: List of objective vectors
            obj_indices: Indices of objectives to plot (x, y)
            save_path: Optional path to save plot

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        if len(pareto_archive) == 0:
            ax.text(0.5, 0.5, 'No Pareto solutions available',
                    ha='center', va='center', transform=ax.transAxes)
            return fig

        # Extract objectives
        objectives = np.array(pareto_archive)
        x_obj = objectives[:, obj_indices[0]]
        y_obj = objectives[:, obj_indices[1]]

        # Plot Pareto front
        ax.scatter(x_obj, y_obj, c='red', s=50, alpha=0.7, label='Pareto Front')

        # Connect points to show front
        sorted_indices = np.argsort(x_obj)
        ax.plot(x_obj[sorted_indices], y_obj[sorted_indices],
                'r--', alpha=0.5, linewidth=1)

        ax.set_xlabel(f'{self.objective_names[obj_indices[0]].replace("_", " ").title()}')
        ax.set_ylabel(f'{self.objective_names[obj_indices[1]].replace("_", " ").title()}')
        ax.set_title('Pareto Front Projection')
        ax.grid(True, alpha=0.3)
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def export_analysis_report(self, analysis_results: Dict, filepath: str):
        """
        Export analysis results to JSON file

        Args:
            analysis_results: Results from analyze_training_data
            filepath: Path to save JSON report
        """

        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            else:
                return obj

        json_data = convert_for_json(analysis_results)

        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)


def create_preference_weights(strategy: str = "uniform", num_objectives: int = 4, **kwargs) -> np.ndarray:
    """
    Create preference weights for scalarization

    Args:
        strategy: Weight generation strategy
        num_objectives: Number of objectives
        **kwargs: Additional arguments for specific strategies

    Returns:
        Normalized preference weights
    """
    if strategy == "uniform":
        weights = np.ones(num_objectives)
    elif strategy == "random":
        weights = np.random.random(num_objectives)
    elif strategy == "focused":
        # Focus on one objective
        focus_idx = kwargs.get("focus_objective", 0)
        focus_strength = kwargs.get("focus_strength", 0.8)
        weights = np.full(num_objectives, (1 - focus_strength) / (num_objectives - 1))
        weights[focus_idx] = focus_strength
    elif strategy == "custom":
        weights = np.array(kwargs.get("weights", np.ones(num_objectives)))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Normalize weights
    return weights / np.sum(weights)


def load_training_stats(filepath: str) -> Dict:
    """
    Load training statistics from JSON file

    Args:
        filepath: Path to training stats file

    Returns:
        Training statistics dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_training_stats(stats: Dict, filepath: str):
    """
    Save training statistics to JSON file

    Args:
        stats: Training statistics
        filepath: Path to save file
    """

    # Convert numpy arrays to lists
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        else:
            return obj

    json_data = convert_for_json(stats)

    with open(filepath, 'w') as f:
        json.dump(json_data, f, indent=2)