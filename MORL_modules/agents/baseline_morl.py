# MORL_modules/agents/baseline_morl.py

import os
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
import pickle
from pathlib import Path

from stable_baselines3 import SAC, PPO, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger

from MORL_modules.wrappers.scalarized_mo_pcs_wrapper import ScalarizedMOPCSWrapper


class MORewardTrackingCallback(BaseCallback):
    """Callback to track multi-objective rewards during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.mo_episode_rewards = []
        self.mo_step_rewards = []

    def _on_step(self) -> bool:
        # Get MO rewards from info if available
        if hasattr(self.locals, 'infos') and self.locals['infos']:
            for info in self.locals['infos']:
                if 'mo_rewards_original' in info:
                    self.mo_step_rewards.append(info['mo_rewards_original'])

                # Track episode completion
                if info.get('episode') is not None:
                    episode_info = info['episode']
                    if 'mo_episode_totals' in info:
                        self.mo_episode_rewards.append(info['mo_episode_totals'])
        return True


class BaselineMORLAgent:
    """
    Baseline MORL agent using Linear Scalarization with multiple weight preferences.

    This implements a simple but effective MORL baseline that:
    1. Trains multiple single-objective agents with different weight combinations
    2. Maintains a collection of diverse policies
    3. Allows policy selection based on user preferences
    """

    def __init__(
            self,
            env_creator_fn,
            algorithm: str = "SAC",
            weight_vectors: Optional[List[List[float]]] = None,
            results_dir: str = "MORL_modules/results",
            seed: int = 42,
            verbose: int = 1
    ):
        """
        Initialize baseline MORL agent.

        Args:
            env_creator_fn: Function that creates the base environment
            algorithm: RL algorithm to use ("SAC", "PPO", "TD3")
            weight_vectors: List of weight vectors for different preferences
            results_dir: Directory to save results
            seed: Random seed
            verbose: Verbosity level
        """
        self.env_creator_fn = env_creator_fn
        self.algorithm = algorithm
        self.seed = seed
        self.verbose = verbose

        # Default weight vectors if none provided
        if weight_vectors is None:
            self.weight_vectors = [
                [1.0, 0.0, 0.0, 0.0],  # Economic only
                [0.0, 1.0, 0.0, 0.0],  # Battery health only
                [0.0, 0.0, 1.0, 0.0],  # Grid support only
                [0.0, 0.0, 0.0, 1.0],  # Autonomy only
                [0.25, 0.25, 0.25, 0.25],  # Balanced
                [0.4, 0.3, 0.2, 0.1],  # Economic focused
                [0.1, 0.4, 0.3, 0.2],  # Battery health focused
                [0.2, 0.1, 0.4, 0.3],  # Grid support focused
                [0.3, 0.2, 0.1, 0.4],  # Autonomy focused
            ]
        else:
            self.weight_vectors = weight_vectors

        # Setup directories
        self.results_dir = Path(results_dir)
        self.models_dir = self.results_dir / "models"
        self.logs_dir = self.results_dir / "logs"
        self.figures_dir = self.results_dir / "figures"

        for dir_path in [self.results_dir, self.models_dir, self.logs_dir, self.figures_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Algorithm mapping
        self.algorithm_classes = {
            "SAC": SAC,
            "PPO": PPO,
            "TD3": TD3
        }

        if algorithm not in self.algorithm_classes:
            raise ValueError(f"Algorithm {algorithm} not supported. Choose from {list(self.algorithm_classes.keys())}")

        # Storage for trained agents and results
        self.agents: Dict[str, Any] = {}
        self.training_results: Dict[str, Dict] = {}

        # Setup logging
        self.logger = logging.getLogger(f"BaselineMORL_{id(self)}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(self.logs_dir / "baseline_morl.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info(f"Initialized BaselineMORLAgent with {len(self.weight_vectors)} weight vectors")

    def _create_agent_name(self, weights: List[float]) -> str:
        """Create descriptive name for agent based on weights."""
        weight_str = "_".join([f"{w:.2f}" for w in weights])
        return f"{self.algorithm}_w_{weight_str}"

    def train_all_agents(
            self,
            total_timesteps: int = 50000,
            algorithm_kwargs: Optional[Dict] = None
    ) -> Dict[str, Dict]:
        """
        Train agents for all weight vectors.

        Args:
            total_timesteps: Training timesteps per agent
            algorithm_kwargs: Additional arguments for RL algorithm

        Returns:
            Dictionary of training results for each agent
        """
        if algorithm_kwargs is None:
            algorithm_kwargs = {}

        results = {}

        for i, weights in enumerate(self.weight_vectors):
            agent_name = self._create_agent_name(weights)
            self.logger.info(f"Training agent {i + 1}/{len(self.weight_vectors)}: {agent_name}")

            # Create environment with scalarized wrapper
            env = self.env_creator_fn()
            wrapped_env = ScalarizedMOPCSWrapper(
                env,
                weights=weights,
                normalize_weights=False
            )

            # Create agent
            algorithm_class = self.algorithm_classes[self.algorithm]

            # Default hyperparameters based on algorithm
            default_kwargs = self._get_default_hyperparameters()
            default_kwargs.update(algorithm_kwargs)

            agent = algorithm_class(
                "MlpPolicy",
                wrapped_env,
                seed=self.seed,
                verbose=self.verbose,
                tensorboard_log=str(self.logs_dir),
                **default_kwargs
            )

            # Training callback
            callback = MORewardTrackingCallback()

            # Train agent
            agent.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                tb_log_name=agent_name
            )

            # Save agent
            model_path = self.models_dir / f"{agent_name}.zip"
            agent.save(str(model_path))

            # Store results
            self.agents[agent_name] = agent
            self.training_results[agent_name] = {
                'weights': weights,
                'mo_episode_rewards': callback.mo_episode_rewards,
                'mo_step_rewards': callback.mo_step_rewards,
                'model_path': str(model_path)
            }

            results[agent_name] = self.training_results[agent_name]

            self.logger.info(f"Completed training for {agent_name}")

        # Save all results
        results_path = self.results_dir / "training_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)

        self.logger.info(f"All training completed. Results saved to {results_path}")
        return results

    def _get_default_hyperparameters(self) -> Dict:
        """Get default hyperparameters for each algorithm."""
        defaults = {
            "SAC": {
                "learning_rate": 3e-4,
                "buffer_size": 100000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
            },
            "PPO": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "n_epochs": 10,
            },
            "TD3": {
                "learning_rate": 1e-3,
                "buffer_size": 100000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "policy_delay": 2,
                "target_policy_noise": 0.2,
                "target_noise_clip": 0.5,
            }
        }
        return defaults.get(self.algorithm, {})

    def evaluate_agents(
            self,
            n_eval_episodes: int = 10,
            deterministic: bool = True
    ) -> Dict[str, Dict]:
        """
        Evaluate all trained agents.

        Args:
            n_eval_episodes: Number of episodes for evaluation
            deterministic: Whether to use deterministic actions

        Returns:
            Evaluation results for each agent
        """
        if not self.agents:
            self.logger.warning("No agents trained yet. Loading from saved models...")
            self.load_agents()

        eval_results = {}

        for agent_name, agent in self.agents.items():
            self.logger.info(f"Evaluating {agent_name}")

            # Get original weights for this agent
            weights = self.training_results[agent_name]['weights']

            # Create evaluation environment
            env = self.env_creator_fn()
            wrapped_env = ScalarizedMOPCSWrapper(env, weights=weights, normalize_weights=False)

            episode_rewards = []
            episode_mo_rewards = []

            for episode in range(n_eval_episodes):
                obs, _ = wrapped_env.reset()
                episode_reward = 0
                episode_mo_reward = np.zeros(4)
                done = False

                while not done:
                    action, _ = agent.predict(obs, deterministic=deterministic)
                    obs, reward, terminated, truncated, info = wrapped_env.step(action)
                    done = terminated or truncated

                    episode_reward += reward
                    if 'mo_rewards_original' in info:
                        episode_mo_reward += info['mo_rewards_original']

                episode_rewards.append(episode_reward)
                episode_mo_rewards.append(episode_mo_reward)

            eval_results[agent_name] = {
                'weights': weights,
                'scalar_rewards': episode_rewards,
                'mo_rewards': episode_mo_rewards,
                'mean_scalar_reward': np.mean(episode_rewards),
                'std_scalar_reward': np.std(episode_rewards),
                'mean_mo_rewards': np.mean(episode_mo_rewards, axis=0),
                'std_mo_rewards': np.std(episode_mo_rewards, axis=0)
            }

            self.logger.info(f"Agent {agent_name} - Mean reward: {np.mean(episode_rewards):.3f}")

        return eval_results

    def load_agents(self):
        """Load trained agents from saved models."""
        for agent_name in self.training_results.keys():
            model_path = self.training_results[agent_name]['model_path']
            if os.path.exists(model_path):
                algorithm_class = self.algorithm_classes[self.algorithm]
                agent = algorithm_class.load(model_path)
                self.agents[agent_name] = agent
                self.logger.info(f"Loaded agent {agent_name}")
            else:
                self.logger.warning(f"Model file not found: {model_path}")

    def get_pareto_front_approximation(self, eval_results: Dict[str, Dict]) -> np.ndarray:
        """
        Extract Pareto front approximation from evaluation results.

        Args:
            eval_results: Results from evaluate_agents()

        Returns:
            Array of Pareto-optimal points (n_points, n_objectives)
        """
        # Collect all mean MO rewards
        all_rewards = []
        for result in eval_results.values():
            all_rewards.append(result['mean_mo_rewards'])

        all_rewards = np.array(all_rewards)

        # Find Pareto front (assuming maximization for all objectives)
        pareto_front = []
        for i, point in enumerate(all_rewards):
            is_dominated = False
            for j, other_point in enumerate(all_rewards):
                if i != j and np.all(other_point >= point) and np.any(other_point > point):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(point)

        return np.array(pareto_front) if pareto_front else all_rewards

    def select_agent_by_preference(
            self,
            preference_weights: List[float],
            eval_results: Dict[str, Dict]
    ) -> Tuple[str, float]:
        """
        Select best agent for given preference weights.

        Args:
            preference_weights: Desired objective weights
            eval_results: Evaluation results

        Returns:
            Tuple of (agent_name, scalarized_performance)
        """
        preference_weights = np.array(preference_weights)
        best_agent = None
        best_score = float('-inf')

        for agent_name, result in eval_results.items():
            # Compute scalarized performance with preference weights
            mo_rewards = result['mean_mo_rewards']
            score = np.dot(preference_weights, mo_rewards)

            if score > best_score:
                best_score = score
                best_agent = agent_name

        return best_agent, best_score