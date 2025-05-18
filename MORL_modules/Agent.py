import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces
import random
from collections import deque

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.sac.sac import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import polyak_update


class MOReplayBuffer(ReplayBuffer):
    """
    Extended replay buffer that stores vector rewards for multi-objective RL.
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            num_objectives: int = 4,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
    ):
        """
        Initialize multi-objective replay buffer.

        Args:
            buffer_size: Max number of transitions to store
            observation_space: Observation space
            action_space: Action space
            num_objectives: Number of objectives/rewards
            device: PyTorch device
            n_envs: Number of parallel environments
            optimize_memory_usage: Optimize memory by sharing next_obs
            handle_timeout_termination: Handle timeouts properly
        """
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

        self.num_objectives = num_objectives

        # Modify rewards buffer to store vectors instead of scalars
        # Shape becomes (buffer_size, n_envs, num_objectives)
        self.rewards = np.zeros((self.buffer_size, self.n_envs, self.num_objectives), dtype=np.float32)

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:
        """
        Add a new transition to the buffer with vector reward.

        Args:
            obs: Observation
            next_obs: Next observation
            action: Action
            reward: Vector reward with shape (n_envs, num_objectives)
            done: Terminal flag
            infos: Additional information
        """
        # Reshape rewards if needed to ensure correct shape
        if reward.ndim == 1:
            reward = reward.reshape(-1, self.num_objectives)

        # Validate reward shape
        assert reward.shape[1] == self.num_objectives, f"Expected reward with {self.num_objectives} objectives, got {reward.shape[1]}"

        # Use the parent class add method but handle the reward differently
        # Store observations, actions, dones as before
        self.observations[self.pos] = np.array(obs).copy()
        if self.optimize_memory_usage:
            self.next_observations = None  # Save memory
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        # Handle timeouts for off-policy algorithms
        if self.handle_timeout_termination:
            self._handle_timeouts(infos)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[GymEnv] = None) -> Dict[str, th.Tensor]:
        """
        Sample a batch of transitions with vector rewards.

        Args:
            batch_size: Number of samples to return
            env: Associated gym environment (not used)

        Returns:
            Dictionary with sampled transitions including vector rewards
        """
        # Sample indices
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)

        # Create the return dictionary
        data = {
            "observations": self._normalize_obs(self.observations[batch_inds, 0, :]),
            "actions": self.actions[batch_inds, 0, :],
            "rewards": self.rewards[batch_inds, 0, :],  # Multi-objective rewards
            "dones": self.dones[batch_inds, 0],
        }

        if not self.optimize_memory_usage:
            data["next_observations"] = self._normalize_obs(self.next_observations[batch_inds, 0, :])
        else:
            # If next_obs is not stored separately, use the next indices
            next_inds = (batch_inds + 1) % self.buffer_size
            # Check if the next indices correspond to the start of an episode
            valid_indices = ~self.dones[batch_inds, 0].astype(bool)
            # Replace invalid indices with valid ones to avoid invalid next states
            next_inds = next_inds * valid_indices + batch_inds * (~valid_indices)
            data["next_observations"] = self._normalize_obs(self.observations[next_inds, 0, :])

        # Convert to PyTorch tensors
        return {k: th.as_tensor(v, device=self.device) for k, v in data.items()}


class MOQNetwork(nn.Module):
    """
    Multi-objective Q-Network that outputs Q-values for each objective.

    Architecture options:
    1. Shared feature extractor with separate heads (share_features=True)
    2. Completely separate networks for each objective (share_features=False)
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            num_objectives: int = 4,
            share_features: bool = True,
            net_arch: List[int] = [256, 256],
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
    ):
        """
        Initialize Q-network for multiple objectives.

        Args:
            observation_space: Observation space
            action_space: Action space
            num_objectives: Number of objectives/rewards
            share_features: Whether to share feature extraction across objectives
            net_arch: Network architecture (sizes of hidden layers)
            activation_fn: Activation function
            normalize_images: Whether to normalize images
        """
        super().__init__()
        self.num_objectives = num_objectives
        self.share_features = share_features
        self.normalize_images = normalize_images

        # Get dimensions of observation and action spaces
        if isinstance(observation_space, spaces.Box):
            self.obs_dim = int(np.prod(observation_space.shape))
        else:
            raise ValueError(f"Unsupported observation space: {observation_space}")

        if isinstance(action_space, spaces.Box):
            self.action_dim = int(np.prod(action_space.shape))
        else:
            raise ValueError(f"Unsupported action space: {action_space}")

        # Input size is the combined dimension of observation and action
        input_dim = self.obs_dim + self.action_dim

        if self.share_features:
            # Shared feature extractor
            self.shared_net = self._create_mlp(
                input_dim,
                net_arch[-1],  # Output of shared features is last layer size
                net_arch[:-1],  # Hidden layers for shared features
                activation_fn
            )

            # Separate heads for each objective
            self.q_heads = nn.ModuleList([
                nn.Linear(net_arch[-1], 1) for _ in range(num_objectives)
            ])
        else:
            # Completely separate networks for each objective
            self.q_nets = nn.ModuleList([
                self._create_mlp(
                    input_dim,
                    1,  # Output single Q-value
                    net_arch,  # Full network architecture
                    activation_fn
                ) for _ in range(num_objectives)
            ])

    def _create_mlp(
            self,
            input_dim: int,
            output_dim: int,
            hidden_sizes: List[int],
            activation_fn: Type[nn.Module]
    ) -> nn.Sequential:
        """
        Create a multi-layer perceptron.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_sizes: Sizes of hidden layers
            activation_fn: Activation function

        Returns:
            Sequential MLP model
        """
        layers = []
        current_dim = input_dim

        # Add hidden layers
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation_fn())
            current_dim = hidden_dim

        # Add output layer
        layers.append(nn.Linear(current_dim, output_dim))

        return nn.Sequential(*layers)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> List[th.Tensor]:
        """
        Forward pass returning Q-values for each objective.

        Args:
            obs: Observation tensor
            actions: Action tensor

        Returns:
            List of Q-values [q1, q2, ..., qn], one per objective
        """
        # Flatten observation if needed
        if len(obs.shape) > 2:
            obs = obs.reshape(obs.shape[0], -1)

        # Combine observation and action as input
        x = th.cat([obs, actions], dim=1)

        if self.share_features:
            # Forward through shared network
            features = self.shared_net(x)
            # Get Q-values from each head
            q_values = [head(features) for head in self.q_heads]
        else:
            # Forward through separate networks
            q_values = [net(x) for net in self.q_nets]

        return q_values


class MOSACPolicy(SACPolicy):
    """
    Policy class for MOSAC, extending SACPolicy to handle multiple objectives.
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            num_objectives: int = 4,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            use_sde: bool = False,
            log_std_init: float = -3,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            features_extractor_class = None,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = True,
            share_features_across_objectives: bool = True,
    ):
        """
        Initialize Multi-Objective SAC policy.

        Args:
            observation_space: Observation space
            action_space: Action space
            lr_schedule: Learning rate schedule
            num_objectives: Number of objectives
            Additional standard SACPolicy parameters
            share_features_across_objectives: Whether to share features across objectives
        """
        self.num_objectives = num_objectives
        self.share_features_across_objectives = share_features_across_objectives

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )

    def make_critic(self, features_extractor=None) -> MOQNetwork:
        """
        Create a multi-objective critic.

        Returns:
            Multi-objective Q-Network
        """
        return MOQNetwork(
            self.observation_space,
            self.action_space,
            num_objectives=self.num_objectives,
            share_features=self.share_features_across_objectives,
            net_arch=self.net_arch.get("qf", [256, 256])
        )

    def make_critic_target(self) -> MOQNetwork:
        """
        Create a target multi-objective critic.

        Returns:
            Target multi-objective Q-Network
        """
        return MOQNetwork(
            self.observation_space,
            self.action_space,
            num_objectives=self.num_objectives,
            share_features=self.share_features_across_objectives,
            net_arch=self.net_arch.get("qf", [256, 256])
        )


class MOSAC(SAC):
    """
    Multi-Objective Soft Actor-Critic algorithm.
    Extends SAC to handle vector rewards and implement Pareto optimization.
    """

    def __init__(
            self,
            policy: Union[str, Type[MOSACPolicy]],
            env: Union[GymEnv, str],
            num_objectives: int = 4,
            preference_weights: Optional[np.ndarray] = None,
            learning_rate: Union[float, Schedule] = 3e-4,
            buffer_size: int = 1_000_000,
            learning_starts: int = 100,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = 1,
            gradient_steps: int = 1,
            action_noise = None,
            replay_buffer_class: Optional[Type[MOReplayBuffer]] = MOReplayBuffer,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            ent_coef: Union[str, float] = "auto",
            target_update_interval: int = 1,
            target_entropy: Union[str, float] = "auto",
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            use_sde_at_warmup: bool = False,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            hypervolume_ref_point: Optional[np.ndarray] = None,
    ):
        """
        Initialize MOSAC algorithm.

        Args:
            policy: Policy class or string
            env: Environment
            num_objectives: Number of objectives
            preference_weights: Default weights for scalarization
            hypervolume_ref_point: Reference point for hypervolume calculation
            Additional standard SAC parameters
        """
        self.num_objectives = num_objectives

        # Default preference weights if none provided (equal weighting)
        if preference_weights is None:
            self.preference_weights = np.ones(num_objectives) / num_objectives
        else:
            assert len(preference_weights) == num_objectives, f"Preference weights must have length {num_objectives}"
            # Normalize weights to sum to 1
            self.preference_weights = np.array(preference_weights) / np.sum(preference_weights)

        # For hypervolume calculation (optional Pareto front metric)
        if hypervolume_ref_point is None:
            # Default to zeros (assuming rewards are positive)
            self.hypervolume_ref_point = np.zeros(num_objectives)
        else:
            self.hypervolume_ref_point = hypervolume_ref_point

        # Ensure policy kwargs contain num_objectives
        if policy_kwargs is None:
            policy_kwargs = {}
        policy_kwargs["num_objectives"] = num_objectives

        # Ensure replay buffer kwargs contain num_objectives
        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {}
        replay_buffer_kwargs["num_objectives"] = num_objectives

        # To track Pareto front during training
        self.pareto_front = []

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            ent_coef,
            target_update_interval,
            target_entropy,
            use_sde,
            sde_sample_freq,
            use_sde_at_warmup,
            tensorboard_log,
            create_eval_env,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )

    def _setup_model(self) -> None:
        """
        Setup model: create policy, replay buffer, and optimizer.
        """
        super()._setup_model()

        # Initialize critics list for each objective
        self.critic_list = []
        self.critic_target_list = []

        if self.policy is not None:
            # Use policy's critics
            self.critic = self.policy.critic
            self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Train the model for gradient_steps with multi-objective rewards.

        Args:
            gradient_steps: Number of gradient steps
            batch_size: Batch size
        """
        # Sample from the buffer
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size=batch_size)

            # Get current and next Q-values for each objective
            with th.no_grad():
                # Sample actions according to current policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)

                # Compute target Q-values for each objective
                next_q_values_list = self.critic_target(
                    replay_data.next_observations, next_actions
                )

                # Apply entropy term for each objective
                target_q_values_list = []
                for next_q_values in next_q_values_list:
                    target_q_values = next_q_values - self.ent_coef * next_log_prob.reshape(-1, 1)
                    target_q_values_list.append(target_q_values)

                # Apply discount factor
                target_q_values_list = [
                    replay_data.rewards[:, i].reshape(-1, 1) +
                    (1 - replay_data.dones).reshape(-1, 1) *
                    self.gamma * target_q_values
                    for i, target_q_values in enumerate(target_q_values_list)
                ]

            # Get current Q-values for each objective
            current_q_values_list = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss for each objective
            critic_loss = 0
            for i, (current_q, target_q) in enumerate(zip(current_q_values_list, target_q_values_list)):
                # Apply preference weight to each objective
                objective_loss = F.mse_loss(current_q, target_q) * self.preference_weights[i]
                critic_loss += objective_loss

                # Log each objective's loss
                self.logger.record(f"train/critic_loss_obj{i+1}", objective_loss.item())

            # Optimize critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Log overall critic loss
            self.logger.record("train/critic_loss", critic_loss.item())

            # Actor training
            # Sample actions according to current policy
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)

            # Compute actor loss based on multi-objective Q-values
            q_values_pi_list = self.critic(replay_data.observations, actions_pi)

            # Scalarize Q-values using preference weights
            scalarized_q_values_pi = th.zeros_like(q_values_pi_list[0])
            for i, q_values in enumerate(q_values_pi_list):
                scalarized_q_values_pi += self.preference_weights[i] * q_values

            actor_loss = (self.ent_coef * log_prob - scalarized_q_values_pi).mean()

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            self.logger.record("train/actor_loss", actor_loss.item())

            # Update entropy coefficient if needed
            if self.ent_coef_optimizer is not None:
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                self.ent_coef = th.exp(self.log_ent_coef.detach())

                self.logger.record("train/ent_coef", self.ent_coef.item())
                self.logger.record("train/ent_coef_loss", ent_coef_loss.item())

            # Update target networks
            if self._n_updates % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

            self._n_updates += 1

    def _is_dominated(self, reward_vec1, reward_vec2):
        """
        Check if reward_vec1 is dominated by reward_vec2.
        reward_vec1 is dominated if reward_vec2 is at least as good in all objectives
        and strictly better in at least one objective.

        Args:
            reward_vec1: First reward vector to compare
            reward_vec2: Second reward vector to compare

        Returns:
            True if reward_vec1 is dominated by reward_vec2, False otherwise
        """
        # Check if reward_vec2 is at least as good as reward_vec1 in all objectives
        at_least_as_good = np.all(reward_vec2 >= reward_vec1)
        # Check if reward_vec2 is strictly better than reward_vec1 in at least one objective
        strictly_better = np.any(reward_vec2 > reward_vec1)

        return at_least_as_good and strictly_better

    def _update_pareto_front(self, candidate):
        """
        Update the Pareto front with a new candidate solution.

        Args:
            candidate: Reward vector to consider adding to the Pareto front
        """
        # Check if candidate is dominated by any solution in the Pareto front
        for solution in self.pareto_front:
            if self._is_dominated(candidate, solution):
                return

        # Remove solutions from Pareto front that are dominated by the candidate
        self.pareto_front = [
            solution for solution in self.pareto_front
            if not self._is_dominated(solution, candidate)
        ]

        # Add candidate to Pareto front
        self.pareto_front.append(candidate)

    def calculate_hypervolume(self):
        """
        Calculate the hypervolume indicator of the current Pareto front.

        Returns:
            Hypervolume value (higher is better)
        """
        if not self.pareto_front:
            return 0.0

        # Simple implementation - for more complex cases use specialized libraries
        # like pymoo or pygmo
        # This assumes 2D objective space for simplicity
        if self.num_objectives == 2:
            # Sort points by first objective
            sorted_front = sorted(self.pareto_front, key=lambda x: x[0])
            hypervolume = 0.0

            # Calculate the area
            prev_x = self.hypervolume_ref_point[0]
            for point in sorted_front:
                hypervolume += (point[0] - prev_x) * (point[1] - self.hypervolume_ref_point[1])
                prev_x = point[0]

            return hypervolume
        else:
            # For higher dimensions, just return the count of Pareto-optimal solutions
            # as a simple proxy (a proper implementation would use a specialized library)
            return len(self.pareto_front)

    def learn(
            self,
            total_timesteps: int,
            callback = None,
            log_interval: int = 4,
            eval_env = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "MOSAC",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            update_preference_freq: int = None,
            preference_candidates: Optional[List[np.ndarray]] = None,
    ) -> "MOSAC":
        """
        Modified learning loop for multi-objective setting.
        Tracks Pareto front during training and optionally updates preference weights.

        Args:
            total_timesteps: Total timesteps to train for
            callback: Callback function
            log_interval: Logging interval
            eval_env: Environment for evaluation
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of episodes for evaluation
            tb_log_name: Tensorboard log name
            eval_log_path: Path for evaluation logs
            reset_num_timesteps: Whether to reset number of timesteps
            update_preference_freq: Frequency to update preference weights
            preference_candidates: List of candidate preference weights to cycle through

        Returns:
            Trained MOSAC model
        """
        # If preference weights should be updated periodically
        if update_preference_freq is not None and preference_candidates is not None:
            # Make sure preference_candidates contains valid weights
            for weights in preference_candidates:
                assert len(weights) == self.num_objectives, f"All preference candidates must have {self.num_objectives} weights"
                # Normalize weights
                weights = np.array(weights) / np.sum(weights)

            pref_idx = 0

        # Standard SAC learn loop with modifications for multi-objective
        return_val = super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

        # Additional multi-objective specific logic
        # Update Pareto front when collecting experience
        if hasattr(self, "_last_obs") and hasattr(self, "_last_reward"):
            self._update_pareto_front(self._last_reward)

            # Log hypervolume if there are at least 2 solutions in the Pareto front
            if len(self.pareto_front) >= 2:
                hypervolume = self.calculate_hypervolume()
                self.logger.record("metrics/hypervolume", hypervolume)

        # Update preference weights if needed
        if update_preference_freq is not None and preference_candidates is not None:
            if self.num_timesteps % update_preference_freq == 0:
                self.preference_weights = preference_candidates[pref_idx]
                pref_idx = (pref_idx + 1) % len(preference_candidates)

                # Log current preference weights
                for i, weight in enumerate(self.preference_weights):
                    self.logger.record(f"metrics/preference_weight_{i+1}", weight)

        return return_val

    def predict(
            self,
            observation: np.ndarray,
            state = None,
            mask = None,
            deterministic: bool = False,
            preference_weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the policy action from an observation.
        Allow specifying custom preference weights for prediction.

        Args:
            observation: Observation
            state: State (for recurrent policies)
            mask: Mask (for recurrent policies)
            deterministic: Whether to use deterministic actions
            preference_weights: Custom preference weights to use for this prediction

        Returns:
            Action and state
        """
        # Use provided preference weights if given, otherwise use default
        if preference_weights is not None:
            # Save current weights
            old_weights = self.preference_weights
            # Set temporary weights
            self.preference_weights = preference_weights / np.sum(preference_weights)

        # Get standard prediction
        actions, states = super().predict(observation, state, mask, deterministic)

        # Restore original weights if temporary ones were used
        if preference_weights is not None:
            self.preference_weights = old_weights

        return actions, states


# Utility function to create a MOSAC agent
def create_mosac_agent(
        env,
        num_objectives=4,
        preference_weights=None,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        verbose=1,
        device="auto",
        tensorboard_log=None,
        policy_kwargs=None,
):
    """
    Create a MOSAC agent with appropriate configuration.

    Args:
        env: Environment
        num_objectives: Number of objectives
        preference_weights: Initial preference weights
        learning_rate: Learning rate
        buffer_size: Buffer size
        batch_size: Batch size
        gamma: Discount factor
        verbose: Verbosity level
        device: Device to use
        tensorboard_log: Tensorboard log path
        policy_kwargs: Additional policy kwargs

    Returns:
        MOSAC agent
    """
    # Set default policy kwargs if none
    if policy_kwargs is None:
        policy_kwargs = {
            "net_arch": {
                "pi": [256, 256],
                "qf": [256, 256]
            },
            "share_features_across_objectives": True
        }

    return MOSAC(
        policy="MlpPolicy",
        env=env,
        num_objectives=num_objectives,
        preference_weights=preference_weights,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        verbose=verbose,
        device=device,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs
    )