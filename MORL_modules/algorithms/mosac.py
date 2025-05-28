import numpy as np
import torch as th
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable
from gymnasium import spaces

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.sac.sac import SAC
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.type_aliases import GymEnv, Schedule, TensorDict

def register_mosac():
    from rl_zoo3 import ALGOS
    ALGOS["mosac"] = MOSAC


class MOContinuousCritic(ContinuousCritic):
    """
    Multi-objective critic network that extends SB3's ContinuousCritic.
    Instead of creating our own neural network implementation, we'll use
    SB3's built-in network creation functions.
    """
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space,
            net_arch: List[int],  num_objectives: int = 2,
            features_extractor_class = None, features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            share_features_extractor: bool = True, n_critics: int = 2,
            activation_fn: Type[th.nn.Module] = th.nn.ReLU, normalize_images: bool = True,
            share_features_across_objectives: bool = True):

        super(ContinuousCritic, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images=normalize_images,
        )

        self.num_objectives = num_objectives
        self.share_features_extractor = share_features_extractor
        self.share_features_across_objectives = share_features_across_objectives
        self.n_critics = n_critics
        self.q_networks = []

        action_dim = get_action_dim(self.action_space)

        # Create separate q-networks for each critic ensemble and each objective
        for i in range(self.n_critics):
            # For each critic ensemble, create multiple heads (one per objective)
            if share_features_across_objectives:
                # Shared features extractor for all objectives
                # Using SB3's create_mlp function rather than custom implementation
                q_net = th.nn.Sequential(
                    *create_mlp(
                        self.features_dim + action_dim,
                        net_arch[-1],  # Last layer of shared features
                        net_arch[:-1],  # Hidden layers
                        activation_fn
                    )
                )

                # Create separate output heads for each objective
                q_heads = th.nn.ModuleList([
                    th.nn.Linear(net_arch[-1], 1) for _ in range(num_objectives)
                ])

                # Create a wrapper module that properly routes inputs
                class SharedFeatureQNet(th.nn.Module):
                    def __init__(self, base_net, heads):
                        super().__init__()
                        self.base_net = base_net
                        self.heads = heads

                    def forward(self, obs, actions):
                        x = th.cat([obs, actions], dim=1)
                        shared_features = self.base_net(x)
                        return [head(shared_features) for head in self.heads]

                self.q_networks.append(SharedFeatureQNet(q_net, q_heads))
            else:
                # Separate networks for each objective
                objective_nets = th.nn.ModuleList([
                    th.nn.Sequential(
                        *create_mlp(
                            self.features_dim + action_dim,
                            1,  # Single output
                            net_arch,
                            activation_fn
                        )
                    ) for _ in range(num_objectives)
                ])

                # Create a wrapper module for the separate networks
                class SeparateQNet(th.nn.Module):
                    def __init__(self, nets):
                        super().__init__()
                        self.nets = nets

                    def forward(self, obs, actions):
                        x = th.cat([obs, actions], dim=1)
                        return [net(x) for net in self.nets]

                self.q_networks.append(SeparateQNet(objective_nets))

        # Convert list to ModuleList
        self.q_networks = th.nn.ModuleList(self.q_networks)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> List[List[th.Tensor]]:
        """
        Forward pass of the multi-objective critic.
        Returns Q-values for each critic ensemble and each objective.

        Args:
            obs: Observation tensor
            actions: Action tensor

        Returns:
            List of lists: [critic_ensemble][objective] -> q_value tensor
        """
        features = self.extract_features(obs, self.features_extractor)

        # Get Q-values from each critic ensemble
        q_values = []
        for q_net in self.q_networks:
            q_values.append(q_net(features, actions))

        return q_values

    def q_value(self, obs: th.Tensor, actions: th.Tensor, preference_weights: th.Tensor) -> List[th.Tensor]:
        """
        Get scalarized Q-values using preference weights.

        Args:
            obs: Observation tensor
            actions: Action tensor
            preference_weights: Weights for objectives, shape (batch_size, num_objectives) or (num_objectives,)

        Returns:
            List of scalarized Q-values, one for each critic ensemble
        """
        all_critic_values = self.forward(obs, actions)

        # Ensure proper shape for preference weights
        if preference_weights.dim() == 1:
            preference_weights = preference_weights.unsqueeze(0).expand(obs.shape[0], -1)

        # Scalarize Q-values for each critic
        scalarized_q_values = []
        for critic_values in all_critic_values:
            # Stack objective values for easier computation
            # Shape: (batch_size, num_objectives)
            stacked_q_values = th.cat([q_val for q_val in critic_values], dim=1)

            # Compute weighted sum along objective dimension
            # Shape: (batch_size, 1)
            scalarized = th.sum(stacked_q_values * preference_weights, dim=1, keepdim=True)
            scalarized_q_values.append(scalarized)

        return scalarized_q_values


class MOSACPolicy(SACPolicy):
    """
    Policy class for MOSAC adapted to leverage SB3's policy infrastructure.
    """
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            num_objectives: int = 4,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[th.nn.Module] = th.nn.ReLU,
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

    def make_critic(self, features_extractor=None) -> MOContinuousCritic:
        """
        Create a multi-objective critic.
        Uses SB3's network structure but with multiple objective heads.

        Returns:
            Multi-objective continuous critic
        """
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )

        critic_kwargs.update({
            "num_objectives": self.num_objectives,
            "share_features_across_objectives": self.share_features_across_objectives
        })

        return MOContinuousCritic(**critic_kwargs).to(self.device)


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
        """Initialize multi-objective replay buffer."""
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
        """Add a new transition to the buffer with vector reward."""
        # Reshape rewards if needed to ensure correct shape
        if reward.ndim == 1:
            reward = reward.reshape(-1, self.num_objectives)

        # Validate reward shape
        assert reward.shape[1] == self.num_objectives, f"Expected reward with {self.num_objectives} objectives, got {reward.shape[1]}"

        # Call parent method but handle reward differently
        super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size: int, env: Optional[GymEnv] = None) -> TensorDict:
        """Sample a batch of transitions with vector rewards."""
        # Sample indices
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)

        # Sample using parent implementation but preserve reward vectors
        data = self._get_samples(batch_inds, env=env)
        return data


class MOSAC(SAC):
    """
    Multi-Objective Soft Actor-Critic algorithm.
    Adapted to work with Energy Net environment and use SB3's infrastructure.
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
            replay_buffer_class: Optional[Type[ReplayBuffer]] = MOReplayBuffer,
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
        """
        self.num_objectives = num_objectives

        # Default preference weights if none provided (equal weighting)
        if preference_weights is None:
            self.preference_weights = np.ones(num_objectives) / num_objectives
        else:
            assert len(preference_weights) == num_objectives, f"Preference weights must have length {num_objectives}"
            # Normalize weights to sum to 1
            self.preference_weights = np.array(preference_weights) / np.sum(preference_weights)

        # Convert to tensor for easier computation
        self.preference_weights_tensor = None  # Will be initialized when device is known

        # For hypervolume calculation (optional Pareto front metric)
        self.hypervolume_ref_point = hypervolume_ref_point or np.zeros(num_objectives)

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
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

        # Initialize preference weights tensor after device is known
        self.preference_weights_tensor = th.FloatTensor(self.preference_weights).to(self.device)

    def _setup_model(self) -> None:
        """
        Setup model: create policy, replay buffer, and optimizer.
        """
        super()._setup_model()

        # After setup, ensure preference weights are on the correct device
        self.preference_weights_tensor = th.FloatTensor(self.preference_weights).to(self.device)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Train the model for gradient_steps with multi-objective rewards.
        Customizes training to handle vector rewards and multiple critics.
        """
        # Call parent's train to handle common setup
        if self.should_collect_more_steps():
            return

        if gradient_steps <= 0:
            return

        # Update optimizers learning rate
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses, ent_coef_losses = [], [], []
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled states
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            with th.no_grad():
                # Sample actions according to current policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_log_prob = next_log_prob.reshape(-1, 1)

                # Compute multiple critics' Q-values
                next_q_values_list = self.critic_target(replay_data.next_observations, next_actions)

                # For each critic ensemble, compute the target Q-values for each objective
                target_q_values = []
                for critic_idx, next_q_values_per_critic in enumerate(next_q_values_list):
                    # For this critic ensemble, compute target Q-values for each objective
                    critic_target_q = []
                    for obj_idx, next_q_values in enumerate(next_q_values_per_critic):
                        # Apply entropy and discount to this objective
                        obj_target_q = (
                                replay_data.rewards[:, obj_idx].reshape(-1, 1) +
                                (1 - replay_data.dones).reshape(-1, 1) *
                                self.gamma * (next_q_values - self.ent_coef * next_log_prob)
                        )
                        critic_target_q.append(obj_target_q)
                    target_q_values.append(critic_target_q)

            # Get current Q-values for each critic ensemble and objective
            current_q_values_list = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0
            for critic_idx, (current_q_per_critic, target_q_per_critic) in enumerate(zip(current_q_values_list, target_q_values)):
                for obj_idx, (current_q, target_q) in enumerate(zip(current_q_per_critic, target_q_per_critic)):
                    # Apply preference weight to objective loss
                    obj_loss = F.mse_loss(current_q, target_q) * self.preference_weights[obj_idx]
                    critic_loss += obj_loss

                    # Log each objective loss
                    if self.num_timesteps % 1000 == 0 and critic_idx == 0:  # Only log first critic for clarity
                        self.logger.record(f"train/critic_{critic_idx}_obj_{obj_idx}_loss", obj_loss.item())

            critic_losses.append(critic_loss.item())

            # Optimize critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            q_values_pi_list = self.critic(replay_data.observations, actions_pi)

            # Modified actor loss computation with scalarization
            actor_loss = 0
            for critic_idx, q_values_per_critic in enumerate(q_values_pi_list):
                # Scalarize Q-values for this critic using preference weights
                scalarized_q = 0
                for obj_idx, q_values in enumerate(q_values_per_critic):
                    scalarized_q += self.preference_weights[obj_idx] * q_values

                # Contribution to actor loss from this critic
                critic_actor_loss = (self.ent_coef * log_prob - scalarized_q).mean()
                actor_loss += critic_actor_loss / len(q_values_pi_list)  # Average across critics

            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Optimize entropy coefficient if needed
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())

                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

                self.ent_coef = th.exp(self.log_ent_coef.detach())

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        # Log mean values
        self._n_updates += gradient_steps

        # Logging
        self.logger.record("train/n_updates", self._n_updates)
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        if len(critic_losses) > 0:
            self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0 and ent_coef_loss is not None:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
            self.logger.record("train/ent_coef", self.ent_coef.item())

    def _is_dominated(self, reward_vec1, reward_vec2):
        """
        Check if reward_vec1 is dominated by reward_vec2.
        """
        # Check if reward_vec2 is at least as good as reward_vec1 in all objectives
        at_least_as_good = np.all(reward_vec2 >= reward_vec1)
        # Check if reward_vec2 is strictly better than reward_vec1 in at least one objective
        strictly_better = np.any(reward_vec2 > reward_vec1)

        return at_least_as_good and strictly_better

    def _update_pareto_front(self, candidate):
        """
        Update the Pareto front with a new candidate solution.
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

    def _extract_mo_rewards(self, env_output):
        """
        Extract multi-objective rewards from environment output.
        Adapts the environment's reward to the multi-objective format.

        Args:
            env_output: Output from environment step

        Returns:
            Vector reward with shape (n_envs, num_objectives)
        """
        obs, reward, terminated, truncated, info = env_output

        # Default case: Single scalar reward - convert to vector with first objective
        if isinstance(reward, (int, float, np.number)) or (isinstance(reward, np.ndarray) and reward.ndim == 0):
            mo_reward = np.zeros((self.n_envs, self.num_objectives))
            mo_reward[:, 0] = reward

            # Check if info contains additional objectives
            if "mo_rewards" in info:
                # Use provided multi-objective rewards
                mo_reward = info["mo_rewards"]
            else:
                # Try to extract objectives from info
                # The EnergyNet environment might provide these metrics
                if "battery_level" in info:
                    mo_reward[:, 1] = info["battery_level"] / 100.0  # Normalized battery level

                if "net_exchange" in info and "iso_buy_price" in info and "iso_sell_price" in info:
                    # Economic objective - normalize based on price range
                    net_exchange = info["net_exchange"]
                    if net_exchange > 0:  # Buying energy
                        cost = net_exchange * info["iso_sell_price"]
                    else:  # Selling energy
                        cost = net_exchange * info["iso_buy_price"]

                    # Convert cost to reward and normalize
                    mo_reward[:, 0] = -cost / 100.0  # Normalized economic reward

                if "grid_balance" in info:
                    # Grid support objective
                    mo_reward[:, 2] = info["grid_balance"] / 100.0  # Normalized grid balance

        # Environment already returns vector reward
        elif isinstance(reward, np.ndarray) and reward.ndim >= 1:
            # Ensure correct shape
            if reward.shape[-1] == self.num_objectives:
                mo_reward = reward
            else:
                # Pad or truncate to match expected objectives
                mo_reward = np.zeros((self.n_envs, self.num_objectives))
                mo_reward[:, :min(reward.shape[-1], self.num_objectives)] = reward[:, :min(reward.shape[-1], self.num_objectives)]

        return mo_reward

    def collect_rollouts(
            self,
            env,
            callback,
            train_freq,
            replay_buffer,
            action_noise=None,
            learning_starts=0,
            log_interval=None,
    ):
        """
        Collect rollouts and store them in the replay buffer.
        Custom implementation to handle multi-objective rewards.
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        # Clear history of episodic rewards for individual objectives
        self._last_total_mo_rewards = np.zeros((self.num_envs, self.num_objectives))
        self._episode_mo_rewards = [[] for _ in range(self.num_objectives)]

        # Use parent method with custom reward extraction
        episode_rewards, total_timesteps = [], []
        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "The environment must be vectorized."

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more(
                train_freq,
                num_collected_steps,
                num_collected_episodes,
        ):
            # Select action randomly or according to policy
            if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
                # Warmup phase
                unscaled_action = np.array([self.action_space.sample() for _ in range(env.num_envs)])
            else:
                # Sample actions according to current policy
                unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

            # Rescale and perform action
            action = self._prepare_action(unscaled_action)

            # Add noise to the action
            if action_noise is not None:
                action = self._add_noise_to_action(action, action_noise, env)

            new_obs, reward, terminated, truncated, infos = env.step(action)

            # Extract multi-objective rewards
            mo_reward = self._extract_mo_rewards((new_obs, reward, terminated, truncated, infos))

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None
            if callback.on_step() is False:
                return RolloutReturn(0.0, num_collected_steps, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, terminated)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, action, new_obs, mo_reward, terminated, truncated, infos)

            # Save rewards for each objective
            for i in range(self.num_objectives):
                self._last_total_mo_rewards[:, i] += mo_reward[:, i]

            # Retrieve rewards if using Monitor wrapper
            episode_rewards, total_timesteps = self._get_episode_rewards_timesteps(terminated, truncated, infos)

            # Save data for episodes that have completed
            for i, (done, total_mo_reward) in enumerate(zip(terminated, self._last_total_mo_rewards)):
                if done:
                    # Store individual objective rewards
                    for obj_idx in range(self.num_objectives):
                        self._episode_mo_rewards[obj_idx].append(total_mo_reward[obj_idx])

                    # Update Pareto front
                    self._update_pareto_front(total_mo_reward)

                    # Reset total rewards
                    self._last_total_mo_rewards[i] = 0

            self._last_obs = new_obs
            # Account for the time spent collecting rollouts
            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is done as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            # Log metrics
            if log_interval is not None and self.num_timesteps % log_interval == 0:
                # Log for each objective
                for obj_idx in range(self.num_objectives):
                    if len(self._episode_mo_rewards[obj_idx]) > 0:
                        avg_reward = np.mean(self._episode_mo_rewards[obj_idx])
                        self.logger.record(f"metrics/objective_{obj_idx}_mean_reward", avg_reward)

                # Log Pareto front size
                if len(self.pareto_front) > 0:
                    self.logger.record("metrics/pareto_front_size", len(self.pareto_front))

            if num_collected_episodes >= 1:
                continue_training = callback.on_rollout_end()
                break

        return RolloutReturn(
            np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0,
            num_collected_steps,
            num_collected_episodes,
            continue_training,
        )

    def set_preference_weights(self, preference_weights: np.ndarray) -> None:
        """
        Update preference weights for scalarization.

        Args:
            preference_weights: New preference weights
        """
        assert len(preference_weights) == self.num_objectives, f"Preference weights must have length {self.num_objectives}"

        # Normalize weights to sum to 1
        normalized_weights = np.array(preference_weights) / np.sum(preference_weights)
        self.preference_weights = normalized_weights
        self.preference_weights_tensor = th.FloatTensor(normalized_weights).to(self.device)

        # Log updated weights
        for i, weight in enumerate(self.preference_weights):
            self.logger.record(f"metrics/preference_weight_{i+1}", weight)

    def predict(
            self,
            observation: np.ndarray,
            state=None,
            mask=None,
            deterministic: bool = False,
            preference_weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the policy action from an observation.
        Allow specifying custom preference weights for prediction.
        """
        # Use provided preference weights if given, otherwise use default
        if preference_weights is not None:
            # Save current weights
            old_weights = self.preference_weights
            # Set temporary weights
            normalized_weights = preference_weights / np.sum(preference_weights)
            self.preference_weights = normalized_weights
            self.preference_weights_tensor = th.FloatTensor(normalized_weights).to(self.device)

        # Get standard prediction using parent method
        actions, states = super().predict(observation, state, mask, deterministic)

        # Restore original weights if temporary ones were used
        if preference_weights is not None:
            self.preference_weights = old_weights
            self.preference_weights_tensor = th.FloatTensor(old_weights).to(self.device)

        return actions, states