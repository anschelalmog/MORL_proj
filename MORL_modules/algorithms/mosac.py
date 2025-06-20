import numpy as np
import torch as th
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable
from gymnasium import spaces

from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.sac.sac import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.type_aliases import GymEnv, Schedule, TensorDict, RolloutReturn
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import polyak_update, should_collect_more_steps
from stable_baselines3.common.torch_layers import FlattenExtractor

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

        if features_extractor_class is None:
            features_extractor_class = FlattenExtractor
        features_extractor = features_extractor_class(observation_space, **( features_extractor_kwargs or {}))
        super(ContinuousCritic, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images=normalize_images)

        """
        super(ContinuousCritic, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images=normalize_images,
        )
        """
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

        # Override rewards buffer to store vectors instead of scalars
        # Shape becomes (buffer_size, n_envs, num_objectives)
        self.rewards = np.zeros((self.buffer_size, self.n_envs, self.num_objectives), dtype=np.float32)

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,  # Now expects vector reward
            done: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:
        """Add a new transition to the buffer with vector reward."""
        # Ensure reward has correct shape
        if isinstance(reward, (int, float)):
            # Convert scalar to vector (put it in first objective)
            reward_vec = np.zeros(self.num_objectives)
            reward_vec[0] = reward
            reward = reward_vec

        if reward.ndim == 1:
            reward = reward.reshape(1, -1)  # Add batch dimension

        # Copy to avoid modification of external array
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()  # Vector reward
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecEnv] = None) -> TensorDict:
        """Get samples from the buffer, handling vector rewards."""
        # Get the standard samples
        data = super()._get_samples(batch_inds, env)
        breakpoint()
        # Ensure rewards are properly shaped vectors
        if self.rewards[batch_inds].ndim == 3:  # (batch, n_envs, n_objectives)
            # Squeeze out n_envs dimension if it's 1
            if self.rewards[batch_inds].shape[1] == 1:
                rewards_tensor = th.tensor(self.rewards[batch_inds].squeeze(), dtype=th.float32).to(self.device)
                breakpoint()
            else:
                rewards_tensor=  th.tensor(self.rewards[batch_inds], dtype=th.float32).to(self.device)
        else:
            rewards_tensor =  th.tensor(self.rewards[batch_inds], dtype=th.float32).to(self.device)

        return ReplayBufferSamples( observations=data.observations,
                                    actions=data.actions,
                                    next_observations=data.next_observations,
                                    rewards=rewards_tensor,
                                    dones=data.dones)

class MOSAC(SAC):
    """
    Multi-Objective Soft Actor-Critic algorithm.
    """

    def __init__(self, *args, **kwargs):
        self.num_objectives = kwargs.pop('num_objectives', 4)
        preference_weights = kwargs.pop('preference_weights', None)
        self.hypervolume_ref_point = kwargs.pop('hypervolume_ref_point', None)

        # Set up preference weights
        if preference_weights is None:
            self.preference_weights = np.ones(self.num_objectives) / self.num_objectives
        else:
            assert len(preference_weights) == self.num_objectives
            self.preference_weights = np.array(preference_weights) / np.sum(preference_weights)

        # Ensure policy kwargs contain num_objectives
        policy_kwargs = kwargs.get('policy_kwargs', {})
        policy_kwargs['num_objectives'] = self.num_objectives
        kwargs['policy_kwargs'] = policy_kwargs

        # Ensure replay buffer kwargs contain num_objectives
        replay_buffer_kwargs = kwargs.get('replay_buffer_kwargs', {})
        replay_buffer_kwargs['num_objectives'] = self.num_objectives
        kwargs['replay_buffer_kwargs'] = replay_buffer_kwargs

        # Initialize tracking for objectives
        self._episode_mo_rewards = [[] for _ in range(self.num_objectives)]
        self._last_mo_episode_rewards = np.zeros(self.num_objectives)
        self.pareto_front = []

        super().__init__(*args, **kwargs)

        # Initialize preference weights tensor after device is known
        self.preference_weights_tensor = th.FloatTensor(self.preference_weights).to(self.device)

    def _setup_model(self) -> None:
        """Setup model with multi-objective components."""
        # Set default replay buffer class if not specified
        if self.replay_buffer_class is None:
            self.replay_buffer_class = MOReplayBuffer

        super()._setup_model()
        self.preference_weights_tensor = th.FloatTensor(self.preference_weights).to(self.device)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """Train the model with multi-objective rewards."""
        # Update learning rate
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses, ent_coef_losses = [], [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled states
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)

                # Compute target Q-values for each objective
                # Assuming rewards has shape (batch_size, num_objectives)
                target_q_values = []
                for obj_idx in range(self.num_objectives):
                    obj_reward = replay_data.rewards[:, obj_idx].reshape(-1, 1)
                    obj_target = obj_reward + (1 - replay_data.dones) * self.gamma * next_q_values
                    target_q_values.append(obj_target)

                # Scalarize target using preference weights
                target_q_value = sum(w * tq for w, tq in zip(self.preference_weights, target_q_values))

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss - for SAC, we typically have 2 critics
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_value) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see https://github.com/DLR-RM/stable-baselines3/issues/996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coef.cpu().detach().numpy()))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def _store_transition(
            self,
            replay_buffer: MOReplayBuffer,
            buffer_action: np.ndarray,
            new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
            reward: np.ndarray,  # Now vector reward
            dones: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:
        """Store transition in the replay buffer, handling vector rewards."""
        # Store transition
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            new_obs_ = new_obs
            reward_ = reward

        replay_buffer.add(
            self._last_original_obs,
            new_obs_,
            buffer_action,
            reward_,  # Vector reward
            dones,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_