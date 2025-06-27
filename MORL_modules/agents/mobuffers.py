import numpy as np
import torch as th
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.buffers import ReplayBufferSamples
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.sac.sac import SAC
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.type_aliases import GymEnv, Schedule, TensorDict
import gymnasium as gym
from stable_baselines3.common.torch_layers import FlattenExtractor
import pdb
from stable_baselines3.common.type_aliases import GymEnv, Schedule, TensorDict, RolloutReturn
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import polyak_update, should_collect_more_steps
from stable_baselines3.common.torch_layers import FlattenExtractor

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
            buffer_size = buffer_size,
            observation_space = observation_space,
            action_space = action_space,
            device = device,
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
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

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
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        # If the environment is not provided, we assume normalization is not needed
        # Normalize observations and rewards
        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices, :], env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

