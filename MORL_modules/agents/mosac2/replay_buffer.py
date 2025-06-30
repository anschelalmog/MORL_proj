# MORL_modules/agents/mosac/replay_buffer.py

import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from typing import Any, Dict, List, Optional, Union
import warnings


class MOSACReplayBuffer(ReplayBuffer):
    """
    Extended replay buffer for multi-objective rewards.
    Stores vector rewards instead of scalar rewards.

    This buffer is specifically designed for multi-objective reinforcement learning
    where each transition has a vector of rewards corresponding to different objectives.
    """

    def __init__(
            self,
            buffer_size: int,
            observation_space,
            action_space,
            device: Union[torch.device, str] = "auto",
            n_envs: int = 1,
            num_objectives: int = 4,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
    ):
        """
        Initialize multi-objective replay buffer

        Args:
            buffer_size: Maximum number of transitions to store
            observation_space: Observation space
            action_space: Action space
            device: PyTorch device
            n_envs: Number of parallel environments
            num_objectives: Number of objectives (reward dimensions)
            optimize_memory_usage: Whether to optimize memory usage
            handle_timeout_termination: Whether to handle timeout terminations
        """

        # Initialize parent with dummy scalar reward
        # We'll override the rewards buffer afterwards
        super().__init__(
            buffer_size, observation_space, action_space, device,
            n_envs, optimize_memory_usage, handle_timeout_termination
        )

        self.num_objectives = num_objectives
        self.objective_names = ["economic", "battery_health", "grid_support", "autonomy"]

        # Override reward buffer to store vector rewards
        self.rewards = np.zeros((self.buffer_size, self.n_envs, num_objectives), dtype=np.float32)

        # Additional tracking for statistics
        self.episode_statistics = {
            'total_episodes': 0,
            'objective_sums': np.zeros(num_objectives),
            'objective_counts': np.zeros(num_objectives),
            'episode_lengths': [],
        }

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,  # Vector reward [num_objectives] or scalar
            done: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:
        """
        Add experience to buffer with vector rewards

        Args:
            obs: Current observation
            next_obs: Next observation
            action: Action taken
            reward: Reward received (vector or scalar)
            done: Episode termination flag
            infos: Additional information
        """
        # Convert reward to proper format
        reward = self._process_reward(reward)

        # Store vector reward
        self.rewards[self.pos] = reward

        # Store other data using parent method but override reward with scalar version
        # Use first objective for parent class compatibility
        scalar_reward = reward[:, 0] if reward.shape[1] > 0 else np.zeros((self.n_envs,))

        # Call parent's add method but we'll override some storage
        temp_pos = self.pos
        super().add(obs, next_obs, action, scalar_reward, done, infos)

        # Restore our vector rewards (parent method might have overwritten)
        self.rewards[temp_pos] = reward

        # Update statistics
        self._update_statistics(reward, done)

    def _process_reward(self, reward: Union[np.ndarray, float, Dict]) -> np.ndarray:
        """
        Process reward into the correct vector format

        Args:
            reward: Input reward in various formats

        Returns:
            Processed reward as [n_envs, num_objectives] array
        """
        if isinstance(reward, dict):
            # Extract multi-objective rewards from dictionary
            if 'mo_rewards' in reward:
                reward = reward['mo_rewards']
            elif 'pcs' in reward:
                # Single scalar reward, put in first objective
                reward = np.array([reward['pcs']], dtype=np.float32)
            else:
                # Use first available reward
                reward = np.array([list(reward.values())[0]], dtype=np.float32)

        # Convert to numpy array
        reward = np.array(reward, dtype=np.float32)

        # Ensure correct shape
        if reward.ndim == 0:
            # Scalar reward - convert to vector (put in first objective)
            vector_reward = np.zeros(self.num_objectives, dtype=np.float32)
            vector_reward[0] = reward
            reward = vector_reward
        elif reward.ndim == 1 and len(reward) != self.num_objectives:
            # Wrong vector size
            if len(reward) == 1:
                # Single element - put in first objective
                vector_reward = np.zeros(self.num_objectives, dtype=np.float32)
                vector_reward[0] = reward[0]
                reward = vector_reward
            else:
                warnings.warn(f"Reward vector size {len(reward)} doesn't match num_objectives {self.num_objectives}")
                # Pad or truncate as needed
                vector_reward = np.zeros(self.num_objectives, dtype=np.float32)
                min_len = min(len(reward), self.num_objectives)
                vector_reward[:min_len] = reward[:min_len]
                reward = vector_reward

        # Ensure shape is [n_envs, num_objectives]
        if reward.ndim == 1:
            reward = reward.reshape(1, -1)  # [1, num_objectives]

        # Repeat for n_envs if necessary
        if reward.shape[0] != self.n_envs:
            reward = np.repeat(reward, self.n_envs, axis=0)

        return reward

    def _update_statistics(self, reward: np.ndarray, done: np.ndarray):
        """Update episode statistics"""
        # Update cumulative statistics
        self.episode_statistics['objective_sums'] += np.sum(reward, axis=0)
        self.episode_statistics['objective_counts'] += reward.shape[0]

        # Track episode completions
        if np.any(done):
            self.episode_statistics['total_episodes'] += int(np.sum(done))

    def sample(self, batch_size: int, env=None) -> Dict[str, torch.Tensor]:
        """
        Sample batch with vector rewards

        Args:
            batch_size: Number of transitions to sample
            env: Environment (for normalization, not used here)

        Returns:
            Dictionary containing batch data with vector rewards
        """
        batch_inds = np.random.randint(0, self.size(), size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> Dict[str, torch.Tensor]:
        """
        Get samples with vector rewards

        Args:
            batch_inds: Indices of transitions to sample
            env: Environment (for normalization)

        Returns:
            Dictionary containing sampled batch data
        """
        # Get base samples
        data = {
            "observations": self.to_torch(self.observations[batch_inds, 0, :]),
            "actions": self.to_torch(self.actions[batch_inds, 0, :]),
            "next_observations": self.to_torch(self.next_observations[batch_inds, 0, :]),
            "dones": self.to_torch(self.dones[batch_inds]),
            "rewards": self.to_torch(self.rewards[batch_inds, 0, :]),  # Vector rewards
        }

        return data

    def get_objective_statistics(self) -> Dict[str, float]:
        """
        Get statistics for each objective

        Returns:
            Dictionary with statistics for each objective
        """
        stats = {}

        if self.episode_statistics['objective_counts'][0] > 0:
            means = self.episode_statistics['objective_sums'] / self.episode_statistics['objective_counts']

            for i, obj_name in enumerate(self.objective_names):
                if i < len(means):
                    stats[f'{obj_name}_mean'] = float(means[i])

        stats['total_episodes'] = self.episode_statistics['total_episodes']
        stats['buffer_size'] = self.size()

        return stats

    def reset_statistics(self):
        """Reset episode statistics"""
        self.episode_statistics = {
            'total_episodes': 0,
            'objective_sums': np.zeros(self.num_objectives),
            'objective_counts': np.zeros(self.num_objectives),
            'episode_lengths': [],
        }

    def get_recent_rewards(self, n_recent: int = 1000) -> np.ndarray:
        """
        Get recent rewards for analysis

        Args:
            n_recent: Number of recent transitions to return

        Returns:
            Recent rewards array [n_recent, num_objectives]
        """
        if self.size() == 0:
            return np.zeros((0, self.num_objectives))

        start_idx = max(0, self.size() - n_recent)

        if self.full:
            # Buffer is full, need to handle wraparound
            if start_idx < self.pos:
                # Recent data doesn't wrap around
                recent_rewards = self.rewards[start_idx:self.pos, 0, :]
            else:
                # Recent data wraps around
                part1 = self.rewards[start_idx:self.buffer_size, 0, :]
                part2 = self.rewards[0:self.pos, 0, :]
                recent_rewards = np.concatenate([part1, part2], axis=0)
        else:
            # Buffer not full yet
            recent_rewards = self.rewards[start_idx:self.pos, 0, :]

        return recent_rewards

    def get_reward_correlations(self) -> np.ndarray:
        """
        Compute correlation matrix between objectives

        Returns:
            Correlation matrix [num_objectives, num_objectives]
        """
        recent_rewards = self.get_recent_rewards()

        if recent_rewards.shape[0] < 2:
            return np.eye(self.num_objectives)

        return np.corrcoef(recent_rewards.T)

    def save_buffer_state(self, filepath: str):
        """
        Save buffer state to file

        Args:
            filepath: Path to save the buffer state
        """
        state = {
            'rewards': self.rewards,
            'observations': self.observations,
            'actions': self.actions,
            'next_observations': self.next_observations,
            'dones': self.dones,
            'pos': self.pos,
            'full': self.full,
            'statistics': self.episode_statistics,
            'num_objectives': self.num_objectives,
        }

        np.savez_compressed(filepath, **state)

    def load_buffer_state(self, filepath: str):
        """
        Load buffer state from file

        Args:
            filepath: Path to load the buffer state from
        """
        data = np.load(filepath)

        # Verify compatibility
        if data['num_objectives'].item() != self.num_objectives:
            raise ValueError(f"Loaded buffer has {data['num_objectives'].item()} objectives, "
                             f"but current buffer expects {self.num_objectives}")

        # Load state
        self.rewards = data['rewards']
        self.observations = data['observations']
        self.actions = data['actions']
        self.next_observations = data['next_observations']
        self.dones = data['dones']
        self.pos = data['pos'].item()
        self.full = data['full'].item()
        self.episode_statistics = data['statistics'].item()

    def __len__(self) -> int:
        """Return current buffer size"""
        return self.size()

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get specific transition by index"""
        if idx >= self.size():
            raise IndexError(f"Index {idx} out of range for buffer size {self.size()}")

        return {
            'observation': self.observations[idx, 0, :],
            'action': self.actions[idx, 0, :],
            'reward': self.rewards[idx, 0, :],
            'next_observation': self.next_observations[idx, 0, :],
            'done': self.dones[idx],
        }