import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from typing import List, Dict, Any, Optional, Union

# Import the MOReplayBuffer class and its parent class
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecEnv

# Assuming MOReplayBuffer is in a module like mo_rl.buffers
# Replace with the actual import path
from agents.mobuffers import MOReplayBuffer, ReplayBufferSamples


class TestMOReplayBuffer:

    @pytest.fixture
    def buffer(self):
        """Create a basic MOReplayBuffer for testing."""
        # Create observation and action spaces in the format expected by stable-baselines3

        observation_space = spaces.Box(low=np.array([-1.0,-1.0,-1.0,-1.0]), high=np.array([1.0,1.0,1.0,1.0]), shape=(4,), dtype=np.float64)
        action_space = spaces.Box(low=np.array([-1.0,-1.0]), high=np.array([1.0,1.0]), shape=(2,), dtype=np.float64)

        buffer = MOReplayBuffer(
            buffer_size=10,
            observation_space=observation_space,
            action_space=action_space,
            num_objectives=3,
            n_envs=1
        )
        return buffer

    @pytest.fixture
    def multi_env_buffer(self):
        """Create a MOReplayBuffer with multiple environments for testing."""
        observation_space = spaces.Box(low=np.array([-1.0,-1.0,-1.0,-1.0]), high=np.array([1.0,1.0,1.0,1.0]),shape=(4,), dtype=np.float64)
        action_space = spaces.Box(low=np.array([-1.0,-1.0]), high=np.array([1.0,1.0]), shape=(2,), dtype=np.float64)

        buffer = MOReplayBuffer(
            buffer_size=10,
            observation_space=observation_space,
            action_space=action_space,
            num_objectives=3,
            n_envs=2
        )
        return buffer

    def test_initialization(self, buffer):
        """Test that the buffer is initialized correctly with multi-objective rewards."""
        assert buffer.num_objectives == 3
        assert buffer.buffer_size == 10
        assert buffer.rewards.shape == (10, 1, 3)  # (buffer_size, n_envs, num_objectives)
        assert not buffer.full
        assert buffer.pos == 0

    def test_inheritance_from_replay_buffer(self, buffer):
        """Test that MOReplayBuffer correctly inherits from ReplayBuffer."""
        assert isinstance(buffer, ReplayBuffer)

    def test_add_vector_reward(self, buffer):
        """Test adding a transition with a vector reward."""
        obs = np.zeros((4,), dtype=np.float64)
        next_obs = np.ones((4,), dtype=np.float64)
        action = np.array([0.5, -0.5], dtype=np.float64)
        reward = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # 3 objectives
        done = np.array([False])
        infos = [{}]

        buffer.add(obs, next_obs, action, reward, done, infos)

        assert buffer.pos == 1
        assert not buffer.full  # Buffer should not be full yet

        # Check that the data was added correctly
        breakpoint()
        np.testing.assert_array_equal(buffer.observations[0].squeeze(), obs)
        if buffer.optimize_memory_usage:
            np.testing.assert_array_equal(buffer.observations[1].squeeze(), next_obs)
        else:
            np.testing.assert_array_equal(buffer.next_observations[0].squeeze(), next_obs)
        np.testing.assert_array_equal(buffer.actions[0].squeeze(), action)
        np.testing.assert_array_equal(buffer.rewards[0], reward.reshape(1, -1))  # Should be (1, 3)
        np.testing.assert_array_equal(buffer.dones[0], done)

    def test_add_scalar_reward(self, buffer):
        """Test adding a transition with a scalar reward which should be converted to vector."""
        obs = np.zeros((4,), dtype=np.float64)
        next_obs = np.ones((4,), dtype=np.float64)
        action = np.array([0.5, -0.5], dtype=np.float64)
        reward = 1.0  # Scalar reward
        done = np.array([False])
        infos = [{}]

        buffer.add(obs, next_obs, action, reward, done, infos)

        expected_reward = np.zeros((1, 3), dtype=np.float64)
        expected_reward[0, 0] = 1.0

        assert buffer.pos == 1
        np.testing.assert_array_equal(buffer.rewards[0], expected_reward)

    def test_add_multi_env_vector_reward(self, multi_env_buffer):
        """Test adding transitions with vector rewards for multiple environments."""
        obs = np.zeros((2, 4), dtype=np.float64)  # 2 envs, 4 obs dimensions
        next_obs = np.ones((2, 4), dtype=np.float64)
        action = np.array([[0.5, -0.5], [0.3, -0.7]], dtype=np.float64)  # 2 envs, 2 action dimensions
        reward = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)  # 2 envs, 3 objectives
        done = np.array([False, True])
        infos = [{}, {}]

        multi_env_buffer.add(obs, next_obs, action, reward, done, infos)

        assert multi_env_buffer.pos == 1
        np.testing.assert_array_equal(multi_env_buffer.observations[0], obs)
        np.testing.assert_array_equal(multi_env_buffer.rewards[0], reward)  # Should be (2, 3)
        np.testing.assert_array_equal(multi_env_buffer.dones[0], done)

    def test_buffer_wrapping(self, buffer):
        """Test that buffer wraps around correctly when full."""
        for i in range(15):  # More than buffer size
            obs = np.ones((4,), dtype=np.float64) * i
            next_obs = np.ones((4,), dtype=np.float64) * (i + 1)
            action = np.array([0.1, 0.2], dtype=np.float64) * i
            reward = np.array([i, i * 2, i * 3], dtype=np.float64)
            done = np.array([False])
            infos = [{}]

            buffer.add(obs, next_obs, action, reward, done, infos)

        # Buffer should have wrapped around
        assert buffer.full
        assert buffer.pos == 5  # 15 % 10 = 5

        # Check most recent entries (positions 5-9, 0-4)
        for i in range(10):
            pos = (i + 5) % 10
            actual_i = i + 5  # Entries 5 through 14
            expected_reward = np.array([[actual_i, actual_i * 2, actual_i * 3]], dtype=np.float64)
            np.testing.assert_array_equal(buffer.rewards[pos], expected_reward)

    def test_get_samples_vector_rewards(self, buffer):
        """Test that _get_samples returns vector rewards correctly."""
        # Fill buffer with some data
        for i in range(5):
            obs = np.ones((4,), dtype=np.float64) * i
            next_obs = np.ones((4,), dtype=np.float64) * (i + 1)
            action = np.array([0.1, 0.2], dtype=np.float64) * i
            reward = np.array([i, i * 2, i * 3], dtype=np.float64)  # Vector reward
            done = np.array([i % 2 == 0])  # Alternate True/False
            infos = [{}]

            buffer.add(obs, next_obs, action, reward, done, infos)

        # Get samples
        batch_inds = np.array([0, 2, 4])
        samples = buffer._get_samples(batch_inds)

        # Check types and shapes
        assert isinstance(samples, ReplayBufferSamples)
        assert isinstance(samples.rewards, th.Tensor)
        assert samples.rewards.shape == (3, 3)  # (batch_size, num_objectives)

        # Check values
        expected_rewards = th.tensor([[0, 0, 0], [2, 4, 6], [4, 8, 12]], dtype=th.float32).to(buffer.device)
        th.testing.assert_close(samples.rewards, expected_rewards)

    def test_get_samples_with_multiple_envs(self, multi_env_buffer):
        """Test _get_samples with n_envs > 1."""
        n_envs = 2

        # Fill buffer with some data
        for i in range(5):
            obs = np.ones((n_envs, 4), dtype=np.float64) * i
            next_obs = np.ones((n_envs, 4), dtype=np.float64) * (i + 1)
            action = np.ones((n_envs, 2), dtype=np.float64) * i * 0.1
            # Create different rewards for each env
            reward = np.array([[i, i * 2, i * 3], [i + 0.5, (i + 0.5) * 2, (i + 0.5) * 3]], dtype=np.float64)
            done = np.array([i % 2 == 0, (i + 1) % 2 == 0])
            infos = [{} for _ in range(n_envs)]

            multi_env_buffer.add(obs, next_obs, action, reward, done, infos)

        # Get samples
        batch_inds = np.array([0, 2, 4])
        samples = multi_env_buffer._get_samples(batch_inds)

        # Check shapes
        breakpoint()
        assert multi_env_buffer.rewards[batch_inds].shape == (3, n_envs, 3)  # (batch_size, n_envs, num_objectives)
        assert samples.rewards.shape == (3, 3)  # (batch_size, num_objectives)

        # First batch entry should have rewards for first timestep
        expected_first_entry = th.tensor([
            [0.0, 0.0, 0.0], [0.5, 1.0, 1.5]  # First timestep, both envs
        ], dtype=th.float32).to(multi_env_buffer.device)
        assert (samples.rewards[0] == expected_first_entry[0]) or (samples.rewards[0] == expected_first_entry[1])
        

    def test_timeout_termination_handling(self, buffer):
        """Test the handling of timeout terminations."""
        obs = np.zeros((4,), dtype=np.float64)
        next_obs = np.ones((4,), dtype=np.float64)
        action = np.array([0.5, -0.5], dtype=np.float64)
        reward = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        done = np.array([True])
        infos = [{"TimeLimit.truncated": True}]  # Timeout occurred

        buffer.add(obs, next_obs, action, reward, done, infos)

        assert buffer.timeouts[0] == np.array([True])

    def test_reward_dimension_handling(self, buffer):
        """Test various reward dimension handling cases."""
        obs = np.zeros((4,), dtype=np.float64)
        next_obs = np.ones((4,), dtype=np.float64)
        action = np.array([0.5, -0.5], dtype=np.float64)
        done = np.array([False])
        infos = [{}]

        # Test with scalar
        scalar_reward = 5.0
        buffer.add(obs, next_obs, action, scalar_reward, done, infos)
        expected = np.zeros((1, 3), dtype=np.float64)
        expected[0, 0] = 5.0
        np.testing.assert_array_equal(buffer.rewards[0], expected)

        # Test with 1D array
        vector_reward = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        buffer.add(obs, next_obs, action, vector_reward, done, infos)
        np.testing.assert_array_equal(buffer.rewards[1], vector_reward.reshape(1, -1))

        # Test with already shaped 2D array
        shaped_reward = np.array([[4.0, 5.0, 6.0]], dtype=np.float64)
        buffer.add(obs, next_obs, action, shaped_reward, done, infos)
        np.testing.assert_array_equal(buffer.rewards[2], shaped_reward)

    def test_sample_method(self, buffer):
        """Test the sample method for correct behavior."""
        # Fill the buffer with some data
        for i in range(8):
            obs = np.ones((4,), dtype=np.float64) * i
            next_obs = np.ones((4,), dtype=np.float64) * (i + 1)
            action = np.array([0.1, 0.2], dtype=np.float64) * i
            reward = np.array([i, i * 2, i * 3], dtype=np.float64)
            done = np.array([i % 3 == 0])
            infos = [{}]

            buffer.add(obs, next_obs, action, reward, done, infos)

        # Sample from the buffer
        batch_size = 4
        samples = buffer.sample(batch_size)

        # Verify the sample structure and shapes
        assert isinstance(samples, ReplayBufferSamples)
        assert samples.observations.shape[0] == batch_size
        assert samples.actions.shape[0] == batch_size
        assert samples.next_observations.shape[0] == batch_size
        assert samples.dones.shape[0] == batch_size
        assert samples.rewards.shape[0] == batch_size
        assert samples.rewards.shape[1] == 3  # 3 objectives
