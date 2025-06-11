import numpy as np
import pytest
from unittest.mock import MagicMock
import torch as th
import gymnasium as gym

from algorithms.mosac import MOReplayBuffer  # Replace with actual import path

@pytest.fixture
def buffer():
    buffer_size = 10
    num_envs = 1
    num_objectives = 3
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)
    return MOReplayBuffer(
        buffer_size=buffer_size,
        observation_space=obs_space,
        action_space=action_space,
        num_objectives=num_objectives,
        device="cpu",
        n_envs=num_envs,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
    )

def test_add_scalar_reward_converted_to_vector(buffer):
    obs = np.array([[0.1, 0.2, 0.3, 0.4]])
    next_obs = np.array([[0.5, 0.6, 0.7, 0.8]])
    action = np.array([[1]])
    scalar_reward = np.array([1.0, 0,0])
    done = np.array([False])
    infos = [{}]

    buffer.add(obs, next_obs, action, scalar_reward, done, infos)

    # Check that reward is stored as a vector
    stored_reward = buffer.rewards[0, 0]

    assert stored_reward.shape == (buffer.num_objectives,)
    assert stored_reward[0] == 1.0
    breakpoint()
    assert np.all(stored_reward[1:] == 0)

def test_add_vector_reward(buffer):
    obs = np.array([[0.1, 0.2, 0.3, 0.4]])
    next_obs = np.array([[0.5, 0.6, 0.7, 0.8]])
    action = np.array([[1]])
    vector_reward = np.array([[1.0, 0.5, -0.2]])
    done = np.array([False])
    infos = [{}]

    buffer.add(obs, next_obs, action, vector_reward, done, infos)

    stored_reward = buffer.rewards[0, 0]
    assert np.allclose(stored_reward, vector_reward[0])

def test_sample_shape(buffer):
    # Fill buffer
    for i in range(buffer.buffer_size):
        obs = np.random.rand(1, 4).astype(np.float32)
        next_obs = np.random.rand(1, 4).astype(np.float32)
        action = np.array([[1]])
        reward = np.random.rand(1, buffer.num_objectives).astype(np.float32)
        done = np.array([False])
        infos = [{}]
        buffer.add(obs, next_obs, action, reward, done, infos)

    batch_inds = np.random.randint(0, buffer.buffer_size, size=5)
    samples = buffer._get_samples(batch_inds)

    assert hasattr(samples, "rewards")
    breakpoint()
    assert samples.rewards.shape == (5, buffer.num_objectives)
    assert isinstance(samples.rewards, th.Tensor)
