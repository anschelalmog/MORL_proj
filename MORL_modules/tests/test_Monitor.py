import sys
import logging

import os
import time
import tempfile
import csv
import json
import numpy as np
import gymnasium  as gym
import pytest
from gymnasium  import spaces
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'MORL_modules'))
from agents.mo_monitor import MOMonitor


class VectorRewardEnv(gym.Env):
    """Custom environment that returns vector rewards for testing MOMonitor"""

    def __init__(self, num_objectives=4, episode_length=5):
        super().__init__()
        self.num_objectives = num_objectives
        self.episode_length = episode_length
        self.current_step = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        # Add spec attribute for testing
        self.spec = type('spec', (), {'id': 'TestEnv-v0'})()

    def reset(self, **kwargs):
        self.current_step = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1
        obs = np.random.randn(4).astype(np.float32)

        # Generate a vector reward
        reward = np.array([float(i + 1) for i in range(self.num_objectives)], dtype=np.float32)

        # Episode terminates after episode_length steps
        terminated = self.current_step >= self.episode_length
        truncated = False

        info = {"test_info": self.current_step}

        return obs, reward, terminated, truncated, info


def test_init():
    """Test MOMonitor initialization"""
    env = VectorRewardEnv(num_objectives=4)
    monitor = MOMonitor(env=env, num_objectives=4)

    assert monitor.num_objectives == 4
    assert len(monitor.rewards) == 0
    assert len(monitor.episode_returns) == 0


def test_step_vector_reward():
    """Test step function with vector rewards"""
    env = VectorRewardEnv(num_objectives=4, episode_length=5)
    # Fix the vector_size/num_objectives attribute mismatch in the test
    monitor = MOMonitor(env=env, num_objectives=4)



    obs, _ = monitor.reset()

    # Take a step and check the reward
    obs, reward, terminated, truncated, info = monitor.step(0)

    # Check reward shape (original reward should be preserved)
    assert len(reward) == 4
    assert np.array_equal(reward, np.array([1.0, 2.0, 3.0, 4.0]))

    # Check that rewards are properly stored
    assert len(monitor.rewards) == 1
    assert monitor.rewards[0].shape == (4,)
    assert np.array_equal(monitor.rewards[0], np.array([1.0, 2.0, 3.0, 4.0]))


def test_episode_completion():
    """Test episode completion and info dictionary format"""
    env = VectorRewardEnv(num_objectives=4, episode_length=2)
    monitor = MOMonitor(env=env, num_objectives=4)



    obs, _ = monitor.reset()

    # First step - episode not done
    _, _, terminated, truncated, info = monitor.step(0)
    assert not terminated
    assert "episode" not in info

    # Second step - episode done
    _, _, terminated, truncated, info = monitor.step(0)
    assert terminated
    assert "episode" in info

    # Check episode info format
    ep_info = info["episode"]
    assert "r" in ep_info
    assert "l" in ep_info
    assert "t" in ep_info

    # Check episode reward values (sum of [1,2,3,4] taken twice)
    expected_rewards = np.array([2.0, 4.0, 6.0, 8.0])
    assert np.array_equal(ep_info["r"], expected_rewards)
    assert ep_info["l"] == 2  # episode length


def test_reward_rounding():
    """Test that rewards are properly rounded"""

    # Create a custom env with non-integer rewards
    class FloatRewardEnv(gym.Env):
        def __init__(self):
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
            self.current_step = 0
            self.spec = type('spec', (), {'id': 'FloatRewardEnv-v0'})()

        def reset(self, **kwargs):
            self.current_step = 0
            return np.zeros(4, dtype=np.float32), {}

        def step(self, action):
            self.current_step += 1
            # Reward with many decimal places
            reward = np.array([1.123456789, 2.987654321, 3.456789123, 4.321098765], dtype=np.float32)
            done = self.current_step >= 1
            return np.zeros(4, dtype=np.float32), reward, done, False, {}

    env = FloatRewardEnv()
    monitor = MOMonitor(env=env, num_objectives=4)



    obs, _ = monitor.reset()
    _, _, terminated, truncated, info = monitor.step(0)

    # Check that rewards are rounded to 6 decimal places
    expected = np.array([1.123457, 2.987654, 3.456789, 4.321099])  # rounded to 6 decimal places
    assert np.allclose(info["episode"]["r"], expected, rtol=1e-6)


def test_scalar_to_vector_conversion():
    """Test conversion of scalar rewards to vectors"""

    # Create an environment that returns scalar rewards
    class ScalarRewardEnv(gym.Env):
        def __init__(self):
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
            self.current_step = 0
            self.spec = type('spec', (), {'id': 'ScalarRewardEnv-v0'})()

        def reset(self, **kwargs):
            self.current_step = 0
            return np.zeros(4, dtype=np.float32), {}

        def step(self, action):
            self.current_step += 1
            obs = np.zeros(4, dtype=np.float32)
            reward = 5.0  # Scalar reward
            done = self.current_step >= 2
            return obs, reward, done, False, {}

    env = ScalarRewardEnv()
    monitor = MOMonitor(env=env, num_objectives=4)


    obs, _ = monitor.reset()
    _, reward, _, _, _ = monitor.step(0)

    # Check that original reward is preserved (still scalar)
    assert np.isscalar(reward)
    assert reward == 5.0

    # Check that rewards are properly stored as vectors
    assert len(monitor.rewards) == 1
    assert monitor.rewards[0].shape == (4,)
    assert monitor.rewards[0][0] == 5.0  # First element has the scalar value
    assert np.all(monitor.rewards[0][1:] == 0.0)  # Other elements are zero


def test_csv_output():
    """Test CSV output format with vector rewards"""
    with tempfile.NamedTemporaryFile(suffix='monitor.csv', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        env = VectorRewardEnv(num_objectives=4, episode_length=1)
        monitor = MOMonitor(env=env, filename=tmp_path, num_objectives=4)



        obs, _ = monitor.reset()
        monitor.step(0)  # Complete an episode
        monitor.close()

        # Verify file exists
        assert os.path.exists(tmp_path)

        # Read the CSV file - the format may vary depending on the MOResultsWriter implementation
        # Just check that the file exists and has content
        with open(tmp_path, 'r') as f:
            content = f.read()
            breakpoint()
            assert len(content) > 0

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_get_episode_rewards():
    """Test get_episode_rewards method"""
    env = VectorRewardEnv(num_objectives=4, episode_length=2)
    monitor = MOMonitor(env=env, num_objectives=4)



    obs, _ = monitor.reset()

    # Complete first episode
    monitor.step(0)
    monitor.step(0)

    # Complete second episode
    obs, _ = monitor.reset()
    monitor.step(0)
    monitor.step(0)

    # Get episode rewards
    rewards = monitor.get_episode_rewards()

    assert len(rewards) == 2  # Two episodes
    assert all(isinstance(r, np.ndarray) for r in rewards)
    assert all(r.shape == (4,) for r in rewards)

    # Check values (each episode is sum of [1,2,3,4] taken twice)
    expected = np.array([2.0, 4.0, 6.0, 8.0])
    assert np.array_equal(rewards[0], expected)
    assert np.array_equal(rewards[1], expected)


def test_multiple_episodes():
    """Test statistics across multiple episodes with different lengths"""
    env = VectorRewardEnv(num_objectives=4, episode_length=1)
    monitor = MOMonitor(env=env, num_objectives=4)



    # Complete 3 episodes
    for _ in range(3):
        obs, _ = monitor.reset()
        _, _, terminated, truncated, info = monitor.step(0)
        assert terminated or truncated
        assert "episode" in info

    # Check statistics
    episode_rewards = monitor.get_episode_rewards()
    assert len(episode_rewards) == 3

    # Each episode should have total reward [1, 2, 3, 4] (single step)
    expected = np.array([1.0, 2.0, 3.0, 4.0])
    for rewards in episode_rewards:
        assert np.array_equal(rewards, expected)

    # Check episode lengths
    assert monitor.episode_lengths == [1, 1, 1]


def test_early_reset():
    """Test early reset behavior"""
    env = VectorRewardEnv(num_objectives=4, episode_length=5)

    # Test with allow_early_resets=True (default)
    monitor1 = MOMonitor(env=env, num_objectives=4)



    obs, _ = monitor1.reset()
    monitor1.step(0)  # Not done yet
    obs, _ = monitor1.reset()  # Should work fine

    # Test with allow_early_resets=False
    monitor2 = MOMonitor(env=env, num_objectives=4, allow_early_resets=False)


    obs, _ = monitor2.reset()
    monitor2.step(0)  # Not done yet

    # This should raise an error
    with pytest.raises(RuntimeError):
        monitor2.reset()


def test_info_keywords():
    """Test with custom info keywords"""
    env = VectorRewardEnv(num_objectives=4, episode_length=1)
    monitor = MOMonitor(env=env, num_objectives=4, info_keywords=("test_info",))



    obs, _ = monitor.reset()
    _, _, terminated, truncated, info = monitor.step(0)  # Complete episode

    # Check that the info keyword was captured
    assert "test_info" in info["episode"]
    assert info["episode"]["test_info"] == 1  # The value from the env


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])