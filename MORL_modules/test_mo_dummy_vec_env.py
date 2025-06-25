import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from copy import deepcopy

# Import the MODummyVecEnv class - adjust the import path as needed
from agents.mo_dummy_vec_env import MODummyVecEnv


class ScalarRewardEnv(gym.Env):
    """Simple environment that returns scalar rewards"""

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.current_step = 0

    def reset(self, seed=None, options=None):
        self.current_step = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1
        reward = float(action)  # Reward is just the action value
        done = self.current_step >= 5
        obs = np.ones(2, dtype=np.float32) * self.current_step
        return obs, reward, done, False, {}


class VectorRewardEnv(gym.Env):
    """Environment that returns vector rewards"""

    def __init__(self, num_objectives=3):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.current_step = 0
        self.num_objectives = num_objectives

    def reset(self, seed=None, options=None):
        self.current_step = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1
        # Create a vector reward with different values
        reward = np.arange(1, self.num_objectives + 1, dtype=np.float32) * (action + 1)
        done = self.current_step >= 5
        obs = np.ones(2, dtype=np.float32) * self.current_step
        return obs, reward, done, False, {}


def test_mo_dummy_vec_env_init():
    """Test initialization of MODummyVecEnv with different numbers of objectives"""
    # Test with explicit number of objectives
    env_fn = lambda: ScalarRewardEnv()
    vec_env = MODummyVecEnv([env_fn], num_objectives=4)

    assert vec_env.num_objectives == 4
    assert vec_env.buf_rews.shape == (1, 4)  # (num_envs, num_objectives)

    # Test auto-detection with vector reward env
    env_fn = lambda: VectorRewardEnv(num_objectives=3)
    vec_env = MODummyVecEnv([env_fn])

    assert vec_env.num_objectives == 3
    assert vec_env.buf_rews.shape == (1, 3)


def test_scalar_reward_conversion():
    """Test that scalar rewards are properly converted to vector rewards"""
    env_fn = lambda: ScalarRewardEnv()
    vec_env = MODummyVecEnv([env_fn], num_objectives=3)

    # Reset environment
    obs = vec_env.reset()[0]

    # Take action 1 (should give reward 1.0)
    obs, rewards, dones, infos = vec_env.step([1])

    # Check that reward was converted to vector
    assert rewards.shape == (1, 3)
    assert rewards[0, 0] == 1.0  # First objective should have the scalar value
    assert np.all(rewards[0, 1:] == 0.0)  # Other objectives should be zero

    # Take action 0 (should give reward 0.0)
    obs, rewards, dones, infos = vec_env.step([0])

    # Check that reward was converted to vector
    assert rewards.shape == (1, 3)
    assert rewards[0, 0] == 0.0
    assert np.all(rewards[0, 1:] == 0.0)


def test_vector_reward_handling():
    """Test that vector rewards are properly handled"""
    num_objectives = 4
    env_fn = lambda: VectorRewardEnv(num_objectives=num_objectives)
    vec_env = MODummyVecEnv([env_fn], num_objectives=num_objectives)

    # Reset environment
    obs = vec_env.reset()[0]

    # Take action 1 (should give reward [2, 4, 6, 8])
    obs, rewards, dones, infos = vec_env.step([1])

    # Check that reward vector is preserved
    assert rewards.shape == (1, num_objectives)
    expected_rewards = np.arange(1, num_objectives + 1, dtype=np.float32) * 2
    np.testing.assert_array_equal(rewards[0], expected_rewards)

    # Take action 0 (should give reward [1, 2, 3, 4])
    obs, rewards, dones, infos = vec_env.step([0])

    # Check that reward vector is preserved
    assert rewards.shape == (1, num_objectives)
    expected_rewards = np.arange(1, num_objectives + 1, dtype=np.float32)
    np.testing.assert_array_equal(rewards[0], expected_rewards)


def test_reset_functionality():
    """Test that reset works correctly"""
    env_fn = lambda: VectorRewardEnv(num_objectives=2)
    vec_env = MODummyVecEnv([env_fn], num_objectives=2)

    # Reset environment
    obs = vec_env.reset()[0]
    assert np.all(obs == 0.0)

    # Take steps until done
    for i in range(5):
        obs, rewards, dones, infos = vec_env.step([1])

        if i == 4:  # Last step
            assert dones[0]
            # Check that next observation is from reset
            assert np.all(obs == 0.0)
        else:
            assert not dones[0]
            assert np.all(obs == (i + 1))


def test_multiple_environments():
    """Test with multiple environments"""
    num_envs = 3
    num_objectives = 2

    env_fns = [lambda: VectorRewardEnv(num_objectives=num_objectives) for _ in range(num_envs)]
    vec_env = MODummyVecEnv(env_fns, num_objectives=num_objectives)

    # Check shape of reward buffer
    assert vec_env.buf_rews.shape == (num_envs, num_objectives)

    # Reset environment
    obs = vec_env.reset()
    assert obs.shape == (num_envs, 2)  # (num_envs, obs_dim)

    # Take different actions for each environment
    actions = [0, 1, 0]
    obs, rewards, dones, infos = vec_env.step(actions)

    # Check rewards for each environment
    assert rewards.shape == (num_envs, num_objectives)

    # Environment 0 took action 0, so rewards should be [1, 2]
    np.testing.assert_array_equal(rewards[0], [1, 2])

    # Environment 1 took action 1, so rewards should be [2, 4]
    np.testing.assert_array_equal(rewards[1], [2, 4])

    # Environment 2 took action 0, so rewards should be [1, 2]
    np.testing.assert_array_equal(rewards[2], [1, 2])


def test_mixed_reward_types():
    """Test handling of different reward types"""

    class MixedRewardEnv(gym.Env):
        def __init__(self, reward_type='scalar'):
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            self.reward_type = reward_type

        def reset(self, seed=None, options=None):
            return np.zeros(2, dtype=np.float32), {}

        def step(self, action):
            if self.reward_type == 'scalar':
                reward = 1.0
            elif self.reward_type == 'vector3':
                reward = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            elif self.reward_type == 'vector2':
                reward = np.array([1.0, 2.0], dtype=np.float32)
            elif self.reward_type == 'list':
                reward = [1.0, 2.0, 3.0]
            else:
                reward = 0.0

            return np.ones(2, dtype=np.float32), reward, False, False, {}

    # Test with 3 objectives but reward has only 2 elements
    env_fn = lambda: MixedRewardEnv(reward_type='vector2')
    vec_env = MODummyVecEnv([env_fn], num_objectives=3)

    # Reset and step
    vec_env.reset()
    obs, rewards, dones, infos = vec_env.step([0])

    # Should pad the reward
    assert rewards.shape == (1, 3)
    np.testing.assert_array_equal(rewards[0, :2], [1.0, 2.0])
    assert rewards[0, 2] == 0.0

    # Test with list reward
    env_fn = lambda: MixedRewardEnv(reward_type='list')
    vec_env = MODummyVecEnv([env_fn], num_objectives=3)

    # Reset and step
    vec_env.reset()
    obs, rewards, dones, infos = vec_env.step([0])

    # Should convert list to numpy array
    assert rewards.shape == (1, 3)
    np.testing.assert_array_equal(rewards[0], [1.0, 2.0, 3.0])


def test_comparison_with_dummy_vec_env():
    """Test that MODummyVecEnv behaves like DummyVecEnv for scalar rewards"""

    # Create standard DummyVecEnv
    env_fn = lambda: ScalarRewardEnv()
    std_vec_env = DummyVecEnv([env_fn])

    # Create MODummyVecEnv with 1 objective
    mo_vec_env = MODummyVecEnv([env_fn], num_objectives=1)

    # Reset both environments
    std_obs = std_vec_env.reset()[0]
    mo_obs = mo_vec_env.reset()[0]

    # Check that observations are the same
    np.testing.assert_array_equal(std_obs, mo_obs)

    # Step both environments with the same action
    std_obs, std_rewards, std_dones, std_infos = std_vec_env.step([1])
    mo_obs, mo_rewards, mo_dones, mo_infos = mo_vec_env.step([1])

    # Check that observations and dones are the same
    np.testing.assert_array_equal(std_obs, mo_obs)
    np.testing.assert_array_equal(std_dones, mo_dones)

    # Check that rewards match (accounting for different shapes)
    assert std_rewards[0] == mo_rewards[0, 0]


def test_done_handling_and_automatic_reset():
    """Test that done episodes are properly handled and environments are reset"""
    env_fn = lambda: ScalarRewardEnv()
    vec_env = MODummyVecEnv([env_fn], num_objectives=2)

    # Reset environment
    obs = vec_env.reset()[0]

    # Take steps until done (should happen after 5 steps)
    for i in range(10):  # More than needed to trigger done
        action = 1
        obs, rewards, dones, infos = vec_env.step([action])

        if i == 4 or i == 9:  # Episode should end after 5 steps
            assert dones[0]
            # Terminal observation should be stored in info
            assert "terminal_observation" in infos[0]
            # Next observation should be from reset
            assert np.all(obs == 0.0)
        elif i > 4:
            # We're in a new episode
            step_in_new_episode = i - 5
            assert np.all(obs == (step_in_new_episode + 1))


def test_terminal_observation_saving():
    """Test that terminal observations are properly saved in info dict"""
    env_fn = lambda: ScalarRewardEnv()
    vec_env = MODummyVecEnv([env_fn], num_objectives=2)

    # Reset environment
    vec_env.reset()

    # Take steps until done
    terminal_obs = None
    for _ in range(5):  # Should be done after 5 steps
        obs, rewards, dones, infos = vec_env.step([1])
        if dones[0]:
            terminal_obs = infos[0]["terminal_observation"]

    # Terminal observation should be the state at step 5
    assert terminal_obs is not None
    np.testing.assert_array_equal(terminal_obs, np.ones(2, dtype=np.float32) * 5)


if __name__ == "__main__":
    # Run tests
    test_mo_dummy_vec_env_init()
    test_scalar_reward_conversion()
    test_vector_reward_handling()
    test_reset_functionality()
    test_multiple_environments()
    test_mixed_reward_types()
    test_comparison_with_dummy_vec_env()
    test_done_handling_and_automatic_reset()
    test_terminal_observation_saving()
    print("All tests passed!")