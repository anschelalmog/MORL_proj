import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
import torch as th

# Import your MOSAC class
from agents.mosac import MOSAC


class SimpleScalarEnv(gym.Env):
    """Simple environment that returns scalar rewards"""

    def __init__(self):
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32)
        self.current_step = 0

    def reset(self, seed=None, options=None):
        self.current_step = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1
        reward = float(np.sum(action))  # Scalar reward
        done = self.current_step >= 5
        obs = np.ones(4, dtype=np.float32) * self.current_step
        return obs, reward, done, False, {}


class SimpleVectorEnv(gym.Env):
    """Environment that already returns vector rewards"""

    def __init__(self, num_objectives=4):
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32)
        self.current_step = 0
        self.num_objectives = num_objectives

    def reset(self, seed=None, options=None):
        self.current_step = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1
        # Vector reward with different values for each objective
        reward = np.ones(self.num_objectives) * np.sum(action) * np.arange(1, self.num_objectives + 1)
        done = self.current_step >= 5
        obs = np.ones(4, dtype=np.float32) * self.current_step
        return obs, reward, done, False, {}


def test_wrap_env_scalar_reward():
    """Test wrapping an environment with scalar rewards"""
    # Create a MOSAC instance
    env = SimpleScalarEnv()
    num_objectives = 4

    # Create MOSAC agent with specific num_objectives
    model = MOSAC(
        policy="MOSACPolicy",
        env=env,
        num_objectives=num_objectives,
        learning_rate=3e-4,
        buffer_size=1000,
        learning_starts=0,
        _init_setup_model=False  # Don't set up model automatically
    )

    # Call the _wrap_env method directly
    wrapped_env = model._wrap_env(env, verbose=0)

    # Check that the wrapped environment is a VecEnv
    assert isinstance(wrapped_env, VecEnv)

    # Check that the environment has the correct number of objectives
    assert hasattr(wrapped_env, 'num_objectives')
    assert wrapped_env.num_objectives == num_objectives

    # Reset the environment
    obs = wrapped_env.reset()[0]

    # Take a step and verify the reward shape
    action = np.array([[0.5, -0.2]])  # VecEnv expects batch of actions
    next_obs, reward, done, info = wrapped_env.step(action)

    # Check that reward is a vector with the correct shape
    assert reward.shape == (1, num_objectives)

    # Since we're using default reward weights ([1,0,0,0]), all reward should be in first objective
    scalar_reward = np.sum(action[0])  # The original scalar reward
    assert reward[0, 0] == pytest.approx(scalar_reward)
    assert np.all(reward[0, 1:] == 0.0)  # Other objectives should be zero


def test_wrap_env_custom_reward_weights():
    """Test wrapping with custom reward weights"""
    # Create a MOSAC instance
    env = SimpleScalarEnv()
    num_objectives = 4
    reward_weights = np.array([0.1, 0.2, 0.3, 0.4])  # Custom weights

    # Create MOSAC agent with specific reward_weights
    model = MOSAC(
        policy="MOSACPolicy",
        env=env,
        num_objectives=num_objectives,
        preference_weights=reward_weights,
        learning_rate=3e-4,
        buffer_size=1000,
        learning_starts=0,
        _init_setup_model=False
    )

    # Call the _wrap_env method
    wrapped_env = model._wrap_env(env, verbose=0)

    # Reset and take a step
    obs = wrapped_env.reset()[0]
    action = np.array([[0.5, -0.2]])
    next_obs, reward, done, info = wrapped_env.step(action)

    # Check that reward is distributed according to reward_weights
    scalar_reward = np.sum(action[0])
    normalized_weights = reward_weights / np.sum(reward_weights)
    expected_reward = normalized_weights * scalar_reward

    # Check each objective's reward
    for i in range(num_objectives):
        assert reward[0, i] == pytest.approx(expected_reward[i], abs=1e-5)


def test_wrap_env_vector_reward():
    """Test wrapping an environment that already returns vector rewards"""
    # Create environment that returns vector rewards
    num_objectives = 4
    env = SimpleVectorEnv(num_objectives=num_objectives)

    # Create MOSAC agent
    model = MOSAC(
        policy="MOSACPolicy",
        env=env,
        num_objectives=num_objectives,
        learning_rate=3e-4,
        buffer_size=1000,
        learning_starts=0,
        _init_setup_model=False
    )

    # Call the _wrap_env method
    wrapped_env = model._wrap_env(env, verbose=0)

    # Reset and take a step
    obs = wrapped_env.reset()[0]
    action = np.array([[0.5, -0.2]])
    next_obs, reward, done, info = wrapped_env.step(action)

    # Check that reward has correct shape
    assert reward.shape == (1, num_objectives)

    # Calculate expected reward
    scalar_reward = np.sum(action[0])
    expected_reward = np.ones(num_objectives) * scalar_reward * np.arange(1, num_objectives + 1)

    # Check that vector reward is preserved correctly
    np.testing.assert_allclose(reward[0], expected_reward)


def test_wrap_env_already_vec_env():
    """Test wrapping an environment that is already a VecEnv"""

    # Create a DummyVecEnv
    def make_env():
        return SimpleScalarEnv()

    vec_env = DummyVecEnv([make_env])
    num_objectives = 4

    # Create MOSAC agent
    model = MOSAC(
        policy="MOSACPolicy",
        env=vec_env,
        num_objectives=num_objectives,
        learning_rate=3e-4,
        buffer_size=1000,
        learning_starts=0,
        _init_setup_model=False
    )

    # Call the _wrap_env method
    wrapped_env = model._wrap_env(vec_env, verbose=0)

    # Check that the wrapped environment is still a VecEnv
    assert isinstance(wrapped_env, VecEnv)

    # Check that buf_rews has been updated to have correct shape
    assert wrapped_env.buf_rews.shape == (1, num_objectives)

    # Reset and take a step
    obs = wrapped_env.reset()[0]
    action = np.array([0.5, -0.2])
    next_obs, reward, done, info = wrapped_env.step([action])

    # Check that reward has correct shape
    assert reward.shape == (1, num_objectives)


def test_wrap_env_in_setup_model():
    """Test that _wrap_env is correctly called during _setup_model"""
    # Create environment
    env = SimpleScalarEnv()
    num_objectives = 4

    # Create MOSAC agent and let it set up the model
    model = MOSAC(
        policy="MOSACPolicy",
        env=env,
        num_objectives=num_objectives,
        learning_rate=3e-4,
        buffer_size=1000,
        learning_starts=0,
        _init_setup_model=True  # Automatically set up model
    )

    # Check that the environment was wrapped correctly
    assert isinstance(model.env, VecEnv)
    assert hasattr(model.env, 'num_objectives')
    assert model.env.num_objectives == num_objectives

    # Check that buf_rews has correct shape
    assert model.env.buf_rews.shape == (1, num_objectives)

    # Take a step using the model's step method
    obs = model.env.reset()[0]
    action = model.predict(obs, deterministic=True)[0]
    next_obs, reward, done, info = model.env.step(action)

    # Check reward shape
    assert reward.shape == (1, num_objectives)


def test_wrap_env_with_monitor():
    """Test wrapping with monitor_wrapper=True"""
    # Create environment
    env = SimpleScalarEnv()
    num_objectives = 4

    # Create MOSAC agent
    model = MOSAC(
        policy="MOSACPolicy",
        env=env,
        num_objectives=num_objectives,
        learning_rate=3e-4,
        buffer_size=1000,
        learning_starts=0,
        _init_setup_model=False
    )

    # Call the _wrap_env method with monitor_wrapper=True
    wrapped_env = model._wrap_env(env, verbose=1, monitor_wrapper=True)

    # Check if Monitor wrapper was applied (indirectly)
    # We can check if the unwrapped environment has get_episode_rewards method
    assert hasattr(wrapped_env, 'get_original_obs')

    # Reset and take a step to ensure it works
    obs = wrapped_env.reset()[0]
    action = np.array([[0.5, -0.2]])
    next_obs, reward, done, info = wrapped_env.step(action)

    # Check reward shape
    assert reward.shape == (1, num_objectives)


def test_wrap_env_image_obs():
    """Test wrapping an environment with image observations"""
    # Skip if VecTransposeImage not available
    try:
        from stable_baselines3.common.vec_env import VecTransposeImage
    except ImportError:
        pytest.skip("VecTransposeImage not available")

    # Create environment with image observations
    class ImageEnv(gym.Env):
        def __init__(self):
            self.action_space = spaces.Discrete(2)
            # RGB image, channels last (height, width, channels)
            self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

        def reset(self, seed=None, options=None):
            return np.zeros((64, 64, 3), dtype=np.uint8), {}

        def step(self, action):
            return np.zeros((64, 64, 3), dtype=np.uint8), 1.0, False, False, {}

    env = ImageEnv()
    num_objectives = 4

    # Create MOSAC agent
    model = MOSAC(
        policy="CnnPolicy",  # Use CNN policy for images
        env=env,
        num_objectives=num_objectives,
        learning_rate=3e-4,
        buffer_size=1000,
        learning_starts=0,
        _init_setup_model=False
    )

    # Call the _wrap_env method
    wrapped_env = model._wrap_env(env, verbose=1)

    # Check if VecTransposeImage was applied
    assert is_vecenv_wrapped(wrapped_env, VecTransposeImage)

    # Reset and get observation to verify channels first format
    obs = wrapped_env.reset()[0]

    # Check if observation is in channels-first format (C, H, W)
    assert obs.shape == (1, 3, 64, 64)


if __name__ == "__main__":
    # Run tests
    test_wrap_env_scalar_reward()
    test_wrap_env_custom_reward_weights()
    test_wrap_env_vector_reward()
    test_wrap_env_already_vec_env()
    test_wrap_env_in_setup_model()
    test_wrap_env_with_monitor()
    try:
        test_wrap_env_image_obs()
    except ImportError:
        print("Skipping image test due to missing dependencies")

    print("All tests passed!")