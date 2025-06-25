import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecTransposeImage, is_vecenv_wrapped
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.preprocessing import check_for_nested_spaces, is_image_space, \
    is_image_space_channels_first
from copy import deepcopy

# Import your MOSAC class and MODummyVecEnv
from agents.mosac import MOSAC
# Import your MOSAC class
from agents.mosac import MOSAC
from  agents.mo_env_wrappers import MODummyVecEnv


class MultiObjectiveEnv(gym.Env):
    """Simple environment that already returns vector rewards"""

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
        # Already returning vector rewards
        reward = np.ones(self.num_objectives) * np.sum(action)
        done = self.current_step >= 5
        obs = np.ones(4, dtype=np.float32) * self.current_step
        return obs, reward, done, False, {}


def test_wrap_env_regular_environment():
    """Test wrapping a regular environment with MODummyVecEnv"""
    # Create environment
    num_objectives = 4
    env = MultiObjectiveEnv(num_objectives=num_objectives)

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
    wrapped_env = model._wrap_env(env, verbose=1)
    breakpoint()

    # Check that the wrapped environment is a MODummyVecEnv
    assert isinstance(wrapped_env, MODummyVecEnv)

    # Reset the environment
    obs = wrapped_env.reset()[0]

    # Take a step and verify the reward shape
    action = np.array([[0.5, -0.2]])  # VecEnv expects batch of actions
    next_obs, reward, done, info = wrapped_env.step(action)

    # Check that reward is a vector with the correct shape
    assert reward.shape == (1, num_objectives)

    # Check that reward values are correct (environment returns sum(action) for each objective)
    scalar_value = np.sum(action[0])
    expected_reward = np.ones(num_objectives) * scalar_value
    np.testing.assert_allclose(reward[0], expected_reward)


def test_wrap_env_already_vec_env():
    """Test that an already vectorized environment is left unchanged"""
    # Create a MODummyVecEnv
    num_objectives = 4
    env_fn = lambda: MultiObjectiveEnv(num_objectives=num_objectives)
    vec_env = MODummyVecEnv([env_fn])

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
    wrapped_env = model._wrap_env(vec_env, verbose=1)

    # Check that the environment reference is unchanged
    assert wrapped_env is vec_env

    # Reset and take a step to ensure it works
    obs = wrapped_env.reset()[0]
    action = np.array([[0.5, -0.2]])
    next_obs, reward, done, info = wrapped_env.step(action)

    # Check reward shape
    assert reward.shape == (1, num_objectives)


def test_wrap_env_image_observations():
    """Test that environments with image observations are properly handled"""
    # Skip if VecTransposeImage not available
    try:
        from stable_baselines3.common.vec_env import VecTransposeImage
    except ImportError:
        pytest.skip("VecTransposeImage not available")

    # Create environment with image observations
    class ImageMOEnv(gym.Env):
        def __init__(self, num_objectives=4):
            self.action_space = spaces.Discrete(2)
            # RGB image, channels last (height, width, channels)
            self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
            self.num_objectives = num_objectives

        def reset(self, seed=None, options=None):
            return np.zeros((64, 64, 3), dtype=np.uint8), {}

        def step(self, action):
            # Return vector reward
            reward = np.ones(self.num_objectives) * action
            return np.zeros((64, 64, 3), dtype=np.uint8), reward, False, False, {}

    num_objectives = 4
    env = ImageMOEnv(num_objectives=num_objectives)

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

    # Take a step and verify reward shape
    action = [1]  # For Discrete action space
    next_obs, reward, done, info = wrapped_env.step(action)

    # Check reward shape
    assert reward.shape == (1, num_objectives)

    def test_wrap_env_dict_observation_with_box_action():
        """Test that environments with Dict observation spaces and Box action spaces are properly handled"""

        class DictMOEnv(gym.Env):
            def __init__(self, num_objectives=4):
                # Use Box action space instead of Discrete for compatibility with MOSAC
                self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

                # Dict observation space
                self.observation_space = spaces.Dict({
                    'vector': spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
                    'image': spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8)
                })
                self.num_objectives = num_objectives

            def reset(self, seed=None, options=None):
                obs = {
                    'vector': np.zeros(4, dtype=np.float32),
                    'image': np.zeros((32, 32, 3), dtype=np.uint8)
                }
                return obs, {}

            def step(self, action):
                # Return vector reward
                reward = np.ones(self.num_objectives) * np.sum(action)
                obs = {
                    'vector': np.ones(4, dtype=np.float32),
                    'image': np.ones((32, 32, 3), dtype=np.uint8)
                }
                return obs, reward, False, False, {}

        num_objectives = 4
        env = DictMOEnv(num_objectives=num_objectives)

        # Create MOSAC agent
        model = MOSAC(
            policy="MultiInputPolicy",  # Use MultiInput policy for dict obs
            env=env,
            num_objectives=num_objectives,
            learning_rate=3e-4,
            buffer_size=1000,
            learning_starts=0,
            _init_setup_model=False
        )

        # Call the _wrap_env method (should wrap with VecTransposeImage due to image in dict)
        wrapped_env = model._wrap_env(env, verbose=1)

        # Check if VecTransposeImage was applied
        assert is_vecenv_wrapped(wrapped_env, VecTransposeImage)

        # Reset and get observation
        obs = wrapped_env.reset()[0]

        # Check that dict structure is preserved
        assert isinstance(obs, dict)
        assert 'vector' in obs
        assert 'image' in obs

        # Check that image is in channels-first format
        assert obs['image'].shape == (1, 3, 32, 32)

        # Take a step and verify reward shape
        action = np.array([[0.5, -0.3]])  # Box action space
        next_obs, reward, done, info = wrapped_env.step(action)

        # Check reward shape
        assert reward.shape == (1, num_objectives)

        # Check that dict structure is preserved in next_obs
        assert isinstance(next_obs, dict)
        assert 'vector' in next_obs
        assert 'image' in next_obs


    def test_wrap_env_with_multiple_spaces():
        """Test wrapping environments with different observation spaces"""
        # Test environments with different observation spaces
        test_spaces = [
            # Simple box space
            spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),

            # Dict space with no images
            spaces.Dict({
                'pos': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                'vel': spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)
            }),

            # Dict space with a mix of types
            spaces.Dict({
                'vector': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
                'discrete': spaces.Discrete(5)
            })
        ]

        num_objectives = 3

        for obs_space in test_spaces:
            # Create custom environment with the current observation space
            class CustomEnv(gym.Env):
                def __init__(self):
                    self.observation_space = obs_space
                    self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
                    self.num_objectives = num_objectives

                def reset(self, seed=None, options=None):
                    if isinstance(self.observation_space, spaces.Dict):
                        # Create zero observation for each space in the dict
                        obs = {}
                        for key, space in self.observation_space.spaces.items():
                            if isinstance(space, spaces.Box):
                                obs[key] = np.zeros(space.shape, dtype=space.dtype)
                            elif isinstance(space, spaces.Discrete):
                                obs[key] = 0
                    else:
                        # Simple Box space
                        obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

                    return obs, {}

                def step(self, action):
                    # Generate similar observation as reset but with ones
                    if isinstance(self.observation_space, spaces.Dict):
                        obs = {}
                        for key, space in self.observation_space.spaces.items():
                            if isinstance(space, spaces.Box):
                                obs[key] = np.ones(space.shape, dtype=space.dtype)
                            elif isinstance(space, spaces.Discrete):
                                obs[key] = 1
                    else:
                        obs = np.ones(self.observation_space.shape, dtype=self.observation_space.dtype)

                    # Vector reward
                    reward = np.ones(self.num_objectives) * np.sum(action)
                    return obs, reward, False, False, {}

            env = CustomEnv()

            # Create MOSAC agent
            model = MOSAC(
                policy="MultiInputPolicy" if isinstance(obs_space, spaces.Dict) else "MlpPolicy",
                env=env,
                num_objectives=num_objectives,
                learning_rate=3e-4,
                buffer_size=1000,
                learning_starts=0,
                _init_setup_model=False
            )

            # Call the _wrap_env method
            wrapped_env = model._wrap_env(env, verbose=0)

            # Check that the wrapped environment is a VecEnv
            assert isinstance(wrapped_env, VecEnv)

            # Reset and take a step
            obs = wrapped_env.reset()[0]
            action = np.array([[0.5, -0.3]])
            next_obs, reward, done, info = wrapped_env.step(action)

            # Check reward shape
            assert reward.shape == (1, num_objectives)


    print("All tests passed!")




def test_wrap_env_dict_observation_space():
    """Test that environments with Dict observation spaces are properly handled"""

    class DictMOEnv(gym.Env):
        def __init__(self, num_objectives=4):
            self.action_space = spaces.Discrete(2)
            # Dict observation space
            self.observation_space = spaces.Dict({
                'vector': spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
                'image': spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8)
            })
            self.num_objectives = num_objectives

        def reset(self, seed=None, options=None):
            obs = {
                'vector': np.zeros(4, dtype=np.float32),
                'image': np.zeros((32, 32, 3), dtype=np.uint8)
            }
            return obs, {}

        def step(self, action):
            # Return vector reward
            reward = np.ones(self.num_objectives) * action
            obs = {
                'vector': np.ones(4, dtype=np.float32),
                'image': np.ones((32, 32, 3), dtype=np.uint8)
            }
            return obs, reward, False, False, {}

    num_objectives = 4
    env = DictMOEnv(num_objectives=num_objectives)

    # Create MOSAC agent
    model = MOSAC(
        policy="MultiInputPolicy",  # Use MultiInput policy for dict obs
        env=env,
        num_objectives=num_objectives,
        learning_rate=3e-4,
        buffer_size=1000,
        learning_starts=0,
        _init_setup_model=False
    )

    # Call the _wrap_env method (should wrap with VecTransposeImage due to image in dict)
    wrapped_env = model._wrap_env(env, verbose=1)

    # Check if VecTransposeImage was applied
    assert is_vecenv_wrapped(wrapped_env, VecTransposeImage)

    # Reset and get observation
    obs = wrapped_env.reset()[0]

    # Check that dict structure is preserved
    assert isinstance(obs, dict)
    assert 'vector' in obs
    assert 'image' in obs

    # Check that image is in channels-first format
    assert obs['image'].shape == (1, 3, 32, 32)

    # Take a step and verify reward shape
    action = [1]  # For Discrete action space
    next_obs, reward, done, info = wrapped_env.step(action)

    # Check reward shape
    assert reward.shape == (1, num_objectives)


def test_wrap_env_in_setup_model():
    """Test that _wrap_env is correctly called during _setup_model"""
    # Create environment
    num_objectives = 4
    env = MultiObjectiveEnv(num_objectives=num_objectives)

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

    # Reset and take a step
    obs = model.env.reset()[0]
    action = model.predict(obs, deterministic=True)[0]
    next_obs, reward, done, info = model.env.step(action)

    # Check reward shape
    assert reward.shape == (1, num_objectives)


if __name__ == "__main__":
    # Run tests
    test_wrap_env_regular_environment()
    test_wrap_env_already_vec_env()
    test_wrap_env_in_setup_model()
    test_wrap_env_dict_observation_with_box_action()
    test_wrap_env_with_multiple_spaces()

