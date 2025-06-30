import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Any, Optional, Union


class DictToBoxWrapper(gym.Wrapper):
    """
    A wrapper that converts dictionary observation and action spaces to Box spaces.
    This is useful for algorithms that only support continuous Box spaces.

    The wrapper flattens dictionary spaces into a single Box space and keeps track
    of the original structure to convert back and forth.
    """

    def __init__(self, env):
        """Initialize the wrapper with the environment."""
        super().__init__(env)

        # Process observation space
        self.original_observation_space = env.observation_space
        #convert the spaces to spaces.Dict for clarity
        if isinstance(self.original_observation_space,  Dict):
            self.original_observation_space =  spaces.Dict(self.original_observation_space)



        if isinstance(self.original_observation_space, spaces.Dict):
            self.obs_keys = list(self.original_observation_space.spaces.keys())
            self.obs_shapes = {}
            self.obs_dtypes = {}
            self.obs_lows = {}
            self.obs_highs = {}

            # Calculate total size and store metadata for each observation component
            total_obs_size = 0
            for key in self.obs_keys:
                space = self.original_observation_space.spaces[key]
                if isinstance(space, spaces.Box):
                    shape = space.shape
                    size = int(np.prod(shape))
                    self.obs_shapes[key] = shape
                    self.obs_dtypes[key] = space.dtype
                    self.obs_lows[key] = space.low
                    self.obs_highs[key] = space.high
                    total_obs_size += size
                else:
                    raise ValueError(f"Unsupported space type for observation key {key}: {type(space)}")

            # Create a new Box observation space
            low = np.zeros(total_obs_size, dtype=np.float32) - float('inf')
            high = np.zeros(total_obs_size, dtype=np.float32) + float('inf')

            # Set the proper bounds for each part
            idx = 0
            self.obs_indices = {}
            for key in self.obs_keys:
                space = self.original_observation_space.spaces[key]
                size = int(np.prod(space.shape))
                self.obs_indices[key] = (idx, idx + size)

                # Set appropriate bounds if they exist
                if hasattr(space, 'low') and hasattr(space, 'high'):
                    flat_low = space.low.flatten() if hasattr(space.low, 'flatten') else np.array([space.low])
                    flat_high = space.high.flatten() if hasattr(space.high, 'flatten') else np.array([space.high])
                    low[idx:idx + size] = flat_low
                    high[idx:idx + size] = flat_high

                idx += size

            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            # If not a Dict, keep the original observation space
            self.observation_space = self.original_observation_space

        # Process action space
        self.original_action_space = env.action_space
        # convert the spaces to spaces.Dict for clarity
        if isinstance(self.original_action_space, Dict):
            self.original_action_space = spaces.Dict(self.original_action_space)
        if isinstance(self.original_action_space, spaces.Dict):
            self.action_keys = list(self.original_action_space.spaces.keys())
            self.action_shapes = {}
            self.action_dtypes = {}

            # Calculate total size and store metadata for each action component
            total_action_size = 0
            for key in self.action_keys:
                space = self.original_action_space.spaces[key]
                if isinstance(space, spaces.Box):
                    shape = space.shape
                    size = int(np.prod(shape))
                    self.action_shapes[key] = shape
                    self.action_dtypes[key] = space.dtype
                    total_action_size += size
                else:
                    raise ValueError(f"Unsupported space type for action key {key}: {type(space)}")

            # Create a new Box action space
            low = np.zeros(total_action_size, dtype=np.float32)
            high = np.ones(total_action_size, dtype=np.float32)

            # Set the proper bounds for each part
            idx = 0
            self.action_indices = {}
            for key in self.action_keys:
                space = self.original_action_space.spaces[key]
                size = int(np.prod(space.shape))
                self.action_indices[key] = (idx, idx + size)

                # Set appropriate bounds if they exist
                if hasattr(space, 'low') and hasattr(space, 'high'):
                    flat_low = space.low.flatten() if hasattr(space.low, 'flatten') else np.array([space.low])
                    flat_high = space.high.flatten() if hasattr(space.high, 'flatten') else np.array([space.high])
                    low[idx:idx + size] = flat_low
                    high[idx:idx + size] = flat_high

                idx += size

            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            # If not a Dict, keep the original action space
            self.action_space = self.original_action_space

    def reset(self, **kwargs):
        """Reset the environment and convert the observation if needed."""
        obs, info = self.env.reset(**kwargs)
        return self._convert_observation(obs), info

    def step(self, action):
        """Take a step in the environment with the given action and convert the observation if needed."""
        # Convert Box action to Dict action if needed
        if isinstance(self.original_action_space, spaces.Dict):
            action = self._convert_action_to_dict(action)

        # Take a step in the environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Convert Dict observation to Box observation if needed
        if isinstance(self.original_observation_space, spaces.Dict):
            obs = self._convert_observation(obs)

        return obs, reward, terminated, truncated, info

    def _convert_observation(self, obs):
        """Convert a dictionary observation to a flat array."""
        if not isinstance(self.original_observation_space, spaces.Dict):
            return obs

        # Initialize the flat observation array
        flat_obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        # Fill the flat observation with values from the dictionary
        for key in self.obs_keys:
            start_idx, end_idx = self.obs_indices[key]
            flat_component = np.array(obs[key], dtype=np.float32).flatten()
            flat_obs[start_idx:end_idx] = flat_component

        return flat_obs

    def _convert_action_to_dict(self, action):
        """Convert a flat array action to a dictionary action."""
        if not isinstance(self.original_action_space, spaces.Dict):
            return action

        # Initialize the dictionary action
        dict_action = {}

        # Fill the dictionary with values from the flat action
        for key in self.action_keys:
            start_idx, end_idx = self.action_indices[key]
            flat_component = action[start_idx:end_idx]
            original_shape = self.action_shapes[key]
            dict_action[key] = flat_component.reshape(original_shape).astype(self.action_dtypes[key])

        return dict_action


# Example usage:
if __name__ == "__main__":
    # Create a sample environment with Dict spaces
    class DictEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Dict({
                'position': spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                'velocity': spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32),
                'sensors': spaces.Box(low=0, high=100, shape=(10,), dtype=np.float32)
            })

            self.action_space = spaces.Dict({
                'motor': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
                'brake': spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
            })

        def reset(self, **kwargs):
            return {
                'position': np.random.uniform(-1.0, 1.0, size=(3,)),
                'velocity': np.random.uniform(-2.0, 2.0, size=(3,)),
                'sensors': np.random.uniform(0, 100, size=(10,))
            }, {}

        def step(self, action):
            # Simple dummy step function
            obs = {
                'position': np.random.uniform(-1.0, 1.0, size=(3,)),
                'velocity': np.random.uniform(-2.0, 2.0, size=(3,)),
                'sensors': np.random.uniform(0, 100, size=(10,))
            }
            reward = 0.0
            terminated = False
            truncated = False
            info = {}
            return obs, reward, terminated, truncated, info


    # Create and wrap the environment
    env = DictEnv()
    wrapped_env = DictToBoxWrapper(env)

    # Check the spaces
    print("Original observation space:", env.observation_space)
    print("Wrapped observation space:", wrapped_env.observation_space)
    print("Original action space:", env.action_space)
    print("Wrapped action space:", wrapped_env.action_space)

    # Test interaction
    obs, info = wrapped_env.reset()
    print("Observation shape:", obs.shape)

    action = wrapped_env.action_space.sample()
    print("Action shape:", action.shape)

    next_obs, reward, terminated, truncated, info = wrapped_env.step(action)
    print("Next observation shape:", next_obs.shape)