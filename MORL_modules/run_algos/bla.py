import numpy as np
#monkey patch for older version use in dependecies
if not hasattr(np, 'bool'):
    np.bool = np.bool_ # or np.bool = np.bool_


import pytest

import torch as th
import gymnasium as gym
from gymnasium import spaces
import tempfile
import os
import sys
from typing import Dict, Any, List, Tuple
import pdb

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'MORL_modules'))
print(project_root)
print(current_dir)
from agents.mosac import MOSAC, MOSACPolicy, MOContinuousCritic
from agents.mobuffers import MOReplayBuffer as MOReplayBuffer
from agents.monets import SharedFeatureQNet, SeparateQNet
from agents.mo_env_wrappers import MODummyVecEnv, MultiObjectiveWrapper
from wrappers.mo_pcs_wrapper import MOPCSWrapper
from wrappers.scalarized_mo_pcs_wrapper import ScalarizedMOPCSWrapper
from wrappers.dict_to_box_wrapper import DictToBoxWrapper
from energy_net.envs.energy_net_v0 import EnergyNetV0

from energy_net.market.pricing.pricing_policy import PricingPolicy
from energy_net.market.pricing.cost_types import CostType
from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern


#from gym.envs.registration import register
from gymnasium.envs.registration import register

from gym.envs.registration import register as gym_register

import gymnasium as gym_new
import gym as gym_old
class GymnasiumToGymWrapper(gym_old.Env):
    """
    Enhanced wrapper with proper action/observation shape handling
    """

    def __init__(self, gymnasium_env):
        super().__init__()
        self.gymnasium_env = gymnasium_env

        # Convert spaces
        self.action_space = convert_gymnasium_space_to_gym(gymnasium_env.action_space)
        self.observation_space = convert_gymnasium_space_to_gym(gymnasium_env.observation_space)

        # Store original spaces
        self._original_action_space = gymnasium_env.action_space
        self._original_observation_space = gymnasium_env.observation_space

        # Copy metadata
        self.metadata = getattr(gymnasium_env, 'metadata', {})
        self.spec = getattr(gymnasium_env, 'spec', None)
        #self.unwrapped = getattr(gymnasium_env, 'unwrapped', gymnasium_env)

    def _fix_action_shape(self, action):
        """
        Fix action shape to match what the environment expects.
        Removes extra dimensions if present.
        """
        if isinstance(action, (list, tuple)):
            action = np.array(action)

        if isinstance(action, np.ndarray):
            # If action has extra dimensions, squeeze them out
            if action.ndim > 1 and action.shape[0] == 1:
                action = action.squeeze(0)  # Remove first dimension if it's 1
            elif action.ndim > 1:
                # If multiple samples, take the first one
                action = action[0]

        return action

    def _fix_observation_shape(self, observation):
        """
        Fix observation shape if needed.
        """
        if isinstance(observation, (list, tuple)):
            observation = np.array(observation)

        # Usually observations don't need dimension fixing for single environments
        # but this can be customized if needed
        return observation

    def reset(self, **kwargs):
        """Reset environment - handle different return signatures"""
        result = self.gymnasium_env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
            obs = self._fix_observation_shape(obs)
            return obs
        else:
            return self._fix_observation_shape(result)

    def step(self, action):
        """Step environment - handle different return signatures and fix action shape"""
        # Fix action shape before passing to environment
        action = self._fix_action_shape(action)

        result = self.gymnasium_env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            obs = self._fix_observation_shape(obs)
            return obs, reward, done, info
        elif len(result) == 4:
            obs, reward, done, info = result
            obs = self._fix_observation_shape(obs)
            return obs, reward, done, info
        else:
            raise ValueError(f"Unexpected step return format: {len(result)} elements")

    def render(self, mode='human', **kwargs):
        """Render environment"""
        if hasattr(self.gymnasium_env, 'render'):
            return self.gymnasium_env.render()
        return None

    def close(self):
        """Close environment"""
        if hasattr(self.gymnasium_env, 'close'):
            return self.gymnasium_env.close()

    def seed(self, seed=None):
        """Seed environment"""
        if hasattr(self.gymnasium_env, 'seed'):
            return self.gymnasium_env.seed(seed)
        elif hasattr(self.gymnasium_env, 'reset'):
            return self.gymnasium_env.reset(seed=seed)

    def action_space_sample(self):
        """Sample action with proper shape"""
        action = self.action_space.sample()
        return self._fix_action_shape(action)

    def __getattr__(self, name):
        """Delegate any other attribute access to the wrapped environment"""
        return getattr(self.gymnasium_env, name)


# Enhanced space conversion with shape validation
def convert_gymnasium_space_to_gym(space):
    """
    Convert a Gymnasium space to a Gym space with proper shape handling.
    """
    if isinstance(space, gym_new.spaces.Box):
        # Ensure proper shape without extra dimensions
        low = np.array(space.low)
        high = np.array(space.high)

        # Remove unnecessary dimensions
        if low.ndim > 1 and low.shape[0] == 1:
            low = low.squeeze(0)
            high = high.squeeze(0)

        return gym_old.spaces.Box(
            low=low,
            high=high,
            shape=low.shape,  # Use the corrected shape
            dtype=space.dtype
        )

    elif isinstance(space, gym_new.spaces.Discrete):
        return gym_old.spaces.Discrete(space.n)

    elif isinstance(space, gym_new.spaces.MultiDiscrete):
        return gym_old.spaces.MultiDiscrete(space.nvec)

    elif isinstance(space, gym_new.spaces.MultiBinary):
        return gym_old.spaces.MultiBinary(space.n)

    elif isinstance(space, gym_new.spaces.Dict):
        converted_spaces = {}
        for key, subspace in space.spaces.items():
            converted_spaces[key] = convert_gymnasium_space_to_gym(subspace)
        return gym_old.spaces.Dict(converted_spaces)

    elif isinstance(space, gym_new.spaces.Tuple):
        converted_spaces = []
        for subspace in space.spaces:
            converted_spaces.append(convert_gymnasium_space_to_gym(subspace))
        return gym_old.spaces.Tuple(converted_spaces)

    else:
        print(f"Warning: Unknown space type {type(space)}")
        try:
            if hasattr(space, 'low') and hasattr(space, 'high'):
                low = np.array(space.low)
                high = np.array(space.high)

                # Remove unnecessary dimensions
                if low.ndim > 1 and low.shape[0] == 1:
                    low = low.squeeze(0)
                    high = high.squeeze(0)

                return gym_old.spaces.Box(
                    low=low,
                    high=high,
                    shape=low.shape,
                    dtype=space.dtype
                )
            else:
                raise ValueError(f"Cannot convert space type {type(space)} to gym")
        except Exception as e:
            raise ValueError(f"Failed to convert space {type(space)}: {e}")


def create_energy_net_env(**kwargs):
    """Create EnergyNet environment."""
    from energy_net.envs.energy_net_v0 import EnergyNetV0

    default_kwargs = {

        'pricing_policy': PricingPolicy.QUADRATIC,
        'demand_pattern': DemandPattern.SINUSOIDAL,
        'cost_type': CostType.CONSTANT,
    }

    default_kwargs.update(kwargs)
    base_env = EnergyNetV0(**default_kwargs)
    return MOPCSWrapper(DictToBoxWrapper( base_env ), num_objectives=4)

# Debug function to check action shapes
def debug_action_shapes(env):
    """
    Debug function to check action and observation shapes
    """
    print("=== Action/Observation Shape Debug ===")
    print(f"Action space: {env.action_space}")
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Observation space: {env.observation_space}")
    print(f"Observation space shape: {env.observation_space.shape}")

    # Sample action
    action = env.action_space.sample()
    print(f"Sampled action: {action}")
    print(f"Sampled action shape: {action.shape}")
    print(f"Sampled action type: {type(action)}")

    # Reset environment
    obs = env.reset()
    print(f"Reset observation shape: {obs.shape if hasattr(obs, 'shape') else 'No shape'}")

    # Step with action
    try:
        obs, reward, done, info = env.step(action)
        print(f"Step successful!")
        print(f"Step observation shape: {obs.shape if hasattr(obs, 'shape') else 'No shape'}")
    except Exception as e:
        print(f"Step failed: {e}")

    print("=== End Debug ===")


# Usage example
if __name__ == "__main__":
    # Test with your environment
    gymnasium_env = create_energy_net_env()  # Your function
    gym_env = GymnasiumToGymWrapper(gymnasium_env)

    # Debug the shapes
    debug_action_shapes(gym_env)

    # Test normal usage
    obs = gym_env.reset()
    action = gym_env.action_space.sample()
    print(f"Action before step: {action}")
    print(f"Action shape: {action.shape}")

    obs, reward, done, info = gym_env.step(action)
    print("Step completed successfully!")