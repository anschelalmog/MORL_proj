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


def convert_gymnasium_space_to_gym(space):
    """
    Convert a Gymnasium space to a Gym space.
    Handles the main space types and their API differences.
    """
    if isinstance(space, gym_new.spaces.Box):
        # Convert Box space
        return gym_old.spaces.Box(
            low=space.low,
            high=space.high,
            shape=space.shape,
            dtype=space.dtype
        )

    elif isinstance(space, gym_new.spaces.Discrete):
        # Convert Discrete space
        return gym_old.spaces.Discrete(space.n)

    elif isinstance(space, gym_new.spaces.MultiDiscrete):
        # Convert MultiDiscrete space
        return gym_old.spaces.MultiDiscrete(space.nvec)

    elif isinstance(space, gym_new.spaces.MultiBinary):
        # Convert MultiBinary space
        return gym_old.spaces.MultiBinary(space.n)

    elif isinstance(space, gym_new.spaces.Dict):
        # Convert Dict space recursively
        converted_spaces = {}
        for key, subspace in space.spaces.items():
            converted_spaces[key] = convert_gymnasium_space_to_gym(subspace)
        return gym_old.spaces.Dict(converted_spaces)

    elif isinstance(space, gym_new.spaces.Tuple):
        # Convert Tuple space recursively
        converted_spaces = []
        for subspace in space.spaces:
            converted_spaces.append(convert_gymnasium_space_to_gym(subspace))
        return gym_old.spaces.Tuple(converted_spaces)

    else:
        # For any other space type, try to create a generic wrapper
        print(f"Warning: Unknown space type {type(space)}, attempting direct conversion")
        try:
            # Try to extract basic attributes and create a Box space as fallback
            if hasattr(space, 'low') and hasattr(space, 'high'):
                return gym_old.spaces.Box(
                    low=space.low,
                    high=space.high,
                    shape=space.shape,
                    dtype=space.dtype
                )
            else:
                raise ValueError(f"Cannot convert space type {type(space)} to gym")
        except Exception as e:
            raise ValueError(f"Failed to convert space {type(space)}: {e}")


class GymnasiumToGymWrapper(gym_old.Env):
    """
    Wrapper to convert a Gymnasium environment to a Gym environment.
    This handles the API differences between gymnasium and gym.
    """

    def __init__(self, gymnasium_env):
        super().__init__()
        self.gymnasium_env = gymnasium_env

        # Copy over the action and observation spaces
        # Convert action and observation spaces to gym format
        self.action_space = convert_gymnasium_space_to_gym(gymnasium_env.action_space)
        self.observation_space = convert_gymnasium_space_to_gym(gymnasium_env.observation_space)

        # Copy metadata
        self.metadata = getattr(gymnasium_env, 'metadata', {})
        self.spec = getattr(gymnasium_env, 'spec', None)
        self.unwrapped = getattr(gymnasium_env, 'unwrapped', gymnasium_env)

    def reset(self, **kwargs):
        """Reset environment - handle different return signatures"""
        result = self.gymnasium_env.reset(**kwargs)
        if isinstance(result, tuple):
            # Gymnasium returns (observation, info)
            obs, info = result
            return obs  # Gym expects just observation
        else:
            # Already in gym format
            return result

    def step(self, action):
        """Step environment - handle different return signatures"""
        result = self.gymnasium_env.step(action)
        if len(result) == 5:
            # Gymnasium returns (observation, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = result
            # Gym expects (observation, reward, done, info)
            done = terminated or truncated
            return obs, reward, done, info
        elif len(result) == 4:
            # Already in gym format (observation, reward, done, info)
            return result
        else:
            raise ValueError(f"Unexpected step return format: {len(result)} elements")

    def render(self, mode='human', **kwargs):
        """Render environment"""
        return self.gymnasium_env.render()

    def close(self):
        """Close environment"""
        if hasattr(self.gymnasium_env, 'close'):
            return self.gymnasium_env.close()

    def seed(self, seed=None):
        """Seed environment"""
        if hasattr(self.gymnasium_env, 'seed'):
            return self.gymnasium_env.seed(seed)
        elif hasattr(self.gymnasium_env, 'reset'):
            # Some environments handle seeding through reset
            return self.gymnasium_env.reset(seed=seed)

    def __getattr__(self, name):
        """Delegate any other attribute access to the wrapped environment"""
        return getattr(self.gymnasium_env, name)


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


def create_energy_net_env_gym(**kwargs):
    """Create EnergyNet environment for Gym (wrapped from Gymnasium)."""

    # Create the gymnasium environment
    gymnasium_env = create_energy_net_env(**kwargs)
    # Wrap it to make it compatible with gym
    gym_env = GymnasiumToGymWrapper(gymnasium_env)

    return gym_env


# Register with Gymnasium
register(
    id="MO-EnergyNet-v0",
    entry_point="MORL_modules.run_algos.create_energy_net_env:create_energy_net_env",
)

# Register with Gym (using the wrapper)
gym_register(
    id="MO-EnergyNet-v0",
    entry_point="MORL_modules.run_algos.create_energy_net_env:create_energy_net_env_gym",
)

