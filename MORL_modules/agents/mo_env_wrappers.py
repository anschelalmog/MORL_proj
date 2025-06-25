import warnings
from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Callable, Optional, List, Dict, Union

import gymnasium as gym
import numpy as np

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.util import dict_to_obs, obs_space_info


class MODummyVecEnv(DummyVecEnv):
    """
    A modified version of DummyVecEnv that supports multi-objective rewards.
    This class overrides only the necessary methods to handle vector rewards.

    :param env_fns: a list of functions that return environments to vectorize
    :param num_objectives: number of objectives for multi-objective RL (default: detected from env)
    """

    def __init__(self, env_fns: list[Callable[[], gym.Env]], num_objectives: Optional[int] = None):
        # Initialize parent class
        super().__init__(env_fns)

        # Determine number of objectives
        self.num_objectives = num_objectives

        # Try to get number of objectives from environment if not provided
        if self.num_objectives is None:
            # Try different attribute names that might contain the number of objectives
            for attr in ["num_objectives", "n_objectives", "_num_objectives", "_n_objectives"]:
                if hasattr(self.envs[0], attr):
                    self.num_objectives = getattr(self.envs[0], attr)
                    break
            # If we can't find it and the first reward is a numpy array, use its length
            if self.num_objectives is None:
                # Step the environment once to see reward shape
                obs = self.envs[0].reset()[0]
                action = self.action_space.sample()
                _, reward, _, _, _ = self.envs[0].step(action)
                # Reset environment to initial state
                self.envs[0].reset()

                if isinstance(reward, np.ndarray):
                    self.num_objectives = reward.shape[0]
                else:
                    # Default to 1 if we can't determine (scalar reward)
                    self.num_objectives = 1

        # Override the reward buffer to support vector rewards
        if self.num_objectives > 1:
            self.buf_rews = np.zeros((self.num_envs, self.num_objectives), dtype=np.float32)
        else:
            # Keep original shape for compatibility if just one objective
            self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)

    def step_wait(self) -> VecEnvStepReturn:
        """
        Override step_wait to handle vector rewards.
        """
        for env_idx in range(self.num_envs):
            obs, reward, terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )

            # Store reward (handle both scalar and vector rewards)
            if self.num_objectives > 1:
                if isinstance(reward, (int, float)):
                    # Convert scalar to vector by putting it in first element
                    vector_reward = np.zeros(self.num_objectives, dtype=np.float32)
                    vector_reward[0] = reward
                    self.buf_rews[env_idx] = vector_reward
                elif isinstance(reward, np.ndarray) and reward.size == self.num_objectives:
                    # Already vector reward with right size
                    self.buf_rews[env_idx] = reward
                else:
                    # Unknown format, try to make it work
                    try:
                        self.buf_rews[env_idx] = np.array(reward, dtype=np.float32)
                    except (ValueError, TypeError):
                        # Fall back to scalar in first position
                        vector_reward = np.zeros(self.num_objectives, dtype=np.float32)
                        vector_reward[0] = float(reward)
                        self.buf_rews[env_idx] = vector_reward
            else:
                # For single objective, keep original behavior
                self.buf_rews[env_idx] = reward

            # Handle done flag and reset
            self.buf_dones[env_idx] = terminated or truncated
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)

        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

#wraper if the environment is not already a multi-objective environment
class MultiObjectiveWrapper(gym.Wrapper):
    def __init__(self, env, num_objectives: Optional[int] = None):
        super().__init__(env)
        # Try to get number of objectives from environment if not provided
        self.num_objectives = num_objectives
        if self.num_objectives is None:
            # Try different attribute names that might contain the number of objectives
            for attr in ["num_objectives", "n_objectives", "_num_objectives", "_n_objectives"]:
                if hasattr(self.env, attr):
                    self.num_objectives = getattr(self.env, attr)
                    break
            # If we can't find it and the first reward is a numpy array, use its length
            if self.num_objectives is None:
                # Step the environment once to see reward shape
                obs = self.env.reset()[0]
                action = self.action_space.sample()
                _, reward, _, _, _ = self.env.step(action)
                # Reset environment to initial state
                self.env.reset()

                if isinstance(reward, np.ndarray):
                    self.num_objectives = reward.shape[0]
                else:
                    # Default to 1 if we can't determine (scalar reward)
                    self.num_objectives = 1


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Convert scalar reward to vector reward if needed
        if isinstance(reward, (int, float)):
            vector_reward = np.zeros(self.num_objectives, dtype=np.float32)
            vector_reward[0] = np.sum(reward)  # Put sum in first objective
        elif isinstance(reward, np.ndarray) and reward.shape == (self.num_objectives,):
            # Already a vector reward of the right shape
            vector_reward = reward
        elif isinstance(reward, np.ndarray):
            # It's an array but not the right shape
            if reward.size == self.num_objectives:
                vector_reward = reward.reshape(self.num_objectives)
            else:
                vector_reward = np.zeros(self.num_objectives, dtype=np.float32)
                vector_reward[0] = np.sum(reward)  # Put sum in first objective
        else:
            # Unknown reward type, use default
            vector_reward = np.zeros(self.num_objectives, dtype=np.float32)
            vector_reward[0] = reward

        return obs, vector_reward, terminated, truncated, info