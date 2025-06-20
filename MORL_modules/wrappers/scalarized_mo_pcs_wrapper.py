# scalarized_mo_pcs_wrapper.py
import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Any, Optional
import logging

from .mo_pcs_wrapper import MOPCSWrapper


class ScalarizedMOPCSWrapper(gym.Wrapper):
    """
    Scalarizes multi-objective rewards from MOPCSWrapper into single scalar reward.

    This wrapper takes the vector rewards from MOPCSWrapper and combines them
    into a single scalar using weighted sum scalarization.
    """

    def __init__(
            self,
            env,
            weights: Optional[np.ndarray] = None,
            normalize_weights: bool = True,
            **mo_wrapper_kwargs
    ):
        """
        Initialize scalarized wrapper.

        Args:
            env: Base environment (will be wrapped with MOPCSWrapper)
            weights: Weights for each objective. If None, uses equal weights.
            normalize_weights: Whether to normalize weights to sum to 1
            **mo_wrapper_kwargs: Additional arguments passed to MOPCSWrapper
        """
        # Wrap with MOPCSWrapper if not already wrapped
        if not isinstance(env, MOPCSWrapper):
            env = MOPCSWrapper(env, **mo_wrapper_kwargs)

        super().__init__(env)

        # Set up weights
        self.num_objectives = env.num_objectives
        if weights is None:
            self.weights = np.ones(self.num_objectives) / self.num_objectives
        else:
            self.weights = np.array(weights, dtype=np.float32)

        # Validate weights
        if len(self.weights) != self.num_objectives:
            raise ValueError(
                f"Weight dimension ({len(self.weights)}) must match "
                f"number of objectives ({self.num_objectives})"
            )

        # Normalize weights if requested
        if normalize_weights:
            weight_sum = np.sum(self.weights)
            if weight_sum > 0:
                self.weights = self.weights / weight_sum
            else:
                raise ValueError("Weights sum to zero")

        self.logger = logging.getLogger(f"ScalarizedMO_{id(self)}")
        self.logger.info(f"Initialized with weights: {self.weights}")

    def reset(self, **kwargs):
        """Reset environment."""
        return self.env.reset(**kwargs)

    def step(self, action):
        """Step environment and scalarize rewards."""
        observation, mo_rewards, terminated, truncated, info = self.env.step(action)

        # Scalarize the multi-objective rewards
        scalar_reward = np.dot(self.weights, mo_rewards)

        # Keep original MO rewards in info
        info['mo_rewards_original'] = mo_rewards
        info['scalarization_weights'] = self.weights
        info['scalar_reward'] = scalar_reward

        return observation, scalar_reward, terminated, truncated, info

    def set_weights(self, weights: np.ndarray, normalize: bool = True):
        """
        Update scalarization weights.

        Args:
            weights: New weights for objectives
            normalize: Whether to normalize weights to sum to 1
        """
        weights = np.array(weights, dtype=np.float32)

        if len(weights) != self.num_objectives:
            raise ValueError(
                f"Weight dimension ({len(weights)}) must match "
                f"number of objectives ({self.num_objectives})"
            )

        if normalize:
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                raise ValueError("Weights sum to zero")

        self.weights = weights
        self.logger.info(f"Updated weights to: {self.weights}")