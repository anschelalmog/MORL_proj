"""
Energy Net V0 Environment

A unified multi-agent environment that integrates both the ISO and PCS agents
into a single simulation. This environment follows the multi-agent extension
of the Gym interface, where step() takes multiple actions and returns multiple
observations, rewards, and done flags.

Key features:
1. Integrated controller for both ISO and PCS agents
2. Sequential processing of agent actions
3. Single timeline and shared state management
4. Direct access to comprehensive metrics

This environment serves as the main interface between RL algorithms and the
underlying energy net simulation, enabling the training of agents that can
efficiently manage electricity markets and battery storage.
"""

import gymnasium as gym
import importlib
from typing import Dict, Any, Union, Optional
from gymnasium import spaces

from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern
from energy_net.envs.env_utils import bounds_to_gym
from energy_net.market.pricing.cost_types import CostType
from energy_net.market.pricing.pricing_policy import PricingPolicy


class EnergyNetV0(gym.Env):
    """
    Multi-agent environment for simulating energy grid dynamics.
    
    This environment integrates both ISO and PCS agents into a single simulation,
    following a multi-agent extension of the Gym interface where step() takes multiple
    actions and returns observations, rewards, and done flags for all agents.
    
    The environment uses a unified controller to manage the sequential
    simulation, where:
    1. ISO agent sets energy prices
    2. PCS agent responds with battery control actions
    3. Energy exchanges occur
    4. State updates and rewards are calculated
    
    This approach eliminates the need for manual transfers between separate
    environments and provides a more realistic simulation with direct access
    to comprehensive metrics.
    """
    
    def __init__(
        self,
        controller_name: str = "EnergyNetController",
        controller_module: str = "energy_net.controllers",
        single_agent: bool = True,  # Default to single-agent mode for backward compatibility
        **controller_kwargs
    ):
        """
        Initialize the unified Energy Net environment.
        
        Args:
            controller_name: Name of the controller class to use (default: "EnergyNetController")
            controller_module: Python module path where the controller is defined (default: "energy_net.controllers")
            single_agent: Whether to use single-agent mode (default: True)
            **controller_kwargs: Additional keyword arguments to pass to the controller
        """
        super().__init__()
        
        self.single_agent = single_agent
        
        # Dynamically import and instantiate the controller
        try:
            module = importlib.import_module(controller_module)
            controller_class = getattr(module, controller_name)
            self.controller = controller_class(**controller_kwargs)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to initialize controller {controller_name} from module {controller_module}: {str(e)}")
        
        # Define agent spaces
        self.agents = ["iso", "pcs"]
        
        iso_obs_space = bounds_to_gym(self.controller.get_iso_observation_space())
        pcs_obs_space = bounds_to_gym(self.controller.get_pcs_observation_space())
        iso_action_space = bounds_to_gym(self.controller.get_iso_action_space())
        pcs_action_space = bounds_to_gym(self.controller.get_pcs_action_space())
        
        if self.single_agent:
            # In single-agent mode, combine spaces using Dict
            self.observation_space = spaces.Dict({
                "iso": iso_obs_space,
                "pcs": pcs_obs_space
            })
            self.action_space = spaces.Dict({
                "iso": iso_action_space,
                "pcs": pcs_action_space
            })
        else:
            # In multi-agent mode, use regular dicts
            self.observation_space = {
                "iso": iso_obs_space,
                "pcs": pcs_obs_space
            }
            self.action_space = {
                "iso": iso_action_space,
                "pcs": pcs_action_space
            }

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple containing:
            - Initial observations for both agents in a dict
            - Info dictionary with initial state information
        """
        observations, info = self.controller.reset(seed=seed, options=options)
        
        # Format observations as a dict for multi-agent compatibility
        obs_dict = {
            "iso": observations[0],
            "pcs": observations[1]
        }
        
        return obs_dict, info

    def step(self, action_dict):
        """
        Execute one time step of the environment.
        
        Args:
            action_dict: Dict containing actions for each agent
                {"iso": iso_action, "pcs": pcs_action}
            
        Returns:
            Tuple containing:
            - Dict of observations for each agent
            - Dict of rewards for each agent
            - Dict of terminated flags for each agent
            - Dict of truncated flags for each agent
            - Dict of info for each agent
        """
        # Extract actions from dict
        iso_action = action_dict["iso"]
        pcs_action = action_dict["pcs"]
        
        # Execute step on the controller
        # New return format: observations, rewards, terminated, truncated, info
        observations, rewards, terminated, truncated, info = self.controller.step(iso_action, pcs_action)
        
        # Format returns as dicts for multi-agent compatibility
        obs_dict = {
            "iso": observations[0],
            "pcs": observations[1]
        }
        
        reward_dict = {
            "iso": rewards[0],
            "pcs": rewards[1]
        }
        
        terminated_dict = {
            "iso": terminated[0] if isinstance(terminated, (list, tuple)) else terminated,
            "pcs": terminated[1] if isinstance(terminated, (list, tuple)) else terminated
        }
        
        # Handle truncated flag
        truncated_dict = {
            "iso": truncated[0] if isinstance(truncated, (list, tuple)) else truncated,
            "pcs": truncated[1] if isinstance(truncated, (list, tuple)) else truncated
        }
        
        return obs_dict, reward_dict, terminated_dict, truncated_dict, info

    def get_metrics(self):
        """
        Get comprehensive metrics for both agents.
        
        Returns:
            Dict containing metrics for both agents and shared metrics
        """
        return self.controller.get_metrics()
    
    def render(self):
        """
        Render the environment (not implemented).
        
        Raises:
            NotImplementedError: Always, as rendering is not implemented.
        """
        raise NotImplementedError("Rendering is not yet implemented for the EnergyNetV0 environment")

    def close(self):
        """
        Clean up any resources used by the environment.
        """
        # Currently, no cleanup is needed
        pass


def make_env(config=None):
    """
    Factory function to create an instance of EnergyNetV0.
    
    Args:
        config: Configuration dictionary for the environment
        
    Returns:
        EnergyNetV0: An instance of the environment
    """
    return EnergyNetV0(**(config or {}))
