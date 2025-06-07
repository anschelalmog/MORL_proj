import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any, List

class MOEnergyNetWrapper(gym.Wrapper):
    """
    Wrapper for the EnergyNet environment to expose multiple reward objectives.

    Objectives:
    1. Economic objective: Profit from energy arbitrage
    2. Battery health: Battery level and cycling
    3. Grid support: Contribution to grid balance
    4. Energy autonomy: Self-consumption of produced energy
    """

    def __init__(self, env, num_objectives=4, reward_weights=None):
        """
        Initialize multi-objective wrapper.

        Args:
            env: EnergyNet environment (already wrapped for PCS)
            num_objectives: Number of reward objectives
            reward_weights: Weights for each objective (for logging purposes)
        """
        super().__init__(env)
        self.num_objectives = num_objectives
        self.reward_weights = reward_weights or np.ones(num_objectives) / num_objectives

        # Track metrics for each episode
        self.episode_metrics = {
            "economic_rewards": [],
            "battery_health_rewards": [],
            "grid_support_rewards": [],
            "autonomy_rewards": [],
            "step_counts": [],
        }

        # Current episode metrics
        self.current_economic_reward = 0
        self.current_battery_health_reward = 0
        self.current_grid_support_reward = 0
        self.current_autonomy_reward = 0
        self.steps = 0
        self.pcsunit = None
        self.battery =None
        self.battery_manager = None
        if hasattr(self.env.unwrapped, "controller") and hasattr(self.env.unwrapped.controller, "pcsunit"):
            self.pcsunit = self.env.unwrapped.controller.pcsunit
            self.battery = self.env.unwrapped.controller.pcsunit.battery
        # Store previous battery level for computing changes
        if hasattr(self.env.unwrapped, "controller"):
            self.battery_manager = self.env.unwrapped.controller.battery_manager
        self.prev_battery_level = None


    def reset(self, **kwargs):
        """Reset environment and metrics."""
        observation, info = self.env.reset(**kwargs)

        #reset pscunit, battery and battery manager from the environment
        if hasattr(self.env.unwrapped, "controller") and hasattr(self.env.unwrapped.controller, "pcsunit"):
            self.pcsunit = self.env.unwrapped.controller.pcsunit
            self.battery = self.env.unwrapped.controller.pcsunit.battery

        if hasattr(self.env.unwrapped, "controller"):
            self.battery_manager = self.env.unwrapped.controller.battery_manager
        # Reset episode metrics
        self.current_economic_reward = 0
        self.current_battery_health_reward = 0
        self.current_grid_support_reward = 0
        self.current_autonomy_reward = 0
        self.steps = 0
        # Store previous battery level for computing changes
        # Get initial battery level
        if "battery_level" in info:
            self.prev_battery_level = info["battery_level"]
        elif self.battery is not None:
            # Try to get battery level from battery
            self.prev_battery_level = self.battery.get_state()
        else:
            self.prev_battery_level = 50.0  # Default assumption

        return observation, info

    def step(self, action):
        """
        Step the environment and compute multi-objective rewards.

        Returns vector rewards instead of scalar rewards.
        """
        # Call parent step
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Initialize vector reward
        mo_rewards = np.zeros(self.num_objectives)

        # 1. Economic objective - normalized profit/cost
        economic_reward = reward  # Assume base reward is economic
        mo_rewards[0] = economic_reward
        self.current_economic_reward += economic_reward

        # Extract metrics from info or environment state
        battery_level = None
        net_exchange = None
        production = None
        consumption = None

        # Get battery level
        if "battery_level" in info:
            battery_level = info["battery_level"]
        elif self.battery is not None:
            battery_level = self.battery.get_state()

        # Get energy exchange info
        if "net_exchange" in info:
            net_exchange = info["net_exchange"]
        # Get production and consumption
        if self.pcsunit is not None:
            if hasattr(self.pcsunit, "get_self_production") and hasattr(self.pcsunit, "get_self_consumption"):
                production = pcsunit.get_self_production()
                consumption = pcsunit.get_self_consumption()

        # 2. Battery health objective
        battery_change = None
        if battery_level is not None and self.prev_battery_level is not None:
            # Penalize large state of charge changes (avoid cycling)
            battery_change = abs(battery_level - self.prev_battery_level)
        elif  self.pcsunit is not None  and hasattr(self.pcsunit, "get_energy_change") and battery_level is not None:
            battery_change = self.pcsunit.get_energy_change()



        if battery_change  is not None and self.battery_manager is not None:


            # Reward being in middle range (30-70%) to avoid extremes
            middle_range_factor = (
                    1.0 - 2.0 * abs(battery_level -(self.battery_manager.battery_max - self.battery_manager.battery_min)/2 ) / (self.battery_manager.battery_max - self.battery_manager.battery_min))

            battery_health_reward = 0.5 * middle_range_factor - 0.5 * (battery_change / 20.0)
            mo_rewards[1] = battery_health_reward
            self.current_battery_health_reward += battery_health_reward

            # Update previous battery level
            self.prev_battery_level = battery_level

        # 3. Grid support objective
        if net_exchange is not None:
            # For grid support, we need grid demand or price signals
            # As a proxy, we'll use the price difference from average as indicator of grid need
            price_diff = 0
            if "iso_buy_price" in info and "iso_sell_price" in info:
                avg_price = (info["iso_buy_price"] + info["iso_sell_price"]) / 2
                price_diff = (info["iso_buy_price"] - avg_price) / avg_price

            # If prices are high, grid_support comes from selling (negative net_exchange)
            # If prices are low, grid_support comes from buying (positive net_exchange)
            grid_support = -net_exchange * price_diff if price_diff != 0 else 0

            # Normalize and clip
            grid_support_reward = np.clip(grid_support / 10.0, -1.0, 1.0)
            mo_rewards[2] = grid_support_reward
            self.current_grid_support_reward += grid_support_reward

        # 4. Energy autonomy objective
        if production is not None and consumption is not None:
            # Autonomy measures how much of consumption is covered by own production
            if consumption > 0:
                autonomy = min(production, consumption) / consumption
            else:
                autonomy = 1.0 if production > 0 else 0.0

            autonomy_reward = autonomy
            mo_rewards[3] = autonomy_reward
            self.current_autonomy_reward += autonomy_reward

        # Track step count
        self.steps += 1

        # If episode ended, save metrics
        if terminated or truncated:
            self.episode_metrics["economic_rewards"].append(self.current_economic_reward)
            self.episode_metrics["battery_health_rewards"].append(self.current_battery_health_reward)
            self.episode_metrics["grid_support_rewards"].append(self.current_grid_support_reward)
            self.episode_metrics["autonomy_rewards"].append(self.current_autonomy_reward)
            self.episode_metrics["step_counts"].append(self.steps)

            # Add episode metrics to info
            info["episode_metrics"] = {
                "economic_reward": self.current_economic_reward,
                "battery_health_reward": self.current_battery_health_reward,
                "grid_support_reward": self.current_grid_support_reward,
                "autonomy_reward": self.current_autonomy_reward,
                "steps": self.steps
            }

        # Add multi-objective rewards to info
        info["mo_rewards"] = mo_rewards

        return observation, mo_rewards, terminated, truncated, info