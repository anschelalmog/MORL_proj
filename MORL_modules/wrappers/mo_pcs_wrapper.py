#mo_pcs_wrapper.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any, List, Optional
import logging
import matplotlib.pyplot as plt
from collections import defaultdict


class MOPCSWrapper(gym.Wrapper):
    """
    Multi-Objective wrapper for EnergyNet environment (PCS-focused).

    Provides four objectives:
    1. Economic efficiency: Profit from energy arbitrage
    2. Battery health: Minimize degradation and extreme cycling
    3. Grid support: Contribute to grid stability
    4. Energy autonomy: Maximize self-consumption ratio
    """

    def __init__(self, env, num_objectives=4, reward_weights=None,
                 normalize_rewards=True, log_level=logging.INFO):
        """
        Initialize multi-objective wrapper.

        Args:
            env: EnergyNet environment (already wrapped for PCS)
            num_objectives: Number of reward objectives
            reward_weights: Weights for each objective (for reference)
            normalize_rewards: Whether to normalize rewards to [-1, 1] range
            log_level: Logging level
        """
        super().__init__(env)
        self.num_objectives = num_objectives
        self.reward_weights = reward_weights if reward_weights is not None else np.ones(num_objectives) / num_objectives
        self.normalize_rewards = normalize_rewards

        # Setup logging
        self.logger = logging.getLogger(f"MOWrapper_{id(self)}")
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Environment component references
        self.pcsunit = None
        self.battery = None
        self.battery_manager = None
        self.controller = None

        # Normalization parameters (learned from episodes)
        self.reward_stats = {
            'economic': {'min': -50.0, 'max': 50.0, 'mean': 0.0, 'std': 10.0},
            'battery_health': {'min': -2.0, 'max': 1.0, 'mean': 0.0, 'std': 0.5},
            'grid_support': {'min': -1.0, 'max': 1.0, 'mean': 0.0, 'std': 0.3},
            'autonomy': {'min': 0.0, 'max': 1.0, 'mean': 0.5, 'std': 0.3}
        }

        # Episode tracking
        self.episode_count = 0
        self.episode_rewards = defaultdict(list)
        self.current_episode_rewards = np.zeros(num_objectives)
        self.step_count = 0

        # Battery tracking
        self.prev_battery_level = None
        self.battery_cycle_count = 0
        self.battery_degradation_factor = 0.0

        # Grid tracking
        self.grid_demand_history = []
        self.price_history = []

        self._validate_environment_structure()

        self.logger.info(f"Initialized MOEnergyNetWrapper with {num_objectives} objectives")

    def _validate_environment_structure(self):
        try:
            assert hasattr(self.env, 'unwrapped'), "Environment missing unwrapped attribute"
            assert hasattr(self.env.unwrapped, 'controller'), "Environment missing controller"

            controller = self.env.unwrapped.controller
            assert hasattr(controller, 'battery_manager'), "Controller missing battery_manager"
            assert hasattr(controller, 'pcs_unit'), "Controller missing pcs_unit"
            assert hasattr(controller.pcs_unit, 'battery'), "PCS Unit missing battery"
            assert hasattr(controller.pcs_unit, 'get_self_production'), "PCS Unit missing get_self_production"
            assert hasattr(controller.pcs_unit, 'get_self_consumption'), "PCS Unit missing get_self_consumption"
            assert hasattr(controller.battery_manager, 'get_level'), "Battery Manager missing get_level"

            self.logger.info("Environment structure validation passed")

        except (AttributeError, AssertionError) as e:
            raise RuntimeError(f"Environment structure validation failed: {e}")

    def _get_environment_components(self):
        """Extract and cache components from environment (they're always available)."""
        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'controller'):
            self.controller = self.env.unwrapped.controller
            self.battery_manager = self.controller.battery_manager
            self.pcsunit = self.controller.pcs_unit
            self.battery = self.pcsunit.battery

            # Validation assertions (non-fatal warnings)
            assert self.controller is not None, "Controller should always be available"
            assert self.battery_manager is not None, "Battery Manager should always be available"
            assert self.pcsunit is not None, "PCS Unit should always be available"
            assert self.battery is not None, "Battery should always be available"

            self.logger.info("All components successfully cached")
        else:
            raise RuntimeError("Environment structure is unexpected - missing controller")

    def _get_battery_level(self) -> Optional[float]:
        """Get current battery level - battery_manager.get_level() is always available."""
        if self.controller is None or self.battery_manager is None:
            return None

        try:
            return self.battery_manager.get_level()
        except Exception as e:
            self.logger.warning(f"Unexpected error getting battery level: {e}")
            return None

    def _get_energy_exchange(self, info: Dict) -> Optional[float]:
        """Get net energy exchange with grid."""
        if 'net_exchange' in info:
            return info['net_exchange']

        # Try to compute from available data
        if 'energy_bought' in info and 'energy_sold' in info:
            return info['energy_bought'] - info['energy_sold']

        return None

    def _get_production_consumption(self) -> Tuple[Optional[float], Optional[float]]:
        """Get production and consumption - pcsunit methods are always available."""
        if self.pcsunit is None:
            return None, None

        try:
            production = self.pcsunit.get_self_production()  # Always works
            consumption = self.pcsunit.get_self_consumption()  # Always works
            return production, consumption

        except Exception as e:
            self.logger.warning(f"Unexpected error getting production/consumption: {e}")
            return None, None

    def _compute_economic_reward(self, base_reward, info: Dict) -> float:
        """Compute economic objective reward."""
        # Handle different reward formats from the environment
        if isinstance(base_reward, dict):
            # Environment returns dict with iso/pcs rewards
            return base_reward.get('pcs', 0.0)  # Use PCS reward for economic objective
        elif isinstance(base_reward, (tuple, list)) and len(base_reward) > 1:
            # Environment returns tuple (iso_reward, pcs_reward)
            return base_reward[1]  # Use PCS reward
        elif isinstance(base_reward, (tuple, list)) and len(base_reward) == 1:
            # Single element tuple/list
            return base_reward[0]
        else:
            # Single scalar reward
            return float(base_reward)

    def _compute_battery_health_reward(self, info: Dict) -> float:
        """Compute battery health objective reward."""
        battery_level = self._get_battery_level()

        if battery_level is None or self.prev_battery_level is None:
            return 0.0

        # Penalty for large state changes (cycling degradation)
        state_change = abs(battery_level - self.prev_battery_level)
        cycling_penalty = -0.1 * (state_change / 10.0)  # Normalize by reasonable change

        # Penalty for extreme states (calendar aging)
        if self.battery_manager is not None:
            battery_range = self.battery_manager.battery_max - self.battery_manager.battery_min
            middle_point = (self.battery_manager.battery_max + self.battery_manager.battery_min) / 2

            # Optimal range is 30-70% of capacity
            optimal_low = self.battery_manager.battery_min + 0.3 * battery_range
            optimal_high = self.battery_manager.battery_min + 0.7 * battery_range

            if optimal_low <= battery_level <= optimal_high:
                range_reward = 0.1  # Small reward for staying in optimal range
            else:
                # Penalty increases with distance from optimal range
                if battery_level < optimal_low:
                    distance = optimal_low - battery_level
                else:
                    distance = battery_level - optimal_high
                range_reward = -0.2 * (distance / battery_range)
        else:
            # Fallback: assume 0-100 range
            if 30 <= battery_level <= 70:
                range_reward = 0.1
            else:
                range_reward = -0.1 * abs(battery_level - 50) / 50

        total_reward = cycling_penalty + range_reward
        return total_reward

    def _compute_grid_support_reward(self, info: Dict) -> float:
        """Compute grid support objective reward."""
        net_exchange = self._get_energy_exchange(info)

        if net_exchange is None:
            return 0.0

        # Grid support reward based on price signals
        # High prices = grid needs energy (reward for selling)
        # Low prices = grid has excess (reward for buying)
        grid_support_reward = 0.0

        if 'iso_buy_price' in info and 'iso_sell_price' in info:
            buy_price = info['iso_buy_price']
            sell_price = info['iso_sell_price']
            avg_price = (buy_price + sell_price) / 2

            # Store price history for learning
            self.price_history.append(avg_price)
            if len(self.price_history) > 100:  # Keep last 100 prices
                self.price_history.pop(0)

            if len(self.price_history) > 10:
                price_baseline = np.mean(self.price_history[-10:])
                price_deviation = (avg_price - price_baseline) / price_baseline

                # If prices are high (positive deviation), reward for selling (negative net_exchange)
                # If prices are low (negative deviation), reward for buying (positive net_exchange)
                grid_support_reward = -net_exchange * price_deviation

                # Normalize to reasonable range
                grid_support_reward = np.clip(grid_support_reward / 5.0, -1.0, 1.0)

        return grid_support_reward

    def _compute_autonomy_reward(self, info: Dict) -> float:
        """Compute energy autonomy objective reward."""
        production, consumption = self._get_production_consumption()

        if production is None or consumption is None:
            return 0.0

        # Autonomy is the fraction of consumption met by own production
        if consumption > 0:
            # How much of our consumption is covered by our production
            self_consumption_ratio = min(production, consumption) / consumption

            # Bonus for having excess production (can sell to grid)
            if production > consumption:
                excess_bonus = min(0.2, (production - consumption) / consumption * 0.1)
            else:
                excess_bonus = 0.0

            autonomy_reward = self_consumption_ratio + excess_bonus
        else:
            # If no consumption, autonomy is perfect if we have no waste
            autonomy_reward = 1.0 if production == 0 else 0.5

        return np.clip(autonomy_reward, 0.0, 1.0)

    def _normalize_reward(self, reward: float, objective: str) -> float:
        """Normalize reward to standard range if enabled."""
        if isinstance(reward, dict):
            reward = reward.get('pcs', 0.0)  # Default to PCS reward

            # Convert to float
        reward = float(reward)
        if not self.normalize_rewards:
            return reward

        stats = self.reward_stats[objective]
        # Clip to expected range
        clipped = np.clip(reward, stats['min'], stats['max'])
        # Normalize to [-1, 1]
        normalized = 2 * (clipped - stats['min']) / (stats['max'] - stats['min']) - 1
        return normalized

    def reset(self, **kwargs):
        """Reset environment and multi-objective tracking."""
        observation, info = self.env.reset(**kwargs)

        # Re-extract environment components (they might have changed)
        self._get_environment_components()

        # Reset episode tracking with bounds checking
        if self.episode_count > 0 and len(self.current_episode_rewards) == self.num_objectives:
            obj_names = ['economic', 'battery_health', 'grid_support', 'autonomy']
            for i, obj_name in enumerate(obj_names):
                if i < len(self.current_episode_rewards):
                    self.episode_rewards[obj_name].append(self.current_episode_rewards[i])

        self.current_episode_rewards = np.zeros(self.num_objectives)
        self.step_count = 0
        self.episode_count += 1

        # Reset battery tracking
        self.prev_battery_level = self._get_battery_level()

        self.logger.info(f"Episode {self.episode_count} started")

        return observation, info

    def step(self, action):
        """Step environment and compute multi-objective rewards."""
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Handle multi-agent termination/truncation flags
        if isinstance(terminated, dict):
            terminated = any(terminated.values()) if terminated else False
        if isinstance(truncated, dict):
            truncated = any(truncated.values()) if truncated else False

        # Compute multi-objective rewards
        mo_rewards = np.zeros(self.num_objectives)

        # 1. Economic objective
        economic_reward = self._compute_economic_reward(reward, info)
        mo_rewards[0] = self._normalize_reward(economic_reward, 'economic')

        # 2. Battery health objective
        battery_health_reward = self._compute_battery_health_reward(info)
        mo_rewards[1] = self._normalize_reward(battery_health_reward, 'battery_health')

        # 3. Grid support objective
        grid_support_reward = self._compute_grid_support_reward(info)
        mo_rewards[2] = self._normalize_reward(grid_support_reward, 'grid_support')

        # 4. Energy autonomy objective
        autonomy_reward = self._compute_autonomy_reward(info)
        mo_rewards[3] = self._normalize_reward(autonomy_reward, 'autonomy')

        # Update episode tracking
        self.current_episode_rewards += mo_rewards
        self.step_count += 1

        # Update battery tracking
        current_battery_level = self._get_battery_level()
        if current_battery_level is not None:
            self.prev_battery_level = current_battery_level

        # Add MO information to info dict
        info.update({
            'mo_rewards': mo_rewards,
            'mo_rewards_raw': {
                'economic': economic_reward,
                'battery_health': battery_health_reward,
                'grid_support': grid_support_reward,
                'autonomy': autonomy_reward
            },
            'episode_mo_totals': self.current_episode_rewards.copy(),
            'step_count': self.step_count
        })

        # Log episode completion
        if terminated or truncated:
            self.logger.info(f"Episode {self.episode_count} completed with {self.step_count} steps")
            self.logger.info(f"Episode rewards: Economic={self.current_episode_rewards[0]:.3f}, "
                             f"Battery={self.current_episode_rewards[1]:.3f}, "
                             f"Grid={self.current_episode_rewards[2]:.3f}, "
                             f"Autonomy={self.current_episode_rewards[3]:.3f}")

        return observation, mo_rewards, terminated, truncated, info

    def get_episode_statistics(self) -> Dict[str, Any]:
        """Get statistics across all completed episodes."""
        if not self.episode_rewards:
            return {}

        stats = {}
        for obj_name, rewards in self.episode_rewards.items():
            if rewards:
                stats[obj_name] = {
                    'mean': np.mean(rewards),
                    'std': np.std(rewards),
                    'min': np.min(rewards),
                    'max': np.max(rewards),
                    'episodes': len(rewards)
                }

        return stats

    def get_pareto_front_data(self) -> Dict[str, List[float]]:
        """Get data for Pareto front analysis."""
        if not self.episode_rewards:
            return {}

        # Ensure all objectives have same number of episodes
        min_episodes = min(len(rewards) for rewards in self.episode_rewards.values() if rewards)

        pareto_data = {}
        for key, rewards in self.episode_rewards.items():
            if rewards:
                pareto_data[key] = rewards[:min_episodes]

        return pareto_data
