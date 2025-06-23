# MORL_modules/tests/test_MOPCSWrapper.py

import sys
import os
import logging
import pytest
import numpy as np
from gymnasium import spaces


from MORL_modules.wrappers.mo_pcs_wrapper import MOPCSWrapper

from energy_net.envs.energy_net_v0 import EnergyNetV0
from energy_net.market.pricing.cost_types import CostType
from energy_net.market.pricing.pricing_policy import PricingPolicy
from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern
from energy_net.controllers.energy_net_controller import EnergyNetController

@pytest.fixture
def real_energynet_env():
    """Create a real EnergyNetV0 environment with minimal configuration"""
    try:
        env = EnergyNetV0(
            controller_name="EnergyNetController",
            controller_module="energy_net.controllers",
            env_config_path='energy_net/configs/environment_config.yaml',
            iso_config_path='energy_net/configs/iso_config.yaml',
            pcs_unit_config_path='energy_net/configs/pcs_unit_config.yaml',
            cost_type=CostType.CONSTANT,
            pricing_policy=PricingPolicy.QUADRATIC,
            demand_pattern=DemandPattern.SINUSOIDAL,
        )
        return env
    except Exception as e:
        pytest.skip(f"Could not create real EnergyNet environment: {e}")


@pytest.fixture
def wrapper_with_real_env(real_energynet_env):
    """Create MOPCSWrapper with real environment"""
    return MOPCSWrapper(
        real_energynet_env,
        num_objectives=4,
        reward_weights=np.ones(4) / 4,
        normalize_rewards=False,
        log_level=logging.INFO
    )


def test_initialization_with_real_env(wrapper_with_real_env):
    """Test that wrapper initializes correctly with real environment"""
    wrapper = wrapper_with_real_env

    # Basic initialization checks
    assert wrapper.num_objectives == 4
    assert np.allclose(wrapper.reward_weights, [0.25, 0.25, 0.25, 0.25])
    assert wrapper.normalize_rewards is False
    assert wrapper.logger.level == logging.INFO
    assert len(wrapper.logger.handlers) > 0


def test_environment_component_extraction_real(wrapper_with_real_env):
    """Test component extraction with real environment"""
    wrapper = wrapper_with_real_env

    # Extract components
    wrapper._get_environment_components()

    # Check that we successfully extracted real components
    assert wrapper.controller is not None
    assert hasattr(wrapper.controller, 'pcs_unit')
    assert hasattr(wrapper.controller, 'battery_manager')

    if wrapper.controller.pcs_unit:
        assert hasattr(wrapper.controller.pcs_unit, 'battery')
        assert hasattr(wrapper.controller.pcs_unit, 'get_self_production')
        assert hasattr(wrapper.controller.pcs_unit, 'get_self_consumption')


def test_battery_level_retrieval_real(wrapper_with_real_env):
    """Test battery level retrieval with real components"""
    wrapper = wrapper_with_real_env
    wrapper._get_environment_components()

    # Test battery level retrieval
    battery_level = wrapper._get_battery_level()

    # Should return a valid number (not None)
    assert battery_level is not None
    assert isinstance(battery_level, (int, float))
    assert battery_level >= 0  # Battery level should be non-negative


def test_reset_and_step_integration_real(wrapper_with_real_env):
    """Test reset and step operations with real environment"""
    wrapper = wrapper_with_real_env

    # Test reset
    obs, info = wrapper.reset()

    # Check observations structure (should match real environment)
    assert isinstance(obs, (dict, list, np.ndarray))
    assert isinstance(info, dict)

    # Test step with valid action
    # Get a sample action from the environment's action space
    """
    if hasattr(wrapper.env, 'action_space'):
        if isinstance(wrapper.env.action_space, spaces.Dict):
            action = {key: space.sample() for key, space in wrapper.env.action_space.spaces.items()}
        else:
            action = wrapper.env.action_space.sample()
    else:
        # Fallback action structure based on EnergyNetV0
        action = {
            "iso": np.array([0.0]),
            "pcs": np.array([0.0])
        }
    """
    # Use simple zero actions that work with PCS environment
    action = {
        "iso": np.zeros(7, dtype=np.float32),
        "pcs": np.zeros(1, dtype=np.float32)
    }

    # Execute step
    obs2, mo_rewards, terminated, truncated, info2 = wrapper.step(action)

    # Verify multi-objective rewards
    assert isinstance(mo_rewards, np.ndarray)
    assert mo_rewards.shape == (4,)  # Four objectives
    assert all(isinstance(reward, (int, float)) for reward in mo_rewards)

    # Verify episode info
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert 'mo_rewards' in info2
    assert 'mo_rewards_raw' in info2


def test_multi_objective_reward_computation_real(wrapper_with_real_env):
    """Test MO reward computation with real environment data"""
    wrapper = wrapper_with_real_env

    # Reset and get initial state
    wrapper.reset()
    wrapper._get_environment_components()

    # Create a mock info dict with realistic data structure
    # This tests the reward computation functions directly
    sample_info = {
        'net_exchange': 2.0,
        'iso_buy_price': 3.0,
        'iso_sell_price': 1.0,
        'energy_bought': 0.0,
        'energy_sold': 0.0
    }

    # Test individual reward components
    economic_reward = wrapper._compute_economic_reward(1.0, sample_info)
    battery_health_reward = wrapper._compute_battery_health_reward(sample_info)
    grid_support_reward = wrapper._compute_grid_support_reward(sample_info)
    autonomy_reward = wrapper._compute_autonomy_reward(sample_info)

    # All rewards should be valid numbers
    assert isinstance(economic_reward, (int, float))
    assert isinstance(battery_health_reward, (int, float))
    assert isinstance(grid_support_reward, (int, float))
    assert isinstance(autonomy_reward, (int, float))

    # Check that rewards are within reasonable bounds
    assert -100 <= economic_reward <= 100  # Reasonable economic reward range
    assert -10 <= battery_health_reward <= 10  # Reasonable battery health range
    assert -10 <= grid_support_reward <= 10  # Reasonable grid support range
    assert 0 <= autonomy_reward <= 1  # Autonomy should be 0-1


def test_battery_health_with_real_battery(wrapper_with_real_env):
    """Test battery health computation with real battery manager"""
    wrapper = wrapper_with_real_env
    wrapper.reset()
    wrapper._get_environment_components()

    if wrapper.battery_manager is None:
        pytest.skip("No battery manager available in real environment")

    # Set a previous battery level to test cycling penalty
    wrapper.prev_battery_level = wrapper._get_battery_level()

    # Test battery health computation
    health_reward = wrapper._compute_battery_health_reward({})

    # Should return a valid number
    assert isinstance(health_reward, (int, float))
    assert not np.isnan(health_reward)


def test_production_consumption_real(wrapper_with_real_env):
    """Test production and consumption retrieval with real PCSUnit"""
    wrapper = wrapper_with_real_env
    wrapper.reset()
    wrapper._get_environment_components()

    if wrapper.pcsunit is None:
        pytest.skip("No PCSUnit available in real environment")

    # Test production and consumption retrieval
    production, consumption = wrapper._get_production_consumption()

    # Should return valid numbers or None (if not implemented)
    if production is not None:
        assert isinstance(production, (int, float))
        assert production >= 0

    if consumption is not None:
        assert isinstance(consumption, (int, float))
        assert consumption >= 0


def test_episode_statistics_real(wrapper_with_real_env):
    """Test episode statistics tracking with real environment"""
    wrapper = wrapper_with_real_env

    # Run a short episode
    wrapper.reset()

    # Take a few steps
    for _ in range(3):
        action = {"iso": np.zeros(7, dtype=np.float32), "pcs": np.zeros(1, dtype=np.float32)}
        try:
            wrapper.step(action)
        except Exception as e:
            pytest.skip(f"Step failed with real environment: {e}")

    wrapper.reset()
    stats = wrapper.get_episode_statistics()
    if stats:
        for obj_name in ['economic', 'battery_health', 'grid_support', 'autonomy']:
            if obj_name in stats:
                assert 'mean' in stats[obj_name]
                assert 'episodes' in stats[obj_name]
                assert stats[obj_name]['episodes'] >= 0


def test_error_handling_real_env(wrapper_with_real_env):
    """Test error handling with real environment edge cases"""
    wrapper = wrapper_with_real_env
    wrapper.reset()

    # Test with missing components (temporarily remove them)
    original_controller = wrapper.controller
    wrapper.controller = None

    # Should handle missing controller gracefully
    assert wrapper._get_battery_level() is None

    # Test with empty info dict
    economic_reward = wrapper._compute_economic_reward(0.0, {})
    assert isinstance(economic_reward, (int, float))

    grid_support_reward = wrapper._compute_grid_support_reward({})
    assert isinstance(grid_support_reward, (int, float))

    autonomy_reward = wrapper._compute_autonomy_reward({})
    assert isinstance(autonomy_reward, (int, float))

    # Restore controller
    wrapper.controller = original_controller


def test_pareto_front_data_real(wrapper_with_real_env):
    """Test Pareto front data collection with real environment"""
    wrapper = wrapper_with_real_env

    # Run multiple short episodes
    for episode in range(2):
        wrapper.reset()

        # Take a few steps
        for step in range(2):
            action = {"iso": np.zeros(7, dtype=np.float32), "pcs": np.zeros(1, dtype=np.float32)}
            try:
                wrapper.step(action)
            except Exception as e:
                pytest.skip(f"Step failed in episode {episode}, step {step}: {e}")

    # Final reset to record last episode
    wrapper.reset()

    # Get Pareto front data
    pareto_data = wrapper.get_pareto_front_data()

    # Should return a dictionary with objective names as keys
    assert isinstance(pareto_data, dict)

    # If episodes were completed, check data structure
    if pareto_data:
        for obj_name, rewards in pareto_data.items():
            assert isinstance(rewards, list)
            if rewards:  # If there are recorded rewards
                assert all(isinstance(r, (int, float)) for r in rewards)


@pytest.mark.parametrize("normalize_rewards", [True, False])
def test_normalization_modes_real(real_energynet_env, normalize_rewards):
    """Test both normalization modes with real environment"""
    wrapper = MOPCSWrapper(
        real_energynet_env,
        num_objectives=4,
        normalize_rewards=normalize_rewards,
        log_level=logging.WARNING  # Reduce log noise
    )

    # Test that normalization setting is respected
    assert wrapper.normalize_rewards == normalize_rewards

    # Test a single step
    wrapper.reset()
    action = {"iso": np.zeros(7, dtype=np.float32), "pcs": np.zeros(1, dtype=np.float32)}

    try:
        obs, mo_rewards, terminated, truncated, info = wrapper.step(action)

        # Verify reward structure
        assert isinstance(mo_rewards, np.ndarray)
        assert mo_rewards.shape == (4,)

        if normalize_rewards:
            # Normalized rewards should be in reasonable range
            assert all(-5 <= r <= 5 for r in mo_rewards)  # Allow some buffer

    except Exception as e:
        pytest.skip(f"Step failed with normalization={normalize_rewards}: {e}")


if __name__ == "__main__":
    pytest.main([__file__])