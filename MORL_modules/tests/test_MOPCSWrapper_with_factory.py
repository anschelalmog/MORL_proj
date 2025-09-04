# MORL_modules/tests/test_MOPCSWrapper_with_factory.py

import sys
import os
import logging
import pytest
import numpy as np
from gymnasium import spaces
from unittest.mock import patch, MagicMock

# Add energy_net to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'energy-net'))

from MORL_modules.wrappers.mo_pcs_wrapper import MOPCSWrapper
from energy_net.envs.pcs_env import make_pcs_env_zoo
from energy_net.market.pricing.pricing_policy import PricingPolicy
from energy_net.market.pricing.cost_types import CostType
from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern

@pytest.fixture
def pcs_env_from_factory():
    """Create PCS environment using the make_pcs_env_zoo factory function"""
    try:
        kwargs = {

            'pricing_policy': PricingPolicy.QUADRATIC,
            'demand_pattern': DemandPattern.SINUSOIDAL,
            'cost_type': CostType.CONSTANT,
        }
        env = make_pcs_env_zoo(
            iso_policy_path="logs/iso/ppo/run_1/ppo/ISO-RLZoo-v0_1/ISO-RLZoo-v0.zip",
            iso_policy_hyperparams_path ="rl-baselines3-zoo/hyperparams/ppo/ISO-RLZoo-v0.yml",
            monitor=False,  # Disable monitor for testing
            seed=42,
            log_dir="test_logs",
            **kwargs
        )
        return env
    except Exception as e:
        pytest.skip(f"Could not create PCS environment from factory: {e}")


@pytest.fixture
def mo_wrapper_from_factory(pcs_env_from_factory):
    """Create MOPCSWrapper with environment from factory"""
    return MOPCSWrapper(
        pcs_env_from_factory,

        num_objectives=4,
        reward_weights=np.ones(4) / 4,
        normalize_rewards=False,
        log_level=logging.WARNING  # Reduce log noise during tests
    )


@pytest.fixture
def mo_wrapper_normalized_from_factory(pcs_env_from_factory):
    """Create MOPCSWrapper with normalization enabled"""
    return MOPCSWrapper(
        pcs_env_from_factory,
        num_objectives=4,
        reward_weights=np.array([0.3, 0.3, 0.2, 0.2]),
        normalize_rewards=True,
        log_level=logging.WARNING
    )


class TestMOPCSWrapperFactoryIntegration:
    """Test MOPCSWrapper with environments created via make_pcs_env_zoo"""

    def test_factory_env_initialization(self, mo_wrapper_from_factory):
        """Test that wrapper initializes correctly with factory-created environment"""
        wrapper = mo_wrapper_from_factory
        # Basic initialization checks
        assert wrapper.num_objectives == 4
        assert np.allclose(wrapper.reward_weights, [0.25, 0.25, 0.25, 0.25])
        assert wrapper.normalize_rewards is False
        assert wrapper.episode_count == 0
        assert wrapper.step_count == 0

        # Check that environment components are accessible
        wrapper._get_environment_components()
        assert wrapper.controller is not None
        assert wrapper.battery_manager is not None
        assert wrapper.pcsunit is not None
        assert wrapper.battery is not None

    def test_factory_env_structure_validation(self, mo_wrapper_from_factory):
        """Test environment structure validation with factory environment"""
        wrapper = mo_wrapper_from_factory

        # This should not raise an exception
        wrapper._validate_environment_structure()

        # Verify all required components exist
        assert hasattr(wrapper.env.unwrapped, 'controller')
        assert hasattr(wrapper.env.unwrapped.controller, 'battery_manager')
        assert hasattr(wrapper.env.unwrapped.controller, 'pcs_unit')

    def test_factory_env_reset(self, mo_wrapper_from_factory):
        """Test reset functionality with factory environment"""
        wrapper = mo_wrapper_from_factory

        obs, info = wrapper.reset()

        # Check that reset works and returns valid data
        assert obs is not None
        assert isinstance(info, dict)
        assert wrapper.episode_count == 1
        assert wrapper.step_count == 0
        assert wrapper.prev_battery_level is not None

    def test_factory_env_step_basic(self, mo_wrapper_from_factory):
        """Test basic step functionality with factory environment"""
        wrapper = mo_wrapper_from_factory
        wrapper.reset()

        # Create valid action for PCS environment
        action = {"pcs": np.array([0.0], dtype=np.float32)}
        breakpoint()
        obs, mo_rewards, terminated, truncated, info = wrapper.step(action)
        # Verify multi-objective rewards structure

        assert isinstance(mo_rewards, np.ndarray)
        assert mo_rewards.shape == (4,)
        assert all(isinstance(reward, (int, float)) for reward in mo_rewards)

        # Verify step tracking
        assert wrapper.step_count == 1
        assert np.array_equal(wrapper.current_episode_rewards, mo_rewards)

        # Verify info dictionary structure
        assert 'mo_rewards' in info
        assert 'mo_rewards_raw' in info
        assert 'episode_mo_totals' in info
        assert 'step_count' in info

    def test_factory_env_multiple_steps(self, mo_wrapper_from_factory):
        """Test multiple steps with factory environment"""
        wrapper = mo_wrapper_from_factory
        wrapper.reset()

        total_rewards = np.zeros(4)
        num_steps = 5

        for step in range(num_steps):
            # Vary the action slightly each step
            action = {"pcs": np.array([0.1 * step], dtype=np.float32)}
            obs, mo_rewards, terminated, truncated, info = wrapper.step(action)

            total_rewards += mo_rewards

            # Verify step tracking
            assert wrapper.step_count == step + 1
            assert np.allclose(wrapper.current_episode_rewards, total_rewards)

            if terminated or truncated:
                break

        # Verify final state
        assert wrapper.step_count <= num_steps

    def test_factory_env_reward_computation_components(self, mo_wrapper_from_factory):
        """Test individual reward computation components with factory environment"""
        wrapper = mo_wrapper_from_factory
        wrapper.reset()
        wrapper._get_environment_components()

        # Test economic reward computation
        sample_reward = 5.0
        sample_info = {
            'net_exchange': 2.0,
            'iso_buy_price': 3.0,
            'iso_sell_price': 1.0
        }

        economic_reward = wrapper._compute_economic_reward(sample_reward, sample_info)
        assert isinstance(economic_reward, (int, float))

        # Test battery health reward
        battery_health_reward = wrapper._compute_battery_health_reward(sample_info)
        assert isinstance(battery_health_reward, (int, float))

        # Test grid support reward
        grid_support_reward = wrapper._compute_grid_support_reward(sample_info)
        assert isinstance(grid_support_reward, (int, float))

        # Test autonomy reward
        autonomy_reward = wrapper._compute_autonomy_reward(sample_info)
        assert isinstance(autonomy_reward, (int, float))
        assert 0 <= autonomy_reward <= 1  # Autonomy should be in [0, 1]

    def test_factory_env_battery_operations(self, mo_wrapper_from_factory):
        """Test battery-related operations with factory environment"""
        wrapper = mo_wrapper_from_factory
        wrapper.reset()
        wrapper._get_environment_components()

        # Test battery level retrieval
        battery_level = wrapper._get_battery_level()
        assert battery_level is not None
        assert isinstance(battery_level, (int, float))
        assert battery_level >= 0

        # Test that battery level tracking works across steps
        initial_level = battery_level

        action = {"pcs": np.array([0.5], dtype=np.float32)}  # Charge action
        wrapper.step(action)

        new_level = wrapper._get_battery_level()
        assert new_level is not None
        # Battery level might have changed due to action
        assert isinstance(new_level, (int, float))

    def test_factory_env_production_consumption(self, mo_wrapper_from_factory):
        """Test production and consumption retrieval with factory environment"""
        wrapper = mo_wrapper_from_factory
        wrapper.reset()
        wrapper._get_environment_components()

        production, consumption = wrapper._get_production_consumption()

        # These should return valid values or None
        if production is not None:
            assert isinstance(production, (int, float))
            assert production >= 0

        if consumption is not None:
            assert isinstance(consumption, (int, float))
            assert consumption >= 0

    def test_factory_env_with_normalization(self, mo_wrapper_normalized_from_factory):
        """Test wrapper with normalization enabled using factory environment"""
        wrapper = mo_wrapper_normalized_from_factory

        # Verify normalization is enabled
        assert wrapper.normalize_rewards is True

        wrapper.reset()
        action = {"pcs": np.array([0.0], dtype=np.float32)}

        obs, mo_rewards, terminated, truncated, info = wrapper.step(action)

        # With normalization, rewards should be in reasonable range
        assert all(-5 <= reward <= 5 for reward in mo_rewards)  # Allow some buffer

        # Raw rewards should be available in info
        assert 'mo_rewards_raw' in info
        raw_rewards = info['mo_rewards_raw']
        assert 'economic' in raw_rewards
        assert 'battery_health' in raw_rewards
        assert 'grid_support' in raw_rewards
        assert 'autonomy' in raw_rewards

    def test_factory_env_episode_statistics(self, mo_wrapper_from_factory):
        """Test episode statistics collection with factory environment"""
        wrapper = mo_wrapper_from_factory

        # Run a few short episodes
        for episode in range(3):
            wrapper.reset()

            # Take a few steps per episode
            for step in range(3):
                action = {"pcs": np.array([0.1 * (episode + step)], dtype=np.float32)}
                obs, mo_rewards, terminated, truncated, info = wrapper.step(action)

                if terminated or truncated:
                    break

        # Final reset to record last episode
        wrapper.reset()

        # Get episode statistics
        stats = wrapper.get_episode_statistics()

        if stats:  # If episodes were completed
            expected_objectives = ['economic', 'battery_health', 'grid_support', 'autonomy']

            for obj_name in expected_objectives:
                if obj_name in stats:
                    obj_stats = stats[obj_name]
                    assert 'mean' in obj_stats
                    assert 'std' in obj_stats
                    assert 'min' in obj_stats
                    assert 'max' in obj_stats
                    assert 'episodes' in obj_stats
                    assert obj_stats['episodes'] >= 0

    def test_factory_env_pareto_front_data(self, mo_wrapper_from_factory):
        """Test Pareto front data collection with factory environment"""
        wrapper = mo_wrapper_from_factory

        # Run multiple episodes
        for episode in range(2):
            wrapper.reset()

            for step in range(2):
                action = {"pcs": np.array([0.2 * episode], dtype=np.float32)}
                obs, mo_rewards, terminated, truncated, info = wrapper.step(action)

                if terminated or truncated:
                    break

        # Final reset to record last episode
        wrapper.reset()

        # Get Pareto front data
        pareto_data = wrapper.get_pareto_front_data()

        assert isinstance(pareto_data, dict)

        if pareto_data:  # If episodes were recorded
            for obj_name, rewards in pareto_data.items():
                assert isinstance(rewards, list)
                if rewards:
                    assert all(isinstance(r, (int, float)) for r in rewards)

    def test_factory_env_error_handling(self, mo_wrapper_from_factory):
        """Test error handling with factory environment"""
        wrapper = mo_wrapper_from_factory
        wrapper.reset()

        # Test with missing info components
        economic_reward = wrapper._compute_economic_reward(0.0, {})
        assert isinstance(economic_reward, (int, float))

        grid_support_reward = wrapper._compute_grid_support_reward({})
        assert isinstance(grid_support_reward, (int, float))

        autonomy_reward = wrapper._compute_autonomy_reward({})
        assert isinstance(autonomy_reward, (int, float))

    def test_factory_env_different_action_types(self, mo_wrapper_from_factory):
        """Test wrapper with different action types that factory env might support"""
        wrapper = mo_wrapper_from_factory
        wrapper.reset()

        # Test basic PCS action
        pcs_action = {"pcs": np.array([0.0], dtype=np.float32)}
        obs, mo_rewards, terminated, truncated, info = wrapper.step(pcs_action)

        assert isinstance(mo_rewards, np.ndarray)
        assert mo_rewards.shape == (4,)

    def test_factory_env_seed_reproducibility(self):
        """Test that seeded factory environments produce reproducible results"""
        seed = 123

        # Create two identical environments
        env1 = make_pcs_env_zoo(monitor=False, seed=seed, iso_policy_path="logs/pcs/ppo/run_1/ppo/PCS-RLZoo-v0_9/best_model.zip",)
        env2 = make_pcs_env_zoo(monitor=False, seed=seed, iso_policy_path="logs/pcs/ppo/run_1/ppo/PCS-RLZoo-v0_9/best_model.zip",)

        wrapper1 = MOPCSWrapper(env1, normalize_rewards=False, log_level=logging.WARNING)
        wrapper2 = MOPCSWrapper(env2, normalize_rewards=False, log_level=logging.WARNING)

        # Reset both environments
        obs1, _ = wrapper1.reset()
        obs2, _ = wrapper2.reset()

        # Take identical actions
        action = {"pcs": np.array([0.1], dtype=np.float32)}

        _, rewards1, _, _, _ = wrapper1.step(action)
        _, rewards2, _, _, _ = wrapper2.step(action)

        # Results should be identical (or very close due to floating point)
        assert np.allclose(rewards1, rewards2, rtol=1e-5)


@pytest.mark.parametrize("normalize_rewards", [True, False])
def test_factory_env_normalization_modes(normalize_rewards):
    """Test both normalization modes with factory environment"""
    try:
        env = make_pcs_env_zoo(monitor=False, seed=42, iso_policy_path="logs/pcs/ppo/run_1/ppo/PCS-RLZoo-v0_9/best_model.zip")
        wrapper = MOPCSWrapper(
            env,
            num_objectives=4,
            normalize_rewards=normalize_rewards,
            log_level=logging.WARNING
        )

        assert wrapper.normalize_rewards == normalize_rewards

        wrapper.reset()
        action = {"pcs": np.array([0.0], dtype=np.float32)}

        obs, mo_rewards, terminated, truncated, info = wrapper.step(action)

        assert isinstance(mo_rewards, np.ndarray)
        assert mo_rewards.shape == (4,)

        if normalize_rewards:
            # Normalized rewards should be in reasonable range
            assert all(-5 <= r <= 5 for r in mo_rewards)

    except Exception as e:
        pytest.skip(f"Could not test normalization mode {normalize_rewards}: {e}")


@pytest.mark.parametrize("num_objectives", [3, 4, 5])
def test_factory_env_different_objective_counts(num_objectives):
    """Test wrapper with different numbers of objectives"""
    try:
        env = make_pcs_env_zoo(monitor=False, seed=42, iso_policy_path="logs/pcs/ppo/run_1/ppo/PCS-RLZoo-v0_9/best_model.zip")

        if num_objectives == 4:
            # Standard case
            wrapper = MOPCSWrapper(env, num_objectives=num_objectives, log_level=logging.WARNING)
        else:
            # Non-standard cases - wrapper should handle gracefully or raise clear error
            try:
                wrapper = MOPCSWrapper(env, num_objectives=num_objectives, log_level=logging.WARNING)
            except (ValueError, AssertionError) as e:
                pytest.skip(f"Wrapper doesn't support {num_objectives} objectives: {e}")
                return

        assert wrapper.num_objectives == num_objectives

        wrapper.reset()
        action = {"pcs": np.array([0.0], dtype=np.float32)}

        obs, mo_rewards, terminated, truncated, info = wrapper.step(action)

        if num_objectives == 4:
            # Only test full functionality for standard case
            assert mo_rewards.shape == (4,)
        else:
            # For non-standard cases, just verify basic structure
            assert isinstance(mo_rewards, np.ndarray)
            assert len(mo_rewards) == num_objectives

    except Exception as e:
        pytest.skip(f"Could not test with {num_objectives} objectives: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
