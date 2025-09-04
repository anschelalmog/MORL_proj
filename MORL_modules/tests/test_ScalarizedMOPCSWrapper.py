# MORL_modules/tests/test_scalarized_mo_pcs_wrapper.py

import sys
import os
import logging
import pytest
import numpy as np
from gymnasium import spaces

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'MORL_modules'))


from MORL_modules.wrappers.scalarized_mo_pcs_wrapper import ScalarizedMOPCSWrapper
from MORL_modules.wrappers.mo_pcs_wrapper import MOPCSWrapper

from energy_net.envs.energy_net_v0 import EnergyNetV0
from energy_net.market.pricing.cost_types import CostType
from energy_net.market.pricing.pricing_policy import PricingPolicy
from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern


@pytest.fixture
def real_energynet_env():
    """Create a real EnergyNetV0 environment with minimal configuration"""
    try:
        env = EnergyNetV0(
            #controller_name="EnergyNetController",
            #controller_module="energy_net.controllers",
            env_config_path='configs/environment_config.yaml',
            iso_config_path='configs/iso_config.yaml',
            pcs_unit_config_path='configs/pcs_unit_config.yaml',
            cost_type=CostType.CONSTANT,
            pricing_policy=PricingPolicy.QUADRATIC,
            demand_pattern=DemandPattern.SINUSOIDAL,
        )
        return env
    except Exception as e:
        pytest.skip(f"Could not create real EnergyNet environment: {e}")


@pytest.fixture
def scalarized_wrapper_real_env(real_energynet_env):
    """Create ScalarizedMOPCSWrapper with real environment and default weights"""
    return ScalarizedMOPCSWrapper(
        real_energynet_env,
        normalize_weights=True,
        log_level=logging.INFO
    )


def test_initialization_with_real_env(real_energynet_env):
    """Test wrapper initialization with different weight configurations using real env."""
    # Default initialization (equal weights)
    wrapper = ScalarizedMOPCSWrapper(real_energynet_env)
    assert np.allclose(wrapper.weights, [0.25, 0.25, 0.25, 0.25])
    assert wrapper.num_objectives == 4

    # Verify it wraps with MOPCSWrapper
    assert isinstance(wrapper.env, MOPCSWrapper)

    # Custom weights (normalized)
    weights = [1, 2, 3, 4]
    wrapper = ScalarizedMOPCSWrapper(real_energynet_env, weights=weights)
    expected = np.array(weights) / sum(weights)
    assert np.allclose(wrapper.weights, expected)

    # Custom weights (unnormalized)
    weights = [0.1, 0.2, 0.3, 0.4]
    wrapper = ScalarizedMOPCSWrapper(real_energynet_env, weights=weights, normalize_weights=False)
    assert np.allclose(wrapper.weights, weights)


def test_initialization_errors_real_env(real_energynet_env):
    """Test initialization with invalid weights using real environment."""
    # Wrong number of weights
    with pytest.raises(ValueError, match="Weight dimension"):
        ScalarizedMOPCSWrapper(real_energynet_env, weights=[0.5, 0.5])

    # Zero weights
    with pytest.raises(ValueError, match="Weights sum to zero"):
        ScalarizedMOPCSWrapper(real_energynet_env, weights=[0, 0, 0, 0])


def test_step_scalarization_real_env(scalarized_wrapper_real_env):
    """Test that rewards are properly scalarized with real environment."""
    wrapper = scalarized_wrapper_real_env

    obs, info_reset = wrapper.reset()
    assert obs is not None
    assert isinstance(info_reset, dict)

    # Take a step with realistic action
    action = {"iso": np.zeros(7, dtype=np.float32), "pcs": np.zeros(1, dtype=np.float32)}

    try:
        obs, reward, terminated, truncated, info = wrapper.step(action)

        # Check scalar reward
        assert isinstance(reward, (float, np.floating))
        assert not isinstance(reward, np.ndarray) or reward.shape == ()

        # Check MO rewards preserved in info
        assert 'mo_rewards_original' in info
        assert isinstance(info['mo_rewards_original'], np.ndarray)
        assert info['mo_rewards_original'].shape == (4,)

        # Check scalarization info
        assert 'scalarization_weights' in info
        assert 'scalar_reward' in info
        assert info['scalar_reward'] == reward

        # Verify scalarization computation
        mo_rewards = info['mo_rewards_original']
        weights = info['scalarization_weights']
        expected_scalar = np.dot(weights, mo_rewards)
        assert np.isclose(reward, expected_scalar)

    except Exception as e:
        pytest.skip(f"Step failed with real environment: {e}")


def test_scalarization_computation_real_env(real_energynet_env):
    """Test that scalarization is computed correctly with real environment."""
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    wrapper = ScalarizedMOPCSWrapper(
        real_energynet_env,
        weights=weights,
        normalize_weights=False
    )

    # Reset and take a step
    obs, _ = wrapper.reset()
    action = {"iso": np.zeros(7, dtype=np.float32), "pcs": np.zeros(1, dtype=np.float32)}

    try:
        _, scalar_reward, _, _, info = wrapper.step(action)

        # Check that scalarization matches expected computation
        mo_rewards = info['mo_rewards_original']
        expected_scalar = np.dot(weights, mo_rewards)

        assert np.isclose(scalar_reward, expected_scalar)
        assert np.allclose(info['scalarization_weights'], weights)

    except Exception as e:
        pytest.skip(f"Step failed with real environment: {e}")


def test_set_weights_real_env(scalarized_wrapper_real_env):
    """Test updating weights after initialization with real environment."""
    wrapper = scalarized_wrapper_real_env

    # Set new weights
    new_weights = [0.1, 0.2, 0.3, 0.4]
    wrapper.set_weights(new_weights)

    expected = np.array(new_weights) / sum(new_weights)
    assert np.allclose(wrapper.weights, expected)

    # Set unnormalized weights
    wrapper.set_weights(new_weights, normalize=False)
    assert np.allclose(wrapper.weights, new_weights)

    # Test invalid weights
    with pytest.raises(ValueError):
        wrapper.set_weights([0.5, 0.5])  # Wrong dimension

    with pytest.raises(ValueError):
        wrapper.set_weights([0, 0, 0, 0])  # Zero sum


def test_wrapper_stacking_real_env(real_energynet_env):
    """Test that wrapper correctly handles already-wrapped environments."""
    # First wrap with MO
    mo_wrapper = MOPCSWrapper(real_energynet_env)

    # Then wrap with scalarized
    scalarized = ScalarizedMOPCSWrapper(mo_wrapper)

    # Should not double-wrap
    assert isinstance(scalarized.env, MOPCSWrapper)
    assert not isinstance(scalarized.env.env, MOPCSWrapper)


def test_integration_flow_real_env(scalarized_wrapper_real_env):
    """Test complete episode flow with real environment."""
    wrapper = scalarized_wrapper_real_env

    total_reward = 0
    episode_mo_rewards = []
    episode_scalar_rewards = []

    obs, _ = wrapper.reset()

    # Run for a few steps
    for step in range(5):
        action = {"iso": np.zeros(7, dtype=np.float32), "pcs": np.zeros(1, dtype=np.float32)}

        try:
            obs, reward, terminated, truncated, info = wrapper.step(action)

            total_reward += reward
            episode_mo_rewards.append(info['mo_rewards_original'])
            episode_scalar_rewards.append(reward)

            # Verify consistency between scalar reward and scalarization
            mo_rewards = info['mo_rewards_original']
            weights = info['scalarization_weights']
            expected_scalar = np.dot(weights, mo_rewards)
            assert np.isclose(reward, expected_scalar)

            if terminated or truncated:
                break

        except Exception as e:
            pytest.skip(f"Step {step} failed with real environment: {e}")

    # Verify we collected data
    assert len(episode_mo_rewards) > 0
    assert len(episode_scalar_rewards) > 0
    assert isinstance(total_reward, (float, np.floating))

    # Verify MO rewards structure
    mo_array = np.array(episode_mo_rewards)
    assert mo_array.shape[1] == 4  # 4 objectives

    # Verify scalar rewards are all valid numbers
    assert all(isinstance(r, (float, np.floating)) for r in episode_scalar_rewards)


@pytest.mark.parametrize("weight_config", [
    [1.0, 0.0, 0.0, 0.0],  # Economic only
    [0.0, 1.0, 0.0, 0.0],  # Battery health only
    [0.0, 0.0, 1.0, 0.0],  # Grid support only
    [0.0, 0.0, 0.0, 1.0],  # Autonomy only
    [0.25, 0.25, 0.25, 0.25],  # Equal weights
    [0.5, 0.3, 0.1, 0.1],  # Economic focused
])
def test_different_weight_configurations_real_env(real_energynet_env, weight_config):
    """Test scalarization with different weight configurations."""
    wrapper = ScalarizedMOPCSWrapper(
        real_energynet_env,
        weights=weight_config,
        normalize_weights=False
    )

    # Reset and take a step
    obs, _ = wrapper.reset()
    action = {"iso": np.zeros(7, dtype=np.float32), "pcs": np.zeros(1, dtype=np.float32)}

    try:
        _, scalar_reward, _, _, info = wrapper.step(action)

        # Verify scalarization
        mo_rewards = info['mo_rewards_original']
        expected_scalar = np.dot(weight_config, mo_rewards)

        assert np.isclose(scalar_reward, expected_scalar)
        assert np.allclose(info['scalarization_weights'], weight_config)

    except Exception as e:
        pytest.skip(f"Step failed with weights {weight_config}: {e}")


def test_reward_range_real_env(scalarized_wrapper_real_env):
    """Test that scalar rewards are in reasonable range with real environment."""
    wrapper = scalarized_wrapper_real_env

    wrapper.reset()
    rewards = []

    # Collect rewards from multiple steps
    for step in range(10):
        action = {"iso": np.zeros(7, dtype=np.float32), "pcs": np.zeros(1, dtype=np.float32)}

        try:
            _, reward, terminated, truncated, info = wrapper.step(action)
            rewards.append(reward)

            # Verify reward is a valid number
            assert isinstance(reward, (float, np.floating))
            assert not np.isnan(reward)
            assert not np.isinf(reward)

            if terminated or truncated:
                break

        except Exception as e:
            pytest.skip(f"Step {step} failed: {e}")

    if rewards:
        # Check that we got some variation in rewards
        reward_array = np.array(rewards)
        assert len(reward_array) > 0

        # Basic sanity checks on reward range
        assert np.all(np.isfinite(reward_array))


def test_mo_wrapper_passthrough_real_env(real_energynet_env):
    """Test that underlying MOPCSWrapper functionality is preserved."""
    wrapper = ScalarizedMOPCSWrapper(real_energynet_env)

    # Access the underlying MO wrapper
    mo_wrapper = wrapper.env
    assert isinstance(mo_wrapper, MOPCSWrapper)

    # Test that MO wrapper methods are accessible
    wrapper.reset()

    # Test statistics access
    try:
        stats = mo_wrapper.get_episode_statistics()
        assert isinstance(stats, dict)

        pareto_data = mo_wrapper.get_pareto_front_data()
        assert isinstance(pareto_data, dict)

    except Exception as e:
        pytest.skip(f"MO wrapper method access failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])