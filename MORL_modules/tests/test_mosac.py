# tests/test_mosac_simple.py

import sys
import os
import pytest
import numpy as np
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the MOSAC agent
try:
    from agents.mosac.mosac import MOSAC
except ImportError as e:
    pytest.skip(f"Could not import MOSAC: {e}")

# Import environment components
try:
    from MORL_modules.wrappers.mo_pcs_wrapper import MOPCSWrapper
    from energy_net.envs.energy_net_v0 import EnergyNetV0
    from energy_net.market.pricing.cost_types import CostType
    from energy_net.market.pricing.pricing_policy import PricingPolicy
    from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern
except ImportError as e:
    pytest.skip(f"Could not import EnergyNet components: {e}")


@pytest.fixture
def energynet_env():
    """Create a real EnergyNetV0 environment."""
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
        pytest.skip(f"Could not create EnergyNet environment: {e}")


@pytest.fixture
def mo_wrapped_env(energynet_env):
    """Create MOPCSWrapper with EnergyNet environment."""
    return MOPCSWrapper(
        energynet_env,
        num_objectives=4,
        reward_weights=np.ones(4) / 4,
        normalize_rewards=False,
        log_level=logging.WARNING  # Reduce log noise during tests
    )


def test_mosac_basic_initialization(mo_wrapped_env):
    """Test basic MOSAC initialization with MOPCSWrapper."""

    # Test initialization with default parameters
    agent = MOSAC(
        env=mo_wrapped_env,
        num_objectives=4,
        verbose=0,
        learning_starts=10,  # Small values for testing
        buffer_size=1000
    )

    # Check basic attributes
    assert agent.num_objectives == 4
    assert len(agent.preference_weights) == 4
    assert np.allclose(agent.preference_weights, [0.25, 0.25, 0.25, 0.25])
    assert np.isclose(np.sum(agent.preference_weights), 1.0)

    # Check that the agent inherits from SAC
    assert hasattr(agent, 'policy')
    assert hasattr(agent, 'replay_buffer')
    assert hasattr(agent, 'learn')

    print("✅ Basic initialization test passed!")


def test_mosac_custom_initialization(mo_wrapped_env):
    """Test MOSAC initialization with custom preference weights."""

    # Test with custom preference weights
    custom_weights = np.array([0.4, 0.3, 0.2, 0.1])
    agent = MOSAC(
        env=mo_wrapped_env,
        num_objectives=4,
        preference_weights=custom_weights,
        verbose=0,
        learning_starts=10,
        buffer_size=1000
    )

    # Check that weights are set correctly
    assert np.allclose(agent.preference_weights, custom_weights)
    assert np.isclose(np.sum(agent.preference_weights), 1.0)

    # Test with unnormalized weights
    unnormalized_weights = np.array([2.0, 4.0, 6.0, 8.0])
    agent2 = MOSAC(
        env=mo_wrapped_env,
        num_objectives=4,
        preference_weights=unnormalized_weights,
        verbose=0,
        learning_starts=10,
        buffer_size=1000
    )

    # Should be normalized to sum to 1
    expected_normalized = unnormalized_weights / np.sum(unnormalized_weights)
    assert np.allclose(agent2.preference_weights, expected_normalized)
    assert np.isclose(np.sum(agent2.preference_weights), 1.0)

    print("✅ Custom initialization test passed!")


def test_preference_weight_handling(mo_wrapped_env):
    """Test preference weight get/set functionality."""

    agent = MOSAC(
        env=mo_wrapped_env,
        num_objectives=4,
        verbose=0,
        learning_starts=10,
        buffer_size=1000
    )

    # Test getting weights
    weights = agent.get_preference_weights()
    assert len(weights) == 4
    assert np.allclose(weights, [0.25, 0.25, 0.25, 0.25])

    # Test setting new weights
    new_weights = np.array([0.1, 0.2, 0.3, 0.4])
    agent.set_preference_weights(new_weights)

    updated_weights = agent.get_preference_weights()
    assert np.allclose(updated_weights, new_weights)
    assert np.isclose(np.sum(updated_weights), 1.0)

    # Test setting unnormalized weights
    unnormalized = np.array([1.0, 2.0, 3.0, 4.0])
    agent.set_preference_weights(unnormalized)

    final_weights = agent.get_preference_weights()
    expected = unnormalized / np.sum(unnormalized)
    assert np.allclose(final_weights, expected)
    assert np.isclose(np.sum(final_weights), 1.0)

    # Test error handling for wrong number of weights
    with pytest.raises(ValueError):
        agent.set_preference_weights(np.array([0.5, 0.5]))  # Only 2 weights instead of 4

    print("✅ Preference weight handling test passed!")


def test_reward_scalarization(mo_wrapped_env):
    """Test reward scalarization functionality."""

    agent = MOSAC(
        env=mo_wrapped_env,
        num_objectives=4,
        verbose=0,
        learning_starts=10,
        buffer_size=1000
    )

    # Test basic scalarization with equal weights
    mo_reward = np.array([1.0, 2.0, 3.0, 4.0])
    scalar_reward = agent.scalarize_reward(mo_reward)

    # With equal weights [0.25, 0.25, 0.25, 0.25]
    expected = np.dot(agent.preference_weights, mo_reward)
    assert np.isclose(scalar_reward, expected)
    assert np.isclose(scalar_reward, 2.5)  # (1+2+3+4)/4 = 2.5

    # Test with different preference weights
    agent.set_preference_weights(np.array([1.0, 0.0, 0.0, 0.0]))  # Only economic objective
    scalar_reward = agent.scalarize_reward(mo_reward)
    assert np.isclose(scalar_reward, 1.0)  # Should equal first objective

    # Test with another weight configuration
    agent.set_preference_weights(np.array([0.0, 0.0, 0.0, 1.0]))  # Only autonomy objective
    scalar_reward = agent.scalarize_reward(mo_reward)
    assert np.isclose(scalar_reward, 4.0)  # Should equal last objective

    # Test with mixed weights
    agent.set_preference_weights(np.array([0.5, 0.3, 0.2, 0.0]))
    scalar_reward = agent.scalarize_reward(mo_reward)
    expected = 0.5 * 1.0 + 0.3 * 2.0 + 0.2 * 3.0 + 0.0 * 4.0  # = 0.5 + 0.6 + 0.6 = 1.7
    assert np.isclose(scalar_reward, expected)

    # Test error handling for wrong reward vector length
    with pytest.raises(ValueError):
        agent.scalarize_reward(np.array([1.0, 2.0]))  # Only 2 rewards instead of 4

    print("✅ Reward scalarization test passed!")


def test_mosac_with_real_mo_rewards(mo_wrapped_env):
    """Test MOSAC with actual multi-objective rewards from MOPCSWrapper."""

    agent = MOSAC(
        env=mo_wrapped_env,
        num_objectives=4,
        verbose=0,
        learning_starts=5,  # Very small for testing
        buffer_size=100
    )

    # Reset environment and get initial observation
    obs, info = mo_wrapped_env.reset()

    # Take a few steps and check that we can handle the MO rewards
    for step in range(3):
        # Get action from agent
        action, _ = agent.predict(obs, deterministic=True)

        # Step environment
        obs, reward, terminated, truncated, info = mo_wrapped_env.step(action)

        # Check that we got multi-objective rewards in info
        assert 'mo_rewards' in info
        mo_rewards = info['mo_rewards']

        # Verify MO reward structure
        assert isinstance(mo_rewards, np.ndarray)
        assert len(mo_rewards) == 4
        assert all(isinstance(r, (int, float)) for r in mo_rewards)

        # Test scalarization with these real rewards
        scalar_reward = agent.scalarize_reward(mo_rewards)
        assert isinstance(scalar_reward, (int, float))
        assert not np.isnan(scalar_reward)

        if terminated or truncated:
            break

    print("✅ Real multi-objective rewards test passed!")


def test_mosac_initialization_errors(mo_wrapped_env):
    """Test error handling in MOSAC initialization."""

    # Test with wrong number of preference weights
    with pytest.raises(AssertionError):
        MOSAC(
            env=mo_wrapped_env,
            num_objectives=4,
            preference_weights=np.array([0.5, 0.5]),  # Only 2 weights for 4 objectives
            verbose=0
        )

    # Test with negative number of objectives
    with pytest.raises(Exception):  # Should fail during SAC initialization or our validation
        MOSAC(
            env=mo_wrapped_env,
            num_objectives=0,
            verbose=0
        )

    print("✅ Error handling test passed!")


if __name__ == "__main__":
    # Run tests individually for debugging
    import logging

    logging.basicConfig(level=logging.WARNING)

    # Create fixtures manually
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

        wrapped_env = MOPCSWrapper(
            env,
            num_objectives=4,
            reward_weights=np.ones(4) / 4,
            normalize_rewards=False,
            log_level=logging.WARNING
        )

        print("Running MOSAC tests...")
        test_mosac_basic_initialization(wrapped_env)
        test_mosac_custom_initialization(wrapped_env)
        test_preference_weight_handling(wrapped_env)
        test_reward_scalarization(wrapped_env)
        test_mosac_with_real_mo_rewards(wrapped_env)
        test_mosac_initialization_errors(wrapped_env)

        print("\n🎉 All MOSAC tests passed!")

    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        print("Make sure EnergyNet and MOPCSWrapper are properly installed and configured.")