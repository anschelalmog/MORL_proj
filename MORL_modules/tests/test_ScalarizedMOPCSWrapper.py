import pytest
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from MORL_modules.wrappers.scalarized_mo_pcs_wrapper import ScalarizedMOPCSWrapper
from MORL_modules.wrappers.mo_pcs_wrapper import MOPCSWrapper


class MockEnv(gym.Env):
    """Mock environment for testing."""

    def __init__(self):
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,))

    def reset(self, **kwargs):
        return np.zeros(4), {}

    def step(self, action):
        obs = np.random.rand(4)
        reward = {'pcs': np.random.randn()}
        terminated = False
        truncated = False
        info = {
            'net_exchange': np.random.randn(),
            'iso_buy_price': np.random.uniform(1, 5),
            'iso_sell_price': np.random.uniform(1, 5),
        }
        return obs, reward, terminated, truncated, info


@pytest.fixture
def mock_env():
    return MockEnv()


@pytest.fixture
def scalarized_wrapper(mock_env):
    return ScalarizedMOPCSWrapper(mock_env, normalize_rewards=False)


def test_initialization(mock_env):
    """Test wrapper initialization with different weight configurations."""
    # Default initialization (equal weights)
    wrapper = ScalarizedMOPCSWrapper(mock_env)
    assert np.allclose(wrapper.weights, [0.25, 0.25, 0.25, 0.25])

    # Custom weights (normalized)
    weights = [1, 2, 3, 4]
    wrapper = ScalarizedMOPCSWrapper(mock_env, weights=weights)
    expected = np.array(weights) / sum(weights)
    assert np.allclose(wrapper.weights, expected)

    # Custom weights (unnormalized)
    weights = [0.1, 0.2, 0.3, 0.4]
    wrapper = ScalarizedMOPCSWrapper(mock_env, weights=weights, normalize_weights=False)
    assert np.allclose(wrapper.weights, weights)


def test_initialization_errors(mock_env):
    """Test initialization with invalid weights."""
    # Wrong number of weights
    with pytest.raises(ValueError, match="Weight dimension"):
        ScalarizedMOPCSWrapper(mock_env, weights=[0.5, 0.5])

    # Zero weights
    with pytest.raises(ValueError, match="Weights sum to zero"):
        ScalarizedMOPCSWrapper(mock_env, weights=[0, 0, 0, 0])


def test_step_scalarization(scalarized_wrapper):
    """Test that rewards are properly scalarized."""
    obs, _ = scalarized_wrapper.reset()

    # Take a step
    action = scalarized_wrapper.action_space.sample()
    obs, reward, terminated, truncated, info = scalarized_wrapper.step(action)

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


def test_scalarization_computation(mock_env):
    """Test that scalarization is computed correctly."""
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    wrapper = ScalarizedMOPCSWrapper(mock_env, weights=weights, normalize_weights=False)

    # Directly access MO wrapper
    mo_wrapper = wrapper.env
    assert isinstance(mo_wrapper, MOPCSWrapper)

    # Mock MO rewards for testing
    mo_rewards = np.array([1.0, -0.5, 0.3, 0.8])
    expected_scalar = np.dot(weights, mo_rewards)

    # Test by mocking the step return
    obs, _ = wrapper.reset()

    # We need to mock the MO wrapper's step to return known rewards
    original_step = mo_wrapper.step
    mo_wrapper.step = lambda action: (obs, mo_rewards, False, False, {})

    _, scalar_reward, _, _, info = wrapper.step(0)

    assert np.isclose(scalar_reward, expected_scalar)
    assert np.allclose(info['mo_rewards_original'], mo_rewards)

    # Restore original step
    mo_wrapper.step = original_step


def test_set_weights(scalarized_wrapper):
    """Test updating weights after initialization."""
    # Set new weights
    new_weights = [0.1, 0.2, 0.3, 0.4]
    scalarized_wrapper.set_weights(new_weights)

    expected = np.array(new_weights) / sum(new_weights)
    assert np.allclose(scalarized_wrapper.weights, expected)

    # Set unnormalized weights
    scalarized_wrapper.set_weights(new_weights, normalize=False)
    assert np.allclose(scalarized_wrapper.weights, new_weights)

    # Test invalid weights
    with pytest.raises(ValueError):
        scalarized_wrapper.set_weights([0.5, 0.5])  # Wrong dimension

    with pytest.raises(ValueError):
        scalarized_wrapper.set_weights([0, 0, 0, 0])  # Zero sum


def test_wrapper_stacking(mock_env):
    """Test that wrapper correctly handles already-wrapped environments."""
    # First wrap with MO
    mo_wrapper = MOPCSWrapper(mock_env)

    # Then wrap with scalarized
    scalarized = ScalarizedMOPCSWrapper(mo_wrapper)

    # Should not double-wrap
    assert isinstance(scalarized.env, MOPCSWrapper)
    assert not isinstance(scalarized.env.env, MOPCSWrapper)


def test_integration_flow(scalarized_wrapper):
    """Test complete episode flow."""
    total_reward = 0
    episode_mo_rewards = []

    obs, _ = scalarized_wrapper.reset()

    for _ in range(10):
        action = scalarized_wrapper.action_space.sample()
        obs, reward, terminated, truncated, info = scalarized_wrapper.step(action)

        total_reward += reward
        episode_mo_rewards.append(info['mo_rewards_original'])

        if terminated or truncated:
            break

    # Verify we collected data
    assert len(episode_mo_rewards) > 0
    assert isinstance(total_reward, (float, np.floating))

    # Verify MO rewards structure
    mo_array = np.array(episode_mo_rewards)
    assert mo_array.shape[1] == 4  # 4 objectives


if __name__ == "__main__":
    pytest.main([__file__, "-v"])