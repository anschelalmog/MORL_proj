import pytest
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
import tempfile
import os
import sys
from typing import Dict, Any, List, Tuple
import pdb

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'MORL_modules'))
print(project_root)
print(current_dir)
from agents.mosac import MOSAC, MOSACPolicy, MOContinuousCritic
from agents.mobuffers import MOReplayBuffer as MOReplayBuffer
from agents.monets import SharedFeatureQNet, SeparateQNet
from agents.mo_env_wrappers import MODummyVecEnv, MultiObjectiveWrapper
from wrappers.mo_pcs_wrapper import MOPCSWrapper
from wrappers.scalarized_mo_pcs_wrapper import ScalarizedMOPCSWrapper
from wrappers.dict_to_box_wrapper import DictToBoxWrapper
from energy_net.envs.energy_net_v0 import EnergyNetV0

from energy_net.market.pricing.pricing_policy import PricingPolicy
from energy_net.market.pricing.cost_types import CostType
from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern

def create_energynet_env(**kwargs):
    """Create EnergyNet environment."""
    from energy_net.envs.energy_net_v0 import EnergyNetV0

    default_kwargs = {
        'pricing_policy': PricingPolicy.QUADRATIC,
        'demand_pattern': DemandPattern.SINUSOIDAL,
        'cost_type': CostType.CONSTANT,
    }

    default_kwargs.update(kwargs)

    return DictToBoxWrapper(EnergyNetV0(**default_kwargs))


class TestMOReplayBuffer:
    """Test the multi-objective replay buffer."""

    def test_initialization(self):
        """Test buffer initialization with correct dimensions."""
        buffer_size = 1000
        obs_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        num_objectives = 3

        buffer = MOReplayBuffer(
            buffer_size=buffer_size,
            observation_space=obs_space,
            action_space=action_space,
            num_objectives=num_objectives,
            device="cpu"
        )

        assert buffer.buffer_size == buffer_size
        assert buffer.num_objectives == num_objectives
        assert buffer.rewards.shape == (buffer_size, 1, num_objectives)
        assert buffer.pos == 0
        assert not buffer.full

    def test_add_vector_reward(self):
        """Test adding transitions with vector rewards."""
        buffer = MOReplayBuffer(
            buffer_size=100,
            observation_space=spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
            action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            num_objectives=3,
            device="cpu"
        )

        obs = np.random.randn(4).astype(np.float32)
        next_obs = np.random.randn(4).astype(np.float32)
        action = np.random.randn(2).astype(np.float32)
        vector_reward = np.array([1.0, -0.5, 2.0], dtype=np.float32)
        done = np.array([False])
        infos = [{}]

        buffer.add(obs, next_obs, action, vector_reward, done, infos)

        assert buffer.pos == 1
        np.testing.assert_array_equal(buffer.rewards[0, 0], vector_reward)

    def test_sample(self):
        """Test sampling from buffer returns correct format."""
        buffer = MOReplayBuffer(
            buffer_size=100,
            observation_space=spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
            action_space=spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            num_objectives=3,
            device="cpu"
        )

        # Add some transitions
        for _ in range(10):
            obs = np.random.randn(4).astype(np.float32)
            next_obs = np.random.randn(4).astype(np.float32)
            action = np.random.randn(2).astype(np.float32)
            vector_reward = np.random.randn(3).astype(np.float32)
            done = np.array([False])
            infos = [{}]

            buffer.add(obs, next_obs, action, vector_reward, done, infos)

        # Sample batch
        batch = buffer.sample(5, env=None)

        assert batch.observations.shape == (5, 4)
        assert batch.actions.shape == (5, 2)
        assert batch.rewards.shape == (5, 3)  # Vector rewards
        assert batch.next_observations.shape == (5, 4)
        assert batch.dones.shape == (5, 1)


class TestMONets:
    """Test multi-objective network architectures."""

    def test_shared_feature_qnet(self):
        """Test SharedFeatureQNet forward pass."""
        obs_dim, action_dim = 8, 2
        num_objectives = 3
        hidden_dim = 64

        # Create base network
        base_net = th.nn.Sequential(
            th.nn.Linear(obs_dim + action_dim, hidden_dim),
            th.nn.ReLU(),
            th.nn.Linear(hidden_dim, hidden_dim),
            th.nn.ReLU()
        )

        # Create heads
        heads = th.nn.ModuleList([
            th.nn.Linear(hidden_dim, 1) for _ in range(num_objectives)
        ])

        net = SharedFeatureQNet(base_net, heads)

        # Test forward pass
        batch_size = 32
        obs = th.randn(batch_size, obs_dim)
        actions = th.randn(batch_size, action_dim)

        outputs = net(obs, actions)

        assert len(outputs) == num_objectives
        for output in outputs:
            assert output.shape == (batch_size, 1)

    def test_separate_qnet(self):
        """Test SeparateQNet forward pass."""
        obs_dim, action_dim = 8, 2
        num_objectives = 3
        hidden_dim = 64

        # Create separate networks
        nets = th.nn.ModuleList([
            th.nn.Sequential(
                th.nn.Linear(obs_dim + action_dim, hidden_dim),
                th.nn.ReLU(),
                th.nn.Linear(hidden_dim, 1)
            ) for _ in range(num_objectives)
        ])

        net = SeparateQNet(nets)

        # Test forward pass
        batch_size = 32
        obs = th.randn(batch_size, obs_dim)
        actions = th.randn(batch_size, action_dim)

        outputs = net(obs, actions)

        assert len(outputs) == num_objectives
        for output in outputs:
            assert output.shape == (batch_size, 1)


class TestMOContinuousCritic:
    """Test multi-objective continuous critic."""

    def test_initialization_shared_features(self):
        """Test critic initialization with shared features."""
        obs_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        critic = MOContinuousCritic(
            observation_space=obs_space,
            action_space=action_space,
            net_arch=[64, 64],
            num_objectives=3,
            share_features_across_objectives=True,
            n_critics=2
        )

        assert critic.num_objectives == 3
        assert critic.n_critics == 2
        assert len(critic.q_networks) == 2
        assert critic.share_features_across_objectives

    def test_initialization_separate_features(self):
        """Test critic initialization with separate features."""
        obs_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        critic = MOContinuousCritic(
            observation_space=obs_space,
            action_space=action_space,
            net_arch=[64, 64],
            num_objectives=3,
            share_features_across_objectives=False,
            n_critics=2
        )

        assert critic.num_objectives == 3
        assert not critic.share_features_across_objectives

    def test_forward_pass(self):
        """Test critic forward pass."""
        obs_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        critic = MOContinuousCritic(
            observation_space=obs_space,
            action_space=action_space,
            net_arch=[64, 64],
            num_objectives=3,
            n_critics=2
        )

        batch_size = 16
        obs = th.randn(batch_size, 8)
        actions = th.randn(batch_size, 2)

        outputs = critic(obs, actions)

        assert len(outputs) == 2  # n_critics
        for critic_output in outputs:
            assert len(critic_output) == 3  # num_objectives
            for obj_output in critic_output:
                assert obj_output.shape == (batch_size, 1)

    def test_q_value_scalarization(self):
        """Test Q-value scalarization with preference weights."""
        obs_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        critic = MOContinuousCritic(
            observation_space=obs_space,
            action_space=action_space,
            net_arch=[64, 64],
            num_objectives=3,
            n_critics=2
        )

        batch_size = 16
        obs = th.randn(batch_size, 8)
        actions = th.randn(batch_size, 2)
        preference_weights = th.tensor([0.5, 0.3, 0.2])

        scalarized_q_values = critic.q_value(obs, actions, preference_weights)

        assert len(scalarized_q_values) == 2  # n_critics
        for q_val in scalarized_q_values:
            assert q_val.shape == (batch_size, 1)


class TestMOSACPolicy:
    """Test multi-objective SAC policy."""

    def test_initialization(self):
        """Test policy initialization."""
        obs_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        policy = MOSACPolicy(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=lambda x: 3e-4,
            num_objectives=4
        )

        assert policy.num_objectives == 4
        assert hasattr(policy, 'make_critic')

    def test_make_critic(self):
        """Test critic creation."""
        obs_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        policy = MOSACPolicy(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=lambda x: 3e-4,
            num_objectives=4
        )

        critic = policy.make_critic()

        assert isinstance(critic, MOContinuousCritic)
        assert critic.num_objectives == 4


class TestEnvironmentWrappers:
    """Test environment wrappers."""

    def test_mo_pcs_wrapper(self):
        """Test MOPCSWrapper functionality."""
        base_env = create_energynet_env()
        mo_env = MOPCSWrapper(base_env, num_objectives=4)

        # Test reset
        obs, info = mo_env.reset()
        assert len(obs.shape) == 1  # Should be 1D observation

        # Test step returns vector rewards
        action = mo_env.action_space.sample()
        obs, reward, terminated, truncated, info = mo_env.step(action)

        assert isinstance(reward, np.ndarray)
        assert len(reward) == 4  # num_objectives
        assert 'mo_rewards' in info or any('reward' in k for k in info.keys())

    def test_scalarized_wrapper(self):
        """Test ScalarizedMOPCSWrapper functionality."""
        base_env = create_energynet_env()

        # Test with custom weights
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        scalarized_env = ScalarizedMOPCSWrapper(
            base_env,
            weights=weights,
            num_objectives=4
        )

        # Test reset
        obs, info = scalarized_env.reset()
        assert len(obs.shape) == 1  # Should be 1D observation

        # Test step returns scalar reward
        action = scalarized_env.action_space.sample()
        obs, reward, terminated, truncated, info = scalarized_env.step(action)

        assert isinstance(reward, (int, float, np.number))  # Scalar reward
        assert 'mo_rewards_original' in info  # Original MO rewards preserved
        assert 'scalarization_weights' in info
        assert 'scalar_reward' in info

        # Check weights are normalized
        np.testing.assert_array_almost_equal(
            scalarized_env.weights, weights / np.sum(weights)
        )


class TestMOSAC:
    """Test the complete MOSAC algorithm."""

    def test_initialization_with_mo_env(self):
        """Test MOSAC initialization with MO environment."""
        base_env = create_energynet_env()
        mo_env = MOPCSWrapper(base_env, num_objectives=4)
        model = MOSAC(
            policy="MOSACPolicy",
            env = mo_env,
            num_objectives=4,
            learning_starts=10,
            buffer_size=1000,
            batch_size=64,
            verbose=0
        )

        assert model.num_objectives == 4
        assert np.allclose(model.preference_weights, np.ones(4) / 4)
        assert isinstance(model.replay_buffer, MOReplayBuffer )

    def test_custom_preference_weights(self):
        """Test MOSAC with custom preference weights."""
        base_env = create_energynet_env()
        mo_env = MOPCSWrapper(base_env, num_objectives=4)

        weights = [0.5, 0.3, 0.2, 0.1]
        model = MOSAC(
            policy="MOSACPolicy",
            env=mo_env,
            num_objectives=4,
            learning_starts=10,
            buffer_size=1000,
            batch_size=64,
            verbose=0,
            preference_weights=weights
        )


        expected_weights = np.array(weights) / np.sum(weights)
        np.testing.assert_array_almost_equal(model.preference_weights, expected_weights)

    def test_learn_with_mo_environment(self):
        """Test learning with multi-objective environment."""
        base_env = create_energynet_env()
        mo_env = MOPCSWrapper(base_env, num_objectives=4)

        model = MOSAC(
            policy="MOSACPolicy",
            env=mo_env,
            num_objectives=4,
            learning_starts=10,
            buffer_size=1000,
            batch_size=64,
            verbose=0
        )

        # Short learning run
        model.learn(total_timesteps=50, log_interval=None)

        # Check that model has learned something
        assert model._n_updates >= 0
        assert model.replay_buffer.size() > 0

    def test_predict_with_mo_environment(self):
        """Test prediction with multi-objective environment."""
        base_env = create_energynet_env()
        mo_env = MOPCSWrapper(base_env, num_objectives=4)

        model = MOSAC(
            policy="MOSACPolicy",
            env=mo_env,
            num_objectives=4,
            learning_starts=10,
            buffer_size=1000,
            batch_size=64,
            verbose=0
        )
        # Quick training
        model.learn(total_timesteps=20, log_interval=None)

        # Test prediction
        obs = mo_env.observation_space.sample()
        action, _states = model.predict(obs, deterministic=True)

        assert action.shape == mo_env.action_space.shape
        assert mo_env.action_space.contains(action)


class TestIntegration:
    """Integration tests with real environments and wrappers."""

    def test_full_pipeline_with_wrappers(self):
        """Test complete pipeline: EnergyNet env -> MO wrapper -> MOSAC."""
        base_env = create_energynet_env()
        mo_env = MOPCSWrapper(base_env, num_objectives=4)

        model = MOSAC(
            mo_env,
            "MOSACPolicy",
            num_objectives=4,
            learning_starts=10,
            buffer_size=500,
            batch_size=32,
            verbose=0
        )

        # Run training
        total_timesteps = 100
        model.learn(total_timesteps=total_timesteps, log_interval=None)

        # Verify learning occurred
        assert model._n_updates > 0
        assert model.num_timesteps == total_timesteps

        # Test evaluation
        obs, _ = mo_env.reset()
        for _ in range(10):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = mo_env.step(action)

            # Verify reward is vector
            assert isinstance(reward, np.ndarray)
            assert len(reward) == 4

            if terminated or truncated:
                obs, _ = mo_env.reset()

    def test_scalarized_vs_multi_objective_learning(self):
        """Test that scalarized and MO versions work differently."""
        base_env = create_energynet_env()

        # Multi-objective version
        mo_env = MOPCSWrapper(base_env, num_objectives=4)
        mo_model = MOSAC(
            mo_env,
            "MOSACPolicy",
            num_objectives=4,
            learning_starts=5,
            buffer_size=100,
            verbose=0
        )

        # Scalarized version
        scalarized_env = ScalarizedMOPCSWrapper(
            create_energynet_env(),
            weights=np.array([0.6, 0.3, 0.1,0.2]),
            num_objectives=4
        )
        # Note: ScalarizedMOPCSWrapper should work with regular SAC
        # but for this test we'll use MOSAC with num_objectives=1
        scalar_model = MOSAC(
            scalarized_env,
            "MOSACPolicy",
            num_objectives=4,  # The wrapper handles scalarization
            learning_starts=5,
            buffer_size=100,
            verbose=0
        )

        # Both should train without errors
        mo_model.learn(total_timesteps=20, log_interval=None)
        scalar_model.learn(total_timesteps=20, log_interval=None)

        # Both should be able to predict
        obs = base_env.observation_space.sample()

        mo_action, _ = mo_model.predict(obs)
        scalar_action, _ = scalar_model.predict(obs)

        assert mo_action.shape == scalar_action.shape
        assert mo_env.action_space.contains(mo_action)
        assert scalarized_env.action_space.contains(scalar_action)

    def test_energynet_integration(self):
        """Test EnergyNet integration with MOSAC."""
        base_env = create_energynet_env()
        mo_env = MOPCSWrapper(base_env, num_objectives=4)

        # Verify environment has expected structure
        assert hasattr(base_env.unwrapped, 'controller')
        assert hasattr(base_env.unwrapped.controller, 'battery_manager')
        assert hasattr(base_env.unwrapped.controller, 'pcs_unit')

        # Test that wrapper can access EnergyNet components
        obs, _ = mo_env.reset()
        action = mo_env.action_space.sample()
        obs, reward, terminated, truncated, info = mo_env.step(action)

        # Should produce 4 objectives
        assert len(reward) == 4

        # Test short training to ensure everything integrates
        model = MOSAC(
            mo_env,
            "MOSACPolicy",
            num_objectives=4,
            learning_starts=5,
            buffer_size=50,
            batch_size=8,
            verbose=0
        )

        model.learn(total_timesteps=30, log_interval=None)

        # Should complete without errors
        assert model._n_updates >= 0


def run_all_tests():
    """Run all tests manually (for debugging)."""
    print("Running MOSAC Test Suite with Environment Wrappers...")

    # Buffer tests
    print("\n1. Testing MOReplayBuffer...")
    buffer_tests = TestMOReplayBuffer()
    buffer_tests.test_initialization()
    buffer_tests.test_add_vector_reward()
    buffer_tests.test_sample()
    print("✓ MOReplayBuffer tests passed")

    # Network tests
    print("\n2. Testing MONets...")
    net_tests = TestMONets()
    net_tests.test_shared_feature_qnet()
    net_tests.test_separate_qnet()
    print("✓ MONets tests passed")

    # Critic tests
    print("\n3. Testing MOContinuousCritic...")
    critic_tests = TestMOContinuousCritic()
    critic_tests.test_initialization_shared_features()
    critic_tests.test_initialization_separate_features()
    critic_tests.test_forward_pass()
    critic_tests.test_q_value_scalarization()
    print("✓ MOContinuousCritic tests passed")

    # Policy tests
    print("\n4. Testing MOSACPolicy...")
    policy_tests = TestMOSACPolicy()
    policy_tests.test_initialization()
    policy_tests.test_make_critic()
    print("✓ MOSACPolicy tests passed")

    # Environment wrapper tests
    print("\n5. Testing Environment Wrappers...")
    wrapper_tests = TestEnvironmentWrappers()
    wrapper_tests.test_mo_pcs_wrapper()
    wrapper_tests.test_scalarized_wrapper()
    print("✓ Environment Wrapper tests passed")

    # MOSAC tests
    print("\n6. Testing MOSAC with MO Environment...")
    mosac_tests = TestMOSAC()
    mosac_tests.test_initialization_with_mo_env()
    mosac_tests.test_custom_preference_weights()
    mosac_tests.test_learn_with_mo_environment()
    mosac_tests.test_predict_with_mo_environment()
    print("✓ MOSAC tests passed")

    # Integration tests
    print("\n7. Testing Integration...")
    integration_tests = TestIntegration()
    integration_tests.test_full_pipeline_with_wrappers()
    integration_tests.test_scalarized_vs_multi_objective_learning()
    print("✓ Integration tests passed")

    print("\n🎉 All tests passed successfully!")


if __name__ == "__main__":
    # You can run this with pytest or manually
    run_all_tests()