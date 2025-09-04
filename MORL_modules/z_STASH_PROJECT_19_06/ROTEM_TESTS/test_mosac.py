import os
import numpy as np
import torch as th
import pytest
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

# Import your MOSAC implementation
from algorithms.mosac import MOSAC, MOSACPolicy, MOContinuousCritic, MOReplayBuffer, register_mosac


# Create a simple multi-objective environment for testing
class SimpleMOEnv(gym.Env):
    """
    Simple multi-objective environment for testing MOSAC.

    This environment has:
    - A continuous observation space (2D)
    - A continuous action space (1D)
    - Two objectives:
        1. Get close to the target
        2. Minimize action magnitude
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.target_pos = np.array([0.5, 0.5], dtype=np.float32)
        self.current_pos = np.zeros(2, dtype=np.float32)
        self.num_objectives = 2
        self.max_steps = 50
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_pos = self.np_random.uniform(-0.5, 0.5, size=2).astype(np.float32)
        self.steps = 0
        return self.current_pos, {}

    def step(self, action):
        self.steps += 1
        breakpoint()
        # Update position based on action
        self.current_pos = self.current_pos + 0.1 * np.clip(action, -1, 1)
        self.current_pos = np.clip(self.current_pos, -1, 1)

        # Calculate rewards for each objective
        dist_to_target = np.linalg.norm(self.current_pos - self.target_pos)
        action_magnitude = np.linalg.norm(action)

        # Multi-objective rewards
        reward = np.array([-dist_to_target, -action_magnitude], dtype=np.float32)

        # Check if done
        done = self.steps >= self.max_steps

        # Info dictionary
        info = {
            "distance": dist_to_target,
            "action_magnitude": action_magnitude
        }

        return self.current_pos, reward, done, False, info


class TestMOSAC:

    @pytest.fixture
    def env(self):
        """Create test environment"""
        return SimpleMOEnv()

    @pytest.fixture
    def vec_env(self, env):
        """Create vectorized environment"""

        def make_env():
            return SimpleMOEnv()

        return DummyVecEnv([make_env])

    def test_mo_continuous_critic(self, env):
        """Test the multi-objective critic initialization and forward pass"""
        observation_space = env.observation_space
        action_space = env.action_space
        num_objectives = 2
        net_arch = [64, 64]

        # Initialize critic
        critic = MOContinuousCritic(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            num_objectives=num_objectives
        )

        # Create dummy batch of observations and actions
        batch_size = 4
        obs = th.randn(batch_size, *observation_space.shape)
        actions = th.randn(batch_size, *action_space.shape)

        # Test forward pass
        q_values = critic.forward(obs, actions)

        # Check shape and structure of output
        assert isinstance(q_values, list)
        assert len(q_values) == critic.n_critics  # Default is 2 critics

        # For each critic ensemble
        for critic_ensemble_values in q_values:
            assert isinstance(critic_ensemble_values, list)
            assert len(critic_ensemble_values) == num_objectives

            # For each objective
            for obj_q_values in critic_ensemble_values:
                assert obj_q_values.shape == (batch_size, 1)

        # Test scalarized Q-value computation
        preference_weights = th.tensor([[0.7, 0.3]] * batch_size)
        scalarized_q_values = critic.q_value(obs, actions, preference_weights)

        assert isinstance(scalarized_q_values, list)
        assert len(scalarized_q_values) == critic.n_critics
        for q_val in scalarized_q_values:
            assert q_val.shape == (batch_size, 1)

    def test_mo_replay_buffer(self, env):
        """Test the multi-objective replay buffer"""
        buffer_size = 10
        num_objectives = 2

        # Initialize buffer
        buffer = MOReplayBuffer(
            buffer_size=buffer_size,
            observation_space=env.observation_space,
            action_space=env.action_space,
            num_objectives=num_objectives
        )

        # Generate sample data
        obs = env.observation_space.sample()
        next_obs = env.observation_space.sample()
        action = env.action_space.sample()
        reward = np.array([0.5, -0.3], dtype=np.float32)  # Two objectives
        done = False
        info = [{}]

        # Add transition
        buffer.add(obs, next_obs, action, reward, done, info)
        assert buffer.pos == 1
        assert not buffer.full
        assert buffer.rewards[0, 0].shape == (num_objectives,)
        assert np.array_equal(buffer.rewards[0, 0], reward)

        # Fill buffer and check
        for _ in range(buffer_size - 1):
            buffer.add(obs, next_obs, action, reward, done, info)

        assert buffer.full
        assert buffer.pos == 0

        # Test sampling
        batch_size = 4
        samples = buffer.sample(batch_size)

        assert samples.rewards.shape == (batch_size, num_objectives)
        assert samples.observations.shape == (batch_size, *env.observation_space.shape)
        assert samples.actions.shape == (batch_size, *env.action_space.shape)
        assert samples.next_observations.shape == (batch_size, *env.observation_space.shape)
        assert samples.dones.shape == (batch_size, 1)

    def test_mo_sac_policy(self, env):
        """Test the multi-objective SAC policy"""
        num_objectives = 2
        net_arch = [64, 64]

        # Initialize policy
        policy = MOSACPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule= lambda _: 3e-4,
            num_objectives=num_objectives,
            net_arch=net_arch
        )

        # Check critic structure
        assert isinstance(policy.critic, MOContinuousCritic)
        assert policy.critic.num_objectives == num_objectives

        # Create dummy batch of observations
        batch_size = 4
        obs = th.FloatTensor(np.random.random((batch_size, *env.observation_space.shape)))

        # Test action prediction
        with th.no_grad():
            actions, log_probs = policy.actor.action_log_prob(obs)

        assert actions.shape == (batch_size, *env.action_space.shape)
        assert log_probs.shape == (batch_size,)

        # Test critic prediction
        with th.no_grad():
            q_values = policy.critic(obs, actions)
        assert isinstance(q_values, list)
        assert len(q_values) == policy.critic.n_critics
        for q_val in q_values:
            assert np.array(q_val).shape == (num_objectives, batch_size, 1)

    def test_mosac_init(self, vec_env):
        """Test MOSAC initialization"""
        num_objectives = 2
        preference_weights = [0.7, 0.3]

        # Initialize MOSAC
        model = MOSAC(
            policy=MOSACPolicy,
            env=vec_env,
            learning_rate=1e-3,
            buffer_size=100,
            learning_starts=50,
            num_objectives=num_objectives,
            preference_weights=preference_weights
        )

        # Check model properties
        assert model.num_objectives == num_objectives
        assert np.allclose(model.preference_weights, np.array(preference_weights) / np.sum(preference_weights))
        assert isinstance(model.replay_buffer, MOReplayBuffer)
        assert model.replay_buffer.num_objectives == num_objectives
        assert isinstance(model.policy, MOSACPolicy)

    def test_mosac_learn(self, vec_env):
        """Test that MOSAC can learn"""
        num_objectives = 2
        model = MOSAC(
            policy=MOSACPolicy,
            env=vec_env,
            learning_rate=1e-3,
            buffer_size=1000,
            batch_size=16,
            learning_starts=100,
            num_objectives=num_objectives,
            verbose=0
        )

        # Train for a small number of steps
        model.learn(total_timesteps=200, log_interval=100)

        # Check that the model has been trained
        assert model._n_updates > 0

    def test_mosac_prediction(self, env):
        """Test MOSAC prediction"""
        num_objectives = 2
        model = MOSAC(
            policy=MOSACPolicy,
            env=DummyVecEnv([lambda: SimpleMOEnv()]),
            learning_rate=1e-3,
            buffer_size=100,
            learning_starts=0,
            num_objectives=num_objectives,
            verbose=0
        )

        # Test prediction
        obs = env.reset()[0]
        action, _states = model.predict(obs, deterministic=True)

        assert action.shape == env.action_space.shape

    def test_register_mosac(self):
        """Test that register_mosac adds MOSAC to RL Zoo"""
        try:
            from rl_zoo3 import ALGOS
            # Save original ALGOS dict
            original_algos = ALGOS.copy()

            # Register MOSAC
            register_mosac()

            # Check that MOSAC is in ALGOS
            assert "mosac" in ALGOS
            assert ALGOS["mosac"] == MOSAC

            # Restore original ALGOS
            ALGOS.clear()
            ALGOS.update(original_algos)
        except ImportError:
            pytest.skip("rl_zoo3 not installed")

    @pytest.mark.parametrize("share_features_across_objectives", [True, False])
    def test_critic_architecture_variants(self, env, share_features_across_objectives):
        """Test different critic architecture variants"""
        num_objectives = 2
        net_arch = [32, 32]

        critic = MOContinuousCritic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            net_arch=net_arch,
            num_objectives=num_objectives,
            share_features_across_objectives=share_features_across_objectives
        )

        batch_size = 4
        obs = th.randn(batch_size, *env.observation_space.shape)
        actions = th.randn(batch_size, *env.action_space.shape)

        # Test forward pass
        q_values = critic.forward(obs, actions)

        # Structure should be the same regardless of sharing
        assert isinstance(q_values, list)
        assert len(q_values) == critic.n_critics

        for critic_ensemble_values in q_values:
            assert isinstance(critic_ensemble_values, list)
            assert len(critic_ensemble_values) == num_objectives

            for obj_q_values in critic_ensemble_values:
                assert obj_q_values.shape == (batch_size, 1)


if __name__ == "__main__":
    # Run the tests
    pytest.main(["-xvs", "test_mosac.py"])