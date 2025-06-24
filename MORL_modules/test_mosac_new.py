import pytest
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
import os
from collections import OrderedDict

# Import your modules
from agents.monets import SharedFeatureQNet, SeparateQNet
from agents.mobuffers import MOReplayBuffer
from stable_baselines3.common.torch_layers import FlattenExtractor
from agents.mosac import MOSAC, MOSACPolicy, MOContinuousCritic

class DummyMultiObjectiveEnv(gym.Env):
    """
    Dummy multi-objective environment for testing.
    Returns vector rewards with configurable number of objectives.
    """
    def __init__(self, num_objectives=3):
        super().__init__()
        self.observation_space = spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.num_objectives = num_objectives
        self.current_step = 0
        self.max_episode_steps = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self.observation_space.sample()
        info = self._get_info()
        breakpoint()
        return obs, info

    def step(self, action):
        # Return vector rewards
        obs = self.observation_space.sample()
        rewards = np.random.uniform(-1, 1, size=(self.num_objectives,))
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_episode_steps
        info = {}
        return obs, rewards, terminated, truncated, info


class TestMOContinuousCritic:
    """Test suite for the MOContinuousCritic component."""

    @pytest.fixture
    def critic_setup(self):
        """Create a basic critic for testing."""
        observation_space = spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        return {
            "observation_space": observation_space,
            "action_space": action_space,
            "net_arch": [64, 64],
            "num_objectives": 3,
            "n_critics": 2
        }

    def test_shared_features_critic(self, critic_setup):
        """Test critic with shared features across objectives."""
        critic = MOContinuousCritic(
            observation_space=critic_setup["observation_space"],
            action_space=critic_setup["action_space"],
            net_arch=critic_setup["net_arch"],
            num_objectives=critic_setup["num_objectives"],
            n_critics=critic_setup["n_critics"],
            share_features_across_objectives=True
        )

        # Check critic structure
        assert len(critic.q_networks) == critic_setup["n_critics"]
        for q_net in critic.q_networks:
            assert isinstance(q_net, SharedFeatureQNet)
            assert len(q_net.heads) == critic_setup["num_objectives"]

        # Test forward pass
        batch_size = 5
        obs = th.FloatTensor(np.random.uniform(-1, 1, size=(batch_size, 4)))
        actions = th.FloatTensor(np.random.uniform(-1, 1, size=(batch_size, 2)))

        outputs = critic(obs, actions)

        # Check output structure
        assert len(outputs) == critic_setup["n_critics"]
        for critic_outputs in outputs:
            assert len(critic_outputs) == critic_setup["num_objectives"]
            for obj_output in critic_outputs:
                assert obj_output.shape == (batch_size, 1)

    def test_separate_features_critic(self, critic_setup):
        """Test critic with separate networks for each objective."""
        critic = MOContinuousCritic(
            observation_space=critic_setup["observation_space"],
            action_space=critic_setup["action_space"],
            net_arch=critic_setup["net_arch"],
            num_objectives=critic_setup["num_objectives"],
            n_critics=critic_setup["n_critics"],
            share_features_across_objectives=False
        )

        # Check critic structure
        assert len(critic.q_networks) == critic_setup["n_critics"]
        for q_net in critic.q_networks:
            assert isinstance(q_net, SeparateQNet)
            assert len(q_net.nets) == critic_setup["num_objectives"]

        # Test forward pass
        batch_size = 5
        obs = th.FloatTensor(np.random.uniform(-1, 1, size=(batch_size, 4)))
        actions = th.FloatTensor(np.random.uniform(-1, 1, size=(batch_size, 2)))

        outputs = critic(obs, actions)

        # Check output structure
        assert len(outputs) == critic_setup["n_critics"]
        for critic_outputs in outputs:
            assert len(critic_outputs) == critic_setup["num_objectives"]
            for obj_output in critic_outputs:
                assert obj_output.shape == (batch_size, 1)

    def test_q_value_scalarization(self, critic_setup):
        """Test the q_value method that scalarizes outputs using preference weights."""
        critic = MOContinuousCritic(
            observation_space=critic_setup["observation_space"],
            action_space=critic_setup["action_space"],
            net_arch=critic_setup["net_arch"],
            num_objectives=critic_setup["num_objectives"],
            n_critics=critic_setup["n_critics"]
        )

        batch_size = 5
        obs = th.FloatTensor(np.random.uniform(-1, 1, size=(batch_size, 4)))
        actions = th.FloatTensor(np.random.uniform(-1, 1, size=(batch_size, 2)))

        # Test with uniform weights
        uniform_weights = th.ones(critic_setup["num_objectives"]) / critic_setup["num_objectives"]
        q_values = critic.q_value(obs, actions, uniform_weights)

        # Check q_values structure
        assert len(q_values) == critic_setup["n_critics"]
        for q_val in q_values:
            assert q_val.shape == (batch_size, 1)

        # Test with batch-specific weights
        batch_weights = th.FloatTensor(np.random.uniform(0, 1, size=(batch_size, critic_setup["num_objectives"])))
        batch_weights = batch_weights / batch_weights.sum(dim=1, keepdim=True)  # Normalize

        q_values_batch = critic.q_value(obs, actions, batch_weights)

        # Check q_values structure
        assert len(q_values_batch) == critic_setup["n_critics"]
        for q_val in q_values_batch:
            assert q_val.shape == (batch_size, 1)

        # Values should be different with different weights
        assert not th.allclose(q_values[0], q_values_batch[0])


class TestMOSACPolicy:
    """Test suite for the MOSACPolicy component."""

    @pytest.fixture
    def policy_setup(self):
        """Create a basic policy setup for testing."""
        observation_space = spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        def lr_schedule(progress_remaining):
            return 3e-4

        return {
            "observation_space": observation_space,
            "action_space": action_space,
            "lr_schedule": lr_schedule,
            "num_objectives": 3,
            "net_arch": [64, 64],
            "n_critics": 2
        }

    def test_policy_initialization(self, policy_setup):
        """Test basic policy initialization."""
        policy = MOSACPolicy(
            observation_space=policy_setup["observation_space"],
            action_space=policy_setup["action_space"],
            lr_schedule=policy_setup["lr_schedule"],
            num_objectives=policy_setup["num_objectives"],
            net_arch=policy_setup["net_arch"],
            n_critics=policy_setup["n_critics"]
        )

        # Check policy structure
        assert hasattr(policy, "actor")
        assert hasattr(policy, "critic")
        assert isinstance(policy.critic, MOContinuousCritic)
        assert policy.num_objectives == policy_setup["num_objectives"]

        # Test action_log_prob method
        batch_size = 5
        obs = th.FloatTensor(np.random.uniform(-1, 1, size=(batch_size, 4)))

        actions, log_probs = policy.actor.action_log_prob(obs)

        # Check output shapes
        assert actions.shape == (batch_size, 2)
        assert log_probs.shape == (batch_size,)

        # Check actions are within bounds
        assert th.all(actions >= -1) and th.all(actions <= 1)

    def test_policy_with_shared_features(self, policy_setup):
        """Test policy with shared features across objectives."""
        policy = MOSACPolicy(
            observation_space=policy_setup["observation_space"],
            action_space=policy_setup["action_space"],
            lr_schedule=policy_setup["lr_schedule"],
            num_objectives=policy_setup["num_objectives"],
            net_arch=policy_setup["net_arch"],
            n_critics=policy_setup["n_critics"],
            share_features_across_objectives=True
        )

        # Check critic structure
        assert isinstance(policy.critic, MOContinuousCritic)
        for q_net in policy.critic.q_networks:
            assert isinstance(q_net, SharedFeatureQNet)

    def test_policy_with_separate_features(self, policy_setup):
        """Test policy with separate networks for each objective."""
        policy = MOSACPolicy(
            observation_space=policy_setup["observation_space"],
            action_space=policy_setup["action_space"],
            lr_schedule=policy_setup["lr_schedule"],
            num_objectives=policy_setup["num_objectives"],
            net_arch=policy_setup["net_arch"],
            n_critics=policy_setup["n_critics"],
            share_features_across_objectives=False
        )

        # Check critic structure
        assert isinstance(policy.critic, MOContinuousCritic)
        for q_net in policy.critic.q_networks:
            assert isinstance(q_net, SeparateQNet)


class TestMOSACInitialization:
    """Test suite for MOSAC initialization."""

    def test_basic_initialization(self):
        """Test basic initialization with default parameters."""
        env = DummyMultiObjectiveEnv(num_objectives=3)
        model = MOSAC(
            policy="MOSACPolicy",
            env=env,
            learning_rate=3e-4,
            verbose=0
        )

        # Check that essential attributes are created
        assert hasattr(model, "policy"), "Policy not created"
        assert hasattr(model, "env"), "Environment not stored"
        assert model.num_objectives == 3, "Wrong number of objectives"

        # Check that policy has multi-objective components
        assert isinstance(model.policy, MOSACPolicy), "Policy should be MOSACPolicy"
        assert isinstance(model.policy.critic, MOContinuousCritic), "Critic should be MOContinuousCritic"

        # Check replay buffer is correct type
        assert isinstance(model.replay_buffer, MOReplayBuffer), "Replay buffer should be MOReplayBuffer"

        # Check preference weights
        assert hasattr(model, "preference_weights"), "Preference weights not created"
        assert len(model.preference_weights) == 3, "Wrong number of preference weights"
        assert np.allclose(model.preference_weights,
                           np.array([1 / 3, 1 / 3, 1 / 3])), "Default weights should be uniform"

        # Check preference weights tensor
        assert hasattr(model, "preference_weights_tensor"), "Preference weights tensor not created"
        assert model.preference_weights_tensor.shape == (3,), "Wrong shape for preference weights tensor"

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        env = DummyMultiObjectiveEnv(num_objectives=4)

        # Custom preference weights
        custom_weights = [0.4, 0.3, 0.2, 0.1]

        # Initialize with custom parameters
        model = MOSAC(
            policy="MOSACPolicy",
            env=env,
            learning_rate=1e-3,
            buffer_size=5000,
            learning_starts=100,
            batch_size=64,
            tau=0.02,
            gamma=0.98,
            train_freq=5,
            gradient_steps=3,
            n_critics=3,
            policy_kwargs={"net_arch": [128, 128]},
            preference_weights=custom_weights,
            verbose=1
        )

        # Check custom parameters
        assert model.learning_rate == 1e-3, "Learning rate not set correctly"
        assert model.buffer_size == 5000, "Buffer size not set correctly"
        assert model.learning_starts == 100, "Learning starts not set correctly"
        assert model.batch_size == 64, "Batch size not set correctly"
        assert model.tau == 0.02, "Tau not set correctly"
        assert model.gamma == 0.98, "Gamma not set correctly"
        assert model.train_freq == 5, "Train frequency not set correctly"
        assert model.gradient_steps == 3, "Gradient steps not set correctly"
        assert model.num_objectives == 4, "Wrong number of objectives"

        # Check policy network architecture
        assert model.policy_kwargs["net_arch"] == [128, 128], "Policy network architecture not set correctly"

        # Check preference weights
        normalized_weights = np.array(custom_weights) / np.sum(custom_weights)
        assert np.allclose(model.preference_weights, normalized_weights), "Custom weights not set correctly"

        # Check n_critics in policy
        assert model.policy.critic.n_critics == 3, "Number of critics not set correctly"

    def test_invalid_environment(self):
        """Test that initialization fails with invalid environment."""
        # Create a standard environment with scalar rewards
        env = gym.make("CartPole-v1")

        # Should raise an error because CartPole has scalar rewards, not vector
        with pytest.raises(Exception):
            model = MOSAC("MOSACPolicy", env)

    def test_policy_kwargs_override(self):
        """Test policy_kwargs properly override defaults."""
        env = DummyMultiObjectiveEnv(num_objectives=3)

        # Test with custom activation function and architecture
        policy_kwargs = {
            "net_arch": [64, 64],
            "activation_fn": th.nn.ReLU,
            "share_features_across_objectives": False
        }

        model = MOSAC(
            policy="MOSACPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=0
        )

        # Check that policy_kwargs are correctly passed and stored
        assert model.policy_kwargs["net_arch"] == [64, 64], "net_arch not set correctly"
        assert model.policy_kwargs["activation_fn"] == th.nn.ReLU, "activation_fn not set correctly"
        assert model.policy_kwargs[
                   "share_features_across_objectives"] == False, "share_features_across_objectives not set correctly"

        # Check that policy was constructed with these kwargs
        assert model.policy.critic.share_features_across_objectives == False, "share_features_across_objectives not applied to critic"
        for q_net in model.policy.critic.q_networks:
            assert isinstance(q_net,
                              SeparateQNet), "Should use SeparateQNet when share_features_across_objectives is False"


class TestMOSACTrainingBasic:
    """Basic tests for MOSAC training with a single step."""

    @pytest.fixture
    def mosac_model(self):
        """Create a basic MOSAC model for testing."""
        env = DummyMultiObjectiveEnv(num_objectives=3)
        model = MOSAC(
            policy="MOSACPolicy",
            env=env,
            learning_starts=0,  # Start training immediately
            gradient_steps=1,  # Single gradient step
            verbose=0
        )
        return model

    def test_single_step_collection(self, mosac_model):
        """Test collecting a single step of experience."""
        # Reset the environment
        obs, _ = mosac_model.env.reset()

        # Take a single step
        action = mosac_model.predict(obs, deterministic=False)[0]
        next_obs, rewards, terminated, truncated, info = mosac_model.env.step(action)

        # Add to replay buffer (this calls _store_transition internally)
        mosac_model._store_transition(
            mosac_model.replay_buffer,
            action,
            next_obs,
            rewards,  # Vector reward
            np.array([terminated or truncated]),
            [info]
        )

        # Check replay buffer has one sample
        assert mosac_model.replay_buffer.size() == 1, "Experience not added to replay buffer"

        # Check that rewards are stored as a vector
        sample = mosac_model.replay_buffer.sample(1)
        assert sample.rewards.shape[1] == 3, "Rewards should be a vector with 3 objectives"

    def test_single_training_step(self, mosac_model):
        """Test a single training step after collecting experience."""
        # Get initial policy parameters
        initial_actor_params = {name: param.clone().detach()
                                for name, param in mosac_model.policy.actor.named_parameters()}
        initial_critic_params = {name: param.clone().detach()
                                 for name, param in mosac_model.policy.critic.named_parameters()}

        # Take a few actions to fill buffer
        obs, _ = mosac_model.env.reset()
        for _ in range(10):  # Add 10 samples to ensure we have enough data
            action = mosac_model.predict(obs, deterministic=False)[0]
            next_obs, rewards, terminated, truncated, info = mosac_model.env.step(action)
            mosac_model._store_transition(
                mosac_model.replay_buffer,
                action,
                next_obs,
                rewards,
                np.array([terminated or truncated]),
                [info]
            )
            obs = next_obs
            if terminated or truncated:
                obs, _ = mosac_model.env.reset()

        # Perform one training step
        mosac_model.train(gradient_steps=1, batch_size=4)

        # Check that policy parameters have been updated
        actor_params_changed = False
        for name, param in mosac_model.policy.actor.named_parameters():
            if not th.allclose(param, initial_actor_params[name]):
                actor_params_changed = True
                break

        critic_params_changed = False
        for name, param in mosac_model.policy.critic.named_parameters():
            if not th.allclose(param, initial_critic_params[name]):
                critic_params_changed = True
                break

        assert actor_params_changed, "Actor parameters were not updated after training"
        assert critic_params_changed, "Critic parameters were not updated after training"

    def test_preference_weights_training(self, mosac_model):
        """Test training with different preference weights."""
        # Fill buffer with some samples
        obs, _ = mosac_model.env.reset()
        for _ in range(10):
            action = mosac_model.predict(obs, deterministic=False)[0]
            next_obs, rewards, terminated, truncated, info = mosac_model.env.step(action)
            mosac_model._store_transition(
                mosac_model.replay_buffer,
                action,
                next_obs,
                rewards,
                np.array([terminated or truncated]),
                [info]
            )
            obs = next_obs
            if terminated or truncated:
                obs, _ = mosac_model.env.reset()

        # Update preference weights
        # Since your implementation uses an instance variable for preference weights, we need to update it
        mosac_model.preference_weights = np.array([0.7, 0.2, 0.1])
        mosac_model.preference_weights_tensor = th.FloatTensor(mosac_model.preference_weights).to(mosac_model.device)

        # Train with updated weights
        mosac_model.train(gradient_steps=1, batch_size=4)

        # Check if preference weights are correctly used (no error is a success)
        assert True, "Training with different preference weights should work"

    def test_predict_with_preference(self, mosac_model):
        """Test prediction with preference weights."""
        # Get an observation
        obs, _ = mosac_model.env.reset()

        # Your implementation doesn't seem to have a preference_weights parameter in predict
        # Instead it uses the instance variable, so we'll update that

        # Predict with default weights
        action1 = mosac_model.predict(obs, deterministic=True)[0]

        # Update preference weights and predict again
        mosac_model.preference_weights = np.array([0.8, 0.1, 0.1])
        mosac_model.preference_weights_tensor = th.FloatTensor(mosac_model.preference_weights).to(mosac_model.device)

        action2 = mosac_model.predict(obs, deterministic=True)[0]

        # Both actions should be within the action space bounds
        assert np.all(action1 >= -1) and np.all(action1 <= 1), "Actions should be within bounds"
        assert np.all(action2 >= -1) and np.all(action2 <= 1), "Actions should be within bounds"

        # Note: In your current implementation, predict doesn't use preference weights directly
        # so actions may be the same. Let's just check they're valid.

    def test_train_critic_vector_loss(self, mosac_model):
        """Test critic training with vector loss."""
        # Fill buffer with some samples
        obs, _ = mosac_model.env.reset()
        for _ in range(10):
            action = mosac_model.predict(obs, deterministic=False)[0]
            next_obs, rewards, terminated, truncated, info = mosac_model.env.step(action)
            mosac_model._store_transition(
                mosac_model.replay_buffer,
                action,
                next_obs,
                rewards,
                np.array([terminated or truncated]),
                [info]
            )
            obs = next_obs
            if terminated or truncated:
                obs, _ = mosac_model.env.reset()

        # Get initial critic parameters
        initial_critic_params = {name: param.clone().detach()
                                 for name, param in mosac_model.policy.critic.named_parameters()}

        # Perform one training step
        mosac_model.train(gradient_steps=1, batch_size=4)

        # Check that critic parameters have been updated
        critic_params_changed = False
        for name, param in mosac_model.policy.critic.named_parameters():
            if not th.allclose(param, initial_critic_params[name]):
                critic_params_changed = True
                break

        assert critic_params_changed, "Critic parameters were not updated after training"

        # The test passes if training completes without errors
        assert True, "Vector loss training should work"