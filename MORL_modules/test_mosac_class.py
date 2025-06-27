import pytest
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
from copy import deepcopy
import os
from collections import OrderedDict
import torch.nn.functional as F

# Mock the imports for your custom modules if needed
# Uncomment and modify paths as needed
# Import your modules
from agents.monets import SharedFeatureQNet, SeparateQNet
from agents.mobuffers import MOReplayBuffer
from stable_baselines3.common.torch_layers import FlattenExtractor
from agents.mosac import MOSAC, MOSACPolicy, MOContinuousCritic
from stable_baselines3.sac.policies import Actor
# Import your implementation
from agents.monets import SharedFeatureQNet, SeparateQNet
from agents.mobuffers import MOReplayBuffer
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
        return obs, {}

    def step(self, action):
        # Return vector rewards
        obs = self.observation_space.sample()
        rewards = np.random.uniform(-1, 1, size=(self.num_objectives,))
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_episode_steps
        info = {}
        return obs, rewards, terminated, truncated, info


class TestMOSACInitialization:
    """Comprehensive test suite for MOSAC initialization."""

    def test_basic_initialization(self):
        """Test that MOSAC initializes correctly with default parameters."""
        env = DummyMultiObjectiveEnv(num_objectives=3)
        model = MOSAC(policy="MOSACPolicy", env=env, num_objectives=3)

        
        # Check that essential attributes are correctly initialized
        assert model.num_objectives == 3, "Number of objectives not set correctly"
        assert isinstance(model.policy, MOSACPolicy), "Policy class incorrect"
        assert isinstance(model.critic, MOContinuousCritic), "Critic class incorrect"
        assert isinstance(model.actor, Actor), "Actor class incorrect"

        # Check preference weights
        expected_weights = np.array([1 / 3, 1 / 3, 1 / 3], np.float32)
        assert np.allclose(model.preference_weights, expected_weights), "Default preference weights incorrect"
        assert th.allclose(model.preference_weights_tensor, th.tensor(expected_weights, device=model.device)), \
            "Preference weights tensor incorrect"

        # Check replay buffer
        assert isinstance(model.replay_buffer, MOReplayBuffer), "Replay buffer class incorrect"

        # Default parameters
        assert model.learning_rate == 3e-4, "Default learning rate incorrect"
        assert model.batch_size == 256, "Default batch size incorrect"
        assert model.tau == 0.005, "Default tau incorrect"
        assert model.gamma == 0.99, "Default gamma incorrect"
        assert model.buffer_size == 1_000_000, "Default buffer size incorrect"

    def test_custom_preference_weights(self):
        """Test initialization with custom preference weights."""
        env = DummyMultiObjectiveEnv(num_objectives=4)
        custom_weights = np.array([0.4, 0.3, 0.2, 0.1], np.float32)  # Must sum to 1.0

        model = MOSAC(
            policy="MOSACPolicy",
            env=env,
            preference_weights=custom_weights
        )

        # Check preference weights are normalized and stored correctly
        expected_weights = np.array(custom_weights)  # Already sums to 1.0
        assert np.allclose(model.preference_weights, expected_weights), "Custom preference weights not stored correctly"
        assert th.allclose(model.preference_weights_tensor, th.tensor(expected_weights, device=model.device)), \
            "Preference weights tensor incorrect"

    def test_preference_weights_normalization(self):
        """Test that preference weights are automatically normalized."""
        env = DummyMultiObjectiveEnv(num_objectives=3)
        unnormalized_weights = [5, 3, 2]  # Sum = 10
        expected_normalized = np.array([0.5, 0.3, 0.2])  # Divided by sum

        model = MOSAC(
            policy="MOSACPolicy",
            env=env,
            preference_weights=unnormalized_weights
        )

        assert np.allclose(model.preference_weights, expected_normalized), \
            "Preference weights not normalized correctly"

    def test_invalid_preference_weights(self):
        """Test that invalid preference weights raise appropriate errors."""
        env = DummyMultiObjectiveEnv(num_objectives=3)

        # Wrong number of weights
        with pytest.raises(AssertionError):
            MOSAC(
                policy="MOSACPolicy",
                env=env,
                preference_weights=[0.5, 0.5]  # Only 2 weights for 3 objectives
            )

    def test_custom_network_architecture(self):
        """Test initialization with custom network architecture."""
        env = DummyMultiObjectiveEnv(num_objectives=3)

        # Custom network architecture
        policy_kwargs = {
            "net_arch": [128, 128],
            "share_features_across_objectives": False
        }

        model = MOSAC(
            policy="MOSACPolicy",
            env=env,
            policy_kwargs=policy_kwargs
        )

        # Check that policy_kwargs are passed to the policy
        assert model.policy_kwargs["net_arch"] == [128, 128], "net_arch not passed to policy"
        assert model.policy_kwargs["share_features_across_objectives"] == False, \
            "share_features_across_objectives not passed to policy"

        # Check that policy was built with correct architecture
        for q_net in model.critic.q_networks:
            assert isinstance(q_net, SeparateQNet), \
                "Policy should use SeparateQNet when share_features_across_objectives is False"

    def test_shared_vs_separate_features(self):
        """Test initialization with shared vs separate features across objectives."""
        env = DummyMultiObjectiveEnv(num_objectives=3)

        # Test with shared features (default)
        model_shared = MOSAC(
            policy="MOSACPolicy",
            env=env,
            policy_kwargs={"share_features_across_objectives": True}
        )

        # Test with separate features
        model_separate = MOSAC(
            policy="MOSACPolicy",
            env=env,
            policy_kwargs={"share_features_across_objectives": False}
        )

        # Check network structure
        for q_net in model_shared.critic.q_networks:
            assert isinstance(q_net, SharedFeatureQNet), \
                "Should use SharedFeatureQNet when share_features_across_objectives is True"

        for q_net in model_separate.critic.q_networks:
            assert isinstance(q_net, SeparateQNet), \
                "Should use SeparateQNet when share_features_across_objectives is False"

    def test_custom_critics(self):
        """Test initialization with custom number of critics."""
        env = DummyMultiObjectiveEnv(num_objectives=3)

        # Test with 3 critics
        model = MOSAC(
            policy="MOSACPolicy",
            env=env,
            policy_kwargs={"n_critics": 3}
        )

        assert hasattr(model.critic, "q_networks"), "Critic should have q_networks attribute"
        assert len(model.critic.q_networks) == 3, "Should have 3 critic networks"

    def test_learning_parameters(self):
        """Test initialization with custom learning parameters."""
        env = DummyMultiObjectiveEnv(num_objectives=3)

        model = MOSAC(
            policy="MOSACPolicy",
            env=env,
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=200,
            batch_size=128,
            tau=0.01,
            gamma=0.95,
            gradient_steps=2,
            target_update_interval=2
        )

        assert model.learning_rate == 1e-3, "Learning rate not set correctly"
        assert model.buffer_size == 50000, "Buffer size not set correctly"
        assert model.learning_starts == 200, "Learning starts not set correctly"
        assert model.batch_size == 128, "Batch size not set correctly"
        assert model.tau == 0.01, "Tau not set correctly"
        assert model.gamma == 0.95, "Gamma not set correctly"
        assert model.gradient_steps == 2, "Gradient steps not set correctly"
        assert model.target_update_interval == 2, "Target update interval not set correctly"

    def test_device_placement(self):
        """Test that models are placed on the correct device."""
        env = DummyMultiObjectiveEnv(num_objectives=3)

        # Test CPU placement
        model_cpu = MOSAC(
            policy="MOSACPolicy",
            env=env,
            device="cpu"
        )

        assert str(model_cpu.device) == "cpu", "Model should be on CPU"
        assert str(model_cpu.policy.device) == "cpu", "Policy should be on CPU"
        assert next(model_cpu.actor.parameters()).device.type == "cpu", "Actor parameters should be on CPU"
        assert next(model_cpu.critic.parameters()).device.type == "cpu", "Critic parameters should be on CPU"

        # Test CUDA placement if available
        if th.cuda.is_available():
            model_cuda = MOSAC(
                policy="MOSACPolicy",
                env=env,
                device="cuda"
            )

            assert str(model_cuda.device) == "cuda:0", "Model should be on CUDA"
            assert str(model_cuda.policy.device) == "cuda:0", "Policy should be on CUDA"
            assert next(model_cuda.actor.parameters()).device.type == "cuda", "Actor parameters should be on CUDA"
            assert next(model_cuda.critic.parameters()).device.type == "cuda", "Critic parameters should be on CUDA"


class TestMOSACTraining:
    """Comprehensive test suite for MOSAC training."""

    @pytest.fixture
    def mosac_model(self):
        """Create a basic MOSAC model for testing."""
        env = DummyMultiObjectiveEnv(num_objectives=3)
        model = MOSAC(
            policy="MOSACPolicy",
            env=env,
            learning_starts=0,  # Start training immediately
            gradient_steps=1,  # Single gradient step
            batch_size=4,  # Small batch size for testing
            verbose=0
        )
        return model

    @pytest.fixture
    def filled_buffer_model(self):
        """Create a MOSAC model with a filled replay buffer."""
        env = DummyMultiObjectiveEnv(num_objectives=3)
        model = MOSAC(
            policy="MOSACPolicy",
            env=env,
            learning_starts=0,
            gradient_steps=1,
            batch_size=10,
            verbose=0,
            num_objectives=3
        )

        # Fill the replay buffer with some samples
        obs= model.env.reset()
        for _ in range(10):
            action = model.predict(obs, deterministic=False)[0]
            next_obs, rewards, terminated,  info = model.env.step(action)
            model._store_transition(
                model.replay_buffer,
                action,
                next_obs,
                rewards,
                terminated,
                info
            )
            obs = next_obs
            if terminated :
                obs = model.env.reset()
        return model

    def test_train_single_step(self, filled_buffer_model):
        """Test a single training step."""

        model = filled_buffer_model
        model._setup_learn(total_timesteps=1)
        # Get initial parameters
        initial_actor_params = {name: param.clone().detach()
                                for name, param in model.actor.named_parameters()}
        initial_critic_params = {name: param.clone().detach()
                                 for name, param in model.critic.named_parameters()}

        # Perform one training step

        #model._setup_learn(total_timesteps=1)

        model.train(gradient_steps=1, batch_size=4)

        # Check that parameters have been updated
        actor_params_changed = False
        for name, param in model.actor.named_parameters():
            if not th.allclose(param, initial_actor_params[name]):
                actor_params_changed = True
                break

        critic_params_changed = False
        for name, param in model.critic.named_parameters():
            if not th.allclose(param, initial_critic_params[name]):
                critic_params_changed = True
                break

        assert actor_params_changed, "Actor parameters were not updated after training"
        assert critic_params_changed, "Critic parameters were not updated after training"

    def test_train_gradient_flow(self, filled_buffer_model):
        """Test that gradients flow correctly during training."""
        model = filled_buffer_model
        model._setup_learn(total_timesteps=1)
        # Sample data from the replay buffer
        replay_data = model.replay_buffer.sample(4, model._vec_normalize_env)

        # Forward pass through the actor
        actions_pi, log_prob = model.actor.action_log_prob(replay_data.observations)
        log_prob = log_prob.reshape(-1, 1)

        # Forward pass through the critic
        q_values_pi = th.cat(
            model.critic.q_value(replay_data.observations, actions_pi, model.preference_weights_tensor), dim=1)
        min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)

        # Compute actor loss
        ent_coef = th.ones(1, device=model.device) * 0.2  # Mock entropy coefficient
        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

        # Verify actor loss requires gradients
        assert actor_loss.requires_grad, "Actor loss should require gradients"

        # Backward pass
        model.actor.optimizer.zero_grad()
        actor_loss.backward()

        # Check that gradients are non-zero for actor parameters
        actor_has_grad = False
        for param in model.actor.parameters():
            if param.grad is not None and th.any(param.grad != 0):
                actor_has_grad = True
                break

        assert actor_has_grad, "Actor should have non-zero gradients"

        # Reset gradients
        model.actor.optimizer.zero_grad()

        # Now test critic gradient flow
        current_q_values = model.critic(replay_data.observations, replay_data.actions)

        # Create mock target values
        target_q_values = []
        for obj_idx in range(model.num_objectives):
            # Mock targets - just ones of the right shape
            target = th.ones_like(current_q_values[0][obj_idx])
            target_q_values.append([target for _ in range(len(current_q_values))])

        # Compute critic loss
        critic_loss = 0.5 * sum(
            sum(
                F.mse_loss(obj_current_q, target_q_values[obj_idx][critic_idx]) * model.preference_weights[obj_idx]
                for obj_idx, obj_current_q in enumerate(critic_ensemble)
            )
            for critic_idx, critic_ensemble in enumerate(current_q_values)
        )

        # Verify critic loss requires gradients
        assert critic_loss.requires_grad, "Critic loss should require gradients"

        # Backward pass
        model.critic.optimizer.zero_grad()
        critic_loss.backward()

        # Check that gradients are non-zero for critic parameters
        critic_has_grad = False
        for param in model.critic.parameters():
            if param.grad is not None and th.any(param.grad != 0):
                critic_has_grad = True
                break

        assert critic_has_grad, "Critic should have non-zero gradients"

    def test_preference_weight_effect(self, filled_buffer_model):
        """Test that different preference weights affect training outcomes."""
        
        model = filled_buffer_model
        # Clone the model to test with different weights
        model2 = deepcopy(model)

        # Set different preference weights
        model.preference_weights = np.array([0.8, 0.1, 0.1])
        model.preference_weights_tensor = th.FloatTensor(model.preference_weights).to(model.device)

        model2.preference_weights = np.array([0.1, 0.8, 0.1])
        model2.preference_weights_tensor = th.FloatTensor(model2.preference_weights).to(model2.device)

        # Save initial parameters
        model1_initial_critic = {name: param.clone().detach()
                                 for name, param in model.critic.named_parameters()}
        model2_initial_critic = {name: param.clone().detach()
                                 for name, param in model2.critic.named_parameters()}


        # Train both models
        model._setup_learn(total_timesteps=1)
        model2._setup_learn(total_timesteps=1)
        model.train(gradient_steps=1, batch_size=4)
        model2.train(gradient_steps=1, batch_size=4)

        # Compare parameter updates
        # The magnitudes and directions should differ due to different preference weights
        param_diffs = []
        for (name1, param1), (name2, param2) in zip(
                model.critic.named_parameters(), model2.critic.named_parameters()):

            # Calculate parameter changes for each model
            diff1 = param1 - model1_initial_critic[name1]
            diff2 = param2 - model2_initial_critic[name2]

            # Check if these changes are different
            if not th.allclose(diff1, diff2, atol=1e-4):
                param_diffs.append(name1)

        # Assert that at least some parameters updated differently
        assert len(param_diffs) > 0, "Different preference weights should cause different parameter updates"

    def test_target_network_update(self, filled_buffer_model):
        """Test that target networks are updated correctly."""
        model = filled_buffer_model

        # Get initial target parameters
        initial_target_params = {name: param.clone().detach()
                                 for name, param in model.critic_target.named_parameters()}

        # Train for one step
        model._setup_learn(total_timesteps=1)
        model.train(gradient_steps=1, batch_size=4)

        # Check that target parameters have been updated
        target_params_changed = False
        for name, param in model.critic_target.named_parameters():
            if not th.allclose(param, initial_target_params[name]):
                target_params_changed = True
                break

        assert target_params_changed, "Target network parameters were not updated"

        # Verify polyak update (soft update)
        # For some parameter pairs, check: target_new = (1-tau)*target_old + tau*critic_current
        current_critic_params = {name: param for name, param in model.critic.named_parameters()}
        current_target_params = {name: param for name, param in model.critic_target.named_parameters()}

        for name in current_critic_params:
            if name in initial_target_params and name in current_target_params:
                expected = (1 - model.tau) * initial_target_params[name] + model.tau * current_critic_params[name]
                actual = current_target_params[name]

                # Check if close within numerical precision
                assert th.allclose(actual, expected, atol=1e-5), f"Polyak update incorrect for {name}"

    def test_vector_reward_handling(self, mosac_model):
        """Test that vector rewards are correctly processed during training."""
        model = mosac_model
        
        # Fill buffer with some vector rewards
        # With this safer version:
        obs, _ = model.env.reset()

        # Generate actions and custom vector rewards
        for i in range(10):
            action = model.predict(obs, deterministic=False)[0]

            next_obs, _, terminated,  info = model.env.envs[0].step(action)

            # Create a custom vector reward with a specific pattern
            # This makes it easier to verify the rewards are used correctly
            reward = np.array([1.0, 0.5, 0.0])  # Simple pattern
            custom_rewards.append(reward)

            model._store_transition(
                model.replay_buffer,
                action,
                next_obs,
                reward,
                np.array([terminated or truncated]),
                info
            )

            obs = next_obs
            if terminated or truncated:
                #the enviroment is in vector of enviroments
                obs, _ = model.env.reset()

        # Sample from replay buffer
        batch = model.replay_buffer.sample(4, model._vec_normalize_env)

        # Verify rewards have the correct shape
        assert batch.rewards.shape[1] == 3, "Vector rewards should have 3 objectives"

        # Check that rewards in the batch match our pattern
        # At least some of the rewards should have our custom pattern
        found_pattern = False
        for i in range(batch.rewards.shape[0]):
            reward_row = batch.rewards[i].cpu().numpy()
            # Check if this row matches our pattern
            if np.allclose(reward_row, [1.0, 0.5, 0.0], atol=1e-4):
                found_pattern = True
                break

        assert found_pattern, "Custom reward pattern not found in sampled batch"

        # Now train and ensure no errors occur with vector rewards
        try:
            model.train(gradient_steps=1, batch_size=4)
            assert True, "Training with vector rewards succeeded"
        except Exception as e:
            pytest.fail(f"Training with vector rewards failed with error: {e}")

    def test_batch_size_effect(self, filled_buffer_model):
        """Test training with different batch sizes."""
        model = filled_buffer_model

        # Fill more samples into the buffer
        obs = model.env.reset()
        for _ in range(30):  # Add 30 more samples
            action = model.predict(obs, deterministic=False)[0]
            next_obs, rewards, terminated,  info = model.env.step(action)
            model._store_transition(
                model.replay_buffer,
                action,
                next_obs,
                rewards,
                terminated,
                info
            )
            obs = next_obs
            if terminated :
                obs = model._vec_normalize_env.reset()

        # Train with different batch sizes
        for batch_size in [4, 8, 16]:
            # Clone the model to compare
            model_copy = deepcopy(model)

            # Train with specific batch size
            model_copy._setup_learn(total_timesteps=1)
            model_copy.train(gradient_steps=1, batch_size=batch_size)

            # We can't directly compare the parameter values as they'll differ by batch
            # But we can verify no errors occur and loss values are reasonable
            assert model_copy.logger.name_to_value.get("train/actor_loss") is not None, \
                f"Actor loss not recorded for batch size {batch_size}"
            assert model_copy.logger.name_to_value.get("train/critic_loss") is not None, \
                f"Critic loss not recorded for batch size {batch_size}"

            # Losses should be finite
            assert np.isfinite(model_copy.logger.name_to_value.get("train/actor_loss")), \
                f"Actor loss is not finite for batch size {batch_size}"
            assert np.isfinite(model_copy.logger.name_to_value.get("train/critic_loss")), \
                f"Critic loss is not finite for batch size {batch_size}"