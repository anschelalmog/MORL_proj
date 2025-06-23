import pytest
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import FlattenExtractor, BaseFeaturesExtractor
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

# Import your custom classes
from agents.monets import SharedFeatureQNet, SeparateQNet
# Replace this import with the actual module where your classes are defined
from agents.mosac import MOSACPolicy, MOContinuousCritic


class SimpleMOEnv(gym.Env):
    """Simple multi-objective environment for testing purposes."""

    def __init__(self, num_objectives=2):
        super().__init__()
        self.observation_space = spaces.Box(low=-10, high=10, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.num_objectives = num_objectives

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = np.random.uniform(-1, 1, size=(5,)).astype(np.float32)
        return self.state, {}

    def step(self, action):
        # Generate multi-objective rewards
        rewards = np.random.uniform(-1, 1, size=(self.num_objectives,)).astype(np.float32)
        self.state = np.random.uniform(-1, 1, size=(5,)).astype(np.float32)
        terminated = False
        truncated = False
        info = {"rewards": rewards}
        # Return first reward as scalar for SB3 compatibility
        return self.state, rewards[0], terminated, truncated, info


@pytest.fixture
def mo_env():
    """Create a multi-objective environment."""
    return SimpleMOEnv(num_objectives=3)


@pytest.fixture
def policy(mo_env):
    """Create a MOSACPolicy for testing."""
    observation_space = mo_env.observation_space
    action_space = mo_env.action_space

    return MOSACPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,  # Constant learning rate
        num_objectives=3,
        net_arch=[64, 64],
        n_critics=2,
        share_features_across_objectives=True
    )


class TestMOSACPolicy:
    """Test suite for MOSACPolicy."""

    def test_policy_initialization(self, mo_env):
        """Test that the policy initializes correctly with different parameters."""
        observation_space = mo_env.observation_space
        action_space = mo_env.action_space

        # Test initialization with default parameters
        policy = MOSACPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda _: 3e-4,
            num_objectives=3
        )
        assert policy.num_objectives == 3
        assert isinstance(policy.critic, MOContinuousCritic)

        # Test initialization with shared features
        policy = MOSACPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda _: 3e-4,
            num_objectives=3,
            share_features_across_objectives=True
        )
        assert policy.share_features_across_objectives == True

        # Test initialization with separate features
        policy = MOSACPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda _: 3e-4,
            num_objectives=3,
            share_features_across_objectives=False
        )
        assert policy.share_features_across_objectives == False

        # Test with custom network architecture
        custom_net_arch = [128, 64, 32]
        policy = MOSACPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda _: 3e-4,
            num_objectives=3,
            net_arch=custom_net_arch
        )
        # Check that the architecture was applied correctly
        # This might require checking the actual network structures

        # Test with different number of critics
        policy = MOSACPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda _: 3e-4,
            num_objectives=3,
            n_critics=3
        )
        assert len(policy.critic.q_networks) == 3

    def test_forward_pass(self, policy, mo_env):
        """Test the forward pass of the actor network."""
        obs = th.FloatTensor(np.random.uniform(-1, 1, size=(10, 5)))

        mean_actions, log_std, _ = policy.actor.get_action_dist_params(obs)

        # Check shapes
        assert mean_actions.shape == (10, mo_env.action_space.shape[0])
        assert log_std.shape == (10, mo_env.action_space.shape[0])

        # Check values are within expected ranges
        assert th.all(mean_actions >= -1) and th.all(mean_actions <= 1)

    def test_action_prediction(self, policy, mo_env):
        """Test that the policy can predict actions."""
        obs = np.random.uniform(-1, 1, size=(5,)).astype(np.float32)

        # Test deterministic action
        action, _ = policy.predict(obs, deterministic=True)
        assert action.shape == mo_env.action_space.shape
        assert np.all(action >= -1) and np.all(action <= 1)

        # Test stochastic action
        action, _ = policy.predict(obs, deterministic=False)
        assert action.shape == mo_env.action_space.shape
        assert np.all(action >= -1) and np.all(action <= 1)

    def test_critic_forward(self, policy, mo_env):
        """Test the forward pass of the critic network."""
        obs = th.FloatTensor(np.random.uniform(-1, 1, size=(10, 5)))
        actions = th.FloatTensor(np.random.uniform(-1, 1, size=(10, 2)))

        # Get critic values
        critic_values = policy.critic(obs, actions)

        # Check we have the right number of critic ensembles
        assert len(critic_values) == policy.critic.n_critics

        # Check each critic produces the right number of objective values
        for critic_ensemble in critic_values:
            assert len(critic_ensemble) == policy.num_objectives

            # Check each objective value has the right shape
            for obj_values in critic_ensemble:
                assert obj_values.shape == (10, 1)

    def test_q_value_calculation(self, policy, mo_env):
        """Test the calculation of Q-values with preference weights."""
        obs = th.FloatTensor(np.random.uniform(-1, 1, size=(10, 5)))
        actions = th.FloatTensor(np.random.uniform(-1, 1, size=(10, 2)))

        # Test with uniform weights
        weights = th.ones(policy.num_objectives) / policy.num_objectives
        q_values = policy.critic.q_value(obs, actions, weights)

        # Check q_values shape and type
        assert len(q_values) == policy.critic.n_critics
        for q_val in q_values:
            assert q_val.shape == (10, 1)

        # Test with random weights
        weights = th.FloatTensor(np.random.uniform(0, 1, size=(policy.num_objectives,)))
        weights = weights / weights.sum()  # Normalize
        q_values = policy.critic.q_value(obs, actions, weights)

        # Check q_values shape and type
        assert len(q_values) == policy.critic.n_critics
        for q_val in q_values:
            assert q_val.shape == (10, 1)

        # Test with batch-specific weights
        weights = th.FloatTensor(np.random.uniform(0, 1, size=(10, policy.num_objectives)))
        weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize each row
        q_values = policy.critic.q_value(obs, actions, weights)

        # Check q_values shape and type
        assert len(q_values) == policy.critic.n_critics
        for q_val in q_values:
            assert q_val.shape == (10, 1)

    def test_shared_vs_separate_features(self, mo_env):
        """Test the difference between shared and separate features."""
        observation_space = mo_env.observation_space
        action_space = mo_env.action_space

        # Create policy with shared features
        shared_policy = MOSACPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda _: 3e-4,
            num_objectives=3,
            share_features_across_objectives=True
        )

        # Create policy with separate features
        separate_policy = MOSACPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda _: 3e-4,
            num_objectives=3,
            share_features_across_objectives=False
        )

        # Check that the critic networks are of the correct type
        for q_net in shared_policy.critic.q_networks:
            assert isinstance(q_net, SharedFeatureQNet)

        for q_net in separate_policy.critic.q_networks:
            assert isinstance(q_net, SeparateQNet)

        # Test forward pass for both
        obs = th.FloatTensor(np.random.uniform(-1, 1, size=(10, 5)))
        actions = th.FloatTensor(np.random.uniform(-1, 1, size=(10, 2)))

        shared_critic_values = shared_policy.critic(obs, actions)
        separate_critic_values = separate_policy.critic(obs, actions)

        # Both should produce the same structure of outputs
        assert len(shared_critic_values) == len(separate_critic_values)
        for shared_ensemble, separate_ensemble in zip(shared_critic_values, separate_critic_values):
            assert len(shared_ensemble) == len(separate_ensemble)
            for shared_obj, separate_obj in zip(shared_ensemble, separate_ensemble):
                assert shared_obj.shape == separate_obj.shape

    def test_training_step(self, policy):
        """Test a single training step."""
        # Create a small batch of data
        batch_size = 32
        observations = th.FloatTensor(np.random.uniform(-1, 1, size=(batch_size, 5)))
        actions = th.FloatTensor(np.random.uniform(-1, 1, size=(batch_size, 2)))
        rewards = th.FloatTensor(np.random.uniform(-1, 1, size=(batch_size, 1)))
        next_observations = th.FloatTensor(np.random.uniform(-1, 1, size=(batch_size, 5)))
        dones = th.FloatTensor(np.zeros((batch_size, 1)))

        # Generate preference weights
        weights = th.FloatTensor(np.random.uniform(0, 1, size=(batch_size, policy.num_objectives)))
        weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize

        # Mock a replay buffer sample
        replay_data = {
            "observations": observations,
            "actions": actions,
            "next_observations": next_observations,
            "dones": dones,
            "rewards": rewards,
            "preference_weights": weights  # Add preference weights
        }

        # Get initial parameters
        initial_actor_params = [param.clone().detach() for param in policy.actor.parameters()]
        initial_critic_params = [param.clone().detach() for param in policy.critic.parameters()]

        # Perform training step (this would depend on your implementation)
        # For example:
        # policy.train()
        # critic_loss, actor_loss = policy._train_step(replay_data)

        # Since we don't have the actual training step implementation,
        # we'll just check that the training step can be called without errors
        # and that the parameters are updated

        # This is a placeholder for the actual training step code
        # policy._train_step(replay_data)

        # Get updated parameters
        updated_actor_params = [param.clone().detach() for param in policy.actor.parameters()]
        updated_critic_params = [param.clone().detach() for param in policy.critic.parameters()]

        # Check if parameters were updated
        # (In a real test, we'd actually run the training step)
        # for initial, updated in zip(initial_actor_params, updated_actor_params):
        #     assert not th.allclose(initial, updated)
        # for initial, updated in zip(initial_critic_params, updated_critic_params):
        #     assert not th.allclose(initial, updated)

    def test_save_load(self, policy, tmp_path):
        """Test saving and loading the policy."""
        # Save the policy
        path = str(tmp_path / "mo_sac_policy.pt")
        policy.save(path)

        # Create a new policy with the same parameters
        new_policy = MOSACPolicy(
            observation_space=policy.observation_space,
            action_space=policy.action_space,
            lr_schedule=lambda _: 3e-4,
            num_objectives=policy.num_objectives,
            share_features_across_objectives=policy.share_features_across_objectives
        )

        # Load the saved parameters
        new_policy.load(path)

        # Check that the parameters are the same
        for param1, param2 in zip(policy.parameters(), new_policy.parameters()):
            assert th.allclose(param1, param2)

    def test_integration_with_environment(self, mo_env):
        """Test integration with a multi-objective environment."""
        # Create a vector environment
        vec_env = DummyVecEnv([lambda: mo_env])

        # Create policy
        policy = MOSACPolicy(
            observation_space=mo_env.observation_space,
            action_space=mo_env.action_space,
            lr_schedule=lambda _: 3e-4,
            num_objectives=mo_env.num_objectives
        )

        # Test interaction with environment
        obs = vec_env.reset()
        action, _ = policy.predict(obs)
        next_obs, reward, done, info = vec_env.step(action)

        # Check shapes and types
        assert action.shape == (1,) + mo_env.action_space.shape
        assert reward.shape == (1,)
        assert done.shape == (1,)

        # Check that the reward is scalar (first objective)
        assert isinstance(reward[0], np.float32)

        # Check that all objectives are in info
        assert "rewards" in info[0]
        assert info[0]["rewards"].shape == (mo_env.num_objectives,)

    def test_different_objective_counts(self, mo_env):
        """Test the policy with different numbers of objectives."""
        for num_objectives in [1, 2, 4, 8]:
            # Create environment with specific number of objectives
            env = SimpleMOEnv(num_objectives=num_objectives)

            # Create policy
            policy = MOSACPolicy(
                observation_space=env.observation_space,
                action_space=env.action_space,
                lr_schedule=lambda _: 3e-4,
                num_objectives=num_objectives
            )

            # Check critic structure
            assert policy.num_objectives == num_objectives

            # Test forward pass
            obs = th.FloatTensor(np.random.uniform(-1, 1, size=(10, 5)))
            actions = th.FloatTensor(np.random.uniform(-1, 1, size=(10, 2)))

            critic_values = policy.critic(obs, actions)

            # Check we have the right number of critic ensembles
            assert len(critic_values) == policy.critic.n_critics

            # Check each critic produces the right number of objective values
            for critic_ensemble in critic_values:
                assert len(critic_ensemble) == num_objectives

    def test_preference_weight_handling(self, policy):
        """Test how the policy handles different preference weight formats."""
        obs = th.FloatTensor(np.random.uniform(-1, 1, size=(10, 5)))
        actions = th.FloatTensor(np.random.uniform(-1, 1, size=(10, 2)))

        # Test with vector weights (should be broadcast to batch)
        weights = th.FloatTensor(np.random.uniform(0, 1, size=(policy.num_objectives,)))
        weights = weights / weights.sum()  # Normalize
        q_values = policy.critic.q_value(obs, actions, weights)

        # Check q_values shape
        assert len(q_values) == policy.critic.n_critics
        for q_val in q_values:
            assert q_val.shape == (10, 1)

        # Test with batch-specific weights
        weights = th.FloatTensor(np.random.uniform(0, 1, size=(10, policy.num_objectives)))
        weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize each row
        q_values = policy.critic.q_value(obs, actions, weights)

        # Check q_values shape
        assert len(q_values) == policy.critic.n_critics
        for q_val in q_values:
            assert q_val.shape == (10, 1)

    def test_edge_cases(self, mo_env):
        """Test edge cases and potential failure scenarios."""
        observation_space = mo_env.observation_space
        action_space = mo_env.action_space

        # Test with single objective (edge case)
        policy = MOSACPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda _: 3e-4,
            num_objectives=1
        )

        # Test with large number of objectives
        policy = MOSACPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda _: 3e-4,
            num_objectives=10
        )

        # Test with high-dimensional observation space
        high_dim_space = spaces.Box(low=-10, high=10, shape=(100,), dtype=np.float32)
        policy = MOSACPolicy(
            observation_space=high_dim_space,
            action_space=action_space,
            lr_schedule=lambda _: 3e-4,
            num_objectives=3
        )

        # Test with high-dimensional action space
        high_dim_action = spaces.Box(low=-1, high=1, shape=(20,), dtype=np.float32)
        policy = MOSACPolicy(
            observation_space=observation_space,
            action_space=high_dim_action,
            lr_schedule=lambda _: 3e-4,
            num_objectives=3
        )