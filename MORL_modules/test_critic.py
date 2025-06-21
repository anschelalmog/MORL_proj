import pytest
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import FlattenExtractor
# Adjust the import to match your module structure
from agents.mosac import MOContinuousCritic
from agents.monets import SharedFeatureQNet, SeparateQNet


# Fixtures for common test data
@pytest.fixture
def observation_space():
    return spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)


@pytest.fixture
def action_space():
    return spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)


@pytest.fixture
def basic_net_arch():
    return [64, 64]


@pytest.fixture
def sample_batch_size():
    return 10


@pytest.fixture
def sample_obs(observation_space, sample_batch_size):
    return th.tensor(
        np.random.uniform(
            low=observation_space.low,
            high=observation_space.high,
            size=(sample_batch_size,) + observation_space.shape
        ),
        dtype=th.float32
    )


@pytest.fixture
def sample_actions(action_space, sample_batch_size):
    return th.tensor(
        np.random.uniform(
            low=action_space.low,
            high=action_space.high,
            size=(sample_batch_size,) + action_space.shape
        ),
        dtype=th.float32
    )


@pytest.fixture
def sample_preferences(sample_batch_size):
    # Random preferences that sum to 1
    prefs = np.random.rand(sample_batch_size, 2)
    prefs = prefs / prefs.sum(axis=1, keepdims=True)
    return th.tensor(prefs, dtype=th.float32)


# Test initialization with shared features
def test_critic_init_shared_features(observation_space, action_space, basic_net_arch):
    """Test initialization with shared features across objectives"""
    num_objectives = 3
    critic = MOContinuousCritic(
        observation_space=observation_space,
        action_space=action_space,
        net_arch=basic_net_arch,
        num_objectives=num_objectives,
        share_features_across_objectives=True
    )

    # Check if the correct number of q-networks was created
    assert len(critic.q_networks) == critic.n_critics

    # Check if each q-network is of the correct type
    for q_net in critic.q_networks:
        assert isinstance(q_net, SharedFeatureQNet)

        # Check if the correct number of heads was created
        assert len(q_net.heads) == num_objectives

        # Check if each head outputs a single value
        for head in q_net.heads:
            assert head.out_features == 1


# Test initialization with separate features
def test_critic_init_separate_features(observation_space, action_space, basic_net_arch):
    """Test initialization with separate features for each objective"""
    num_objectives = 3
    critic = MOContinuousCritic(
        observation_space=observation_space,
        action_space=action_space,
        net_arch=basic_net_arch,
        num_objectives=num_objectives,
        share_features_across_objectives=False
    )

    # Check if the correct number of q-networks was created
    assert len(critic.q_networks) == critic.n_critics

    # Check if each q-network is of the correct type
    for q_net in critic.q_networks:
        assert isinstance(q_net, SeparateQNet)

        # Check if the correct number of objective networks was created
        assert len(q_net.nets) == num_objectives


# Test forward pass
def test_forward_pass(observation_space, action_space, basic_net_arch, sample_obs, sample_actions):
    """Test forward pass and output shape"""
    num_objectives = 2
    critic = MOContinuousCritic(
        observation_space=observation_space,
        action_space=action_space,
        net_arch=basic_net_arch,
        num_objectives=num_objectives
    )

    # Get output from forward pass
    q_values = critic.forward(sample_obs, sample_actions)

    # Check if output is a tuple
    assert isinstance(q_values, tuple)

    # Check if tuple has the correct length (n_critics)
    assert len(q_values) == critic.n_critics

    # Check if each critic's output has the correct shape and length
    for critic_output in q_values:
        # Check if critic output is a list
        assert isinstance(critic_output, list)

        # Check if it has the right number of objectives
        assert len(critic_output) == num_objectives

        # Check the shape of each objective's Q-value tensor
        for obj_q_value in critic_output:
            assert obj_q_value.shape == (sample_obs.shape[0], 1)


# Test Q-value scalarization
def test_q_value_scalarization(observation_space, action_space, basic_net_arch,
                               sample_obs, sample_actions, sample_preferences):
    """Test q_value function with preference weights"""
    num_objectives = 2
    critic = MOContinuousCritic(
        observation_space=observation_space,
        action_space=action_space,
        net_arch=basic_net_arch,
        num_objectives=num_objectives
    )

    # Get scalarized Q-values
    scalarized_q_values = critic.q_value(sample_obs, sample_actions, sample_preferences)

    # Check if output is a list
    assert isinstance(scalarized_q_values, list)

    # Check if list has the correct length (n_critics)
    assert len(scalarized_q_values) == critic.n_critics

    # Check the shape of each scalarized Q-value tensor
    for scalarized_q in scalarized_q_values:
        assert scalarized_q.shape == (sample_obs.shape[0], 1)

    # Test with 1D preference weights
    single_pref = th.tensor([0.7, 0.3], dtype=th.float32)
    scalarized_q_values_single = critic.q_value(sample_obs, sample_actions, single_pref)

    # Check if output has the correct shape
    for scalarized_q in scalarized_q_values_single:
        assert scalarized_q.shape == (sample_obs.shape[0], 1)


# Test different numbers of objectives
def test_multiple_objectives(observation_space, action_space, basic_net_arch, sample_obs, sample_actions):
    """Test with different numbers of objectives"""
    for num_objectives in [1, 2, 5]:
        critic = MOContinuousCritic(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=basic_net_arch,
            num_objectives=num_objectives
        )

        q_values = critic.forward(sample_obs, sample_actions)

        # Check correct number of objectives in output
        for critic_output in q_values:
            assert len(critic_output) == num_objectives


# Test custom feature extractor
def test_custom_feature_extractor(observation_space, action_space, basic_net_arch):
    """Test that the critic properly handles custom feature extractors"""
    # Create a custom feature extractor with a specific feature dimension
    features_dim = 32
    features_extractor = FlattenExtractor(observation_space)

    critic = MOContinuousCritic(
        observation_space=observation_space,
        action_space=action_space,
        net_arch=basic_net_arch,
        features_extractor=features_extractor,
        features_dim=features_dim
    )
    breakpoint()
    # Check if the features_dim was properly set
    assert critic.features_dim == features_dim


# Test scalarization correctness
def test_scalarization_correctness(observation_space, action_space, basic_net_arch,
                                   sample_obs, sample_actions):
    """Test that scalarization correctly computes weighted sums"""
    num_objectives = 2
    critic = MOContinuousCritic(
        observation_space=observation_space,
        action_space=action_space,
        net_arch=basic_net_arch,
        num_objectives=num_objectives
    )

    # Fixed preference weights for deterministic testing
    preferences = th.tensor([[0.3, 0.7]], dtype=th.float32).expand(sample_obs.shape[0], -1)

    # Get raw Q-values and scalarized Q-values
    raw_q_values = critic.forward(sample_obs, sample_actions)
    scalarized_q_values = critic.q_value(sample_obs, sample_actions, preferences)

    # Manually compute weighted sums for verification
    for i in range(critic.n_critics):
        # Get individual objective Q-values for this critic
        critic_obj_values = raw_q_values[i]

        # Stack objective values into a single tensor
        stacked_values = th.cat([q_val for q_val in critic_obj_values], dim=1)

        # Compute weighted sum
        expected_scalarized = th.sum(stacked_values * preferences, dim=1, keepdim=True)

        # Compare with the critic's computation
        assert th.allclose(expected_scalarized, scalarized_q_values[i], rtol=1e-5)


# Test different n_critics values
def test_n_critics_parameter(observation_space, action_space, basic_net_arch):
    """Test initialization with different numbers of critic ensembles"""
    for n_critics in [1, 2, 4]:
        critic = MOContinuousCritic(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=basic_net_arch,
            n_critics=n_critics
        )

        # Check if the correct number of q-networks was created
        assert len(critic.q_networks) == n_critics


# Test network architecture configurations
def test_network_architecture(observation_space, action_space):
    """Test different network architecture configurations"""
    num_objectives = 2

    # Test different network architectures
    for net_arch in [[32, 32], [64, 32, 16], [128, 64]]:
        critic = MOContinuousCritic(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            num_objectives=num_objectives
        )

        # Verify critic was initialized successfully
        assert critic is not None


# Test activation functions
def test_activation_functions(observation_space, action_space, basic_net_arch):
    """Test initialization with different activation functions"""
    for activation_fn in [th.nn.ReLU, th.nn.Tanh, th.nn.LeakyReLU]:
        critic = MOContinuousCritic(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=basic_net_arch,
            activation_fn=activation_fn,
            share_features_across_objectives=True
        )

        # Check if critic initialized successfully
        assert critic is not None

        # For shared networks, verify the activation function is used in the network
        # Find an instance of the activation in the base_net
        activation_found = False
        for q_net in critic.q_networks:
            if isinstance(q_net, SharedFeatureQNet):
                for module in q_net.base_net:
                    if isinstance(module, activation_fn):
                        activation_found = True
                        break
            if activation_found:
                break

        assert activation_found


# Test internal structure of SharedFeatureQNet
def test_shared_feature_qnet_structure(observation_space, action_space, basic_net_arch):
    """Test the internal structure of SharedFeatureQNet instances"""
    num_objectives = 3
    critic = MOContinuousCritic(
        observation_space=observation_space,
        action_space=action_space,
        net_arch=basic_net_arch,
        num_objectives=num_objectives,
        share_features_across_objectives=True
    )

    for q_net in critic.q_networks:
        # Check that it's a SharedFeatureQNet
        assert isinstance(q_net, SharedFeatureQNet)

        # Check base_net structure
        assert isinstance(q_net.base_net, nn.Sequential)

        # Check heads structure
        assert isinstance(q_net.heads, nn.ModuleList)
        assert len(q_net.heads) == num_objectives

        # Check that each head is a linear layer
        for head in q_net.heads:
            assert isinstance(head, nn.Linear)
            assert head.out_features == 1


# Test internal structure of SeparateQNet
def test_separate_qnet_structure(observation_space, action_space, basic_net_arch):
    """Test the internal structure of SeparateQNet instances"""
    num_objectives = 3
    critic = MOContinuousCritic(
        observation_space=observation_space,
        action_space=action_space,
        net_arch=basic_net_arch,
        num_objectives=num_objectives,
        share_features_across_objectives=False
    )

    for q_net in critic.q_networks:
        # Check that it's a SeparateQNet
        assert isinstance(q_net, SeparateQNet)

        # Check nets structure
        assert isinstance(q_net.nets, nn.ModuleList)
        assert len(q_net.nets) == num_objectives

        # Check that each network is a sequential
        for net in q_net.nets:
            assert isinstance(net, nn.Sequential)
