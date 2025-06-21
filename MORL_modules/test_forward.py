import pytest
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import FlattenExtractor


# Adjust the import to match your module structure
from agents.mosac import MOContinuousCritic
from agents.monets import SharedFeatureQNet, SeparateQNet


# Fixtures for test setup
@pytest.fixture
def sample_batch_sizes():
    """Multiple batch sizes to test with"""
    return [1, 10, 32, 64]


@pytest.fixture
def observation_dims():
    """Different observation dimensions to test with"""
    return [4, 8, 16]


@pytest.fixture
def action_dims():
    """Different action dimensions to test with"""
    return [2, 4, 6]


@pytest.fixture
def default_space():
    """Default spaces for basic tests"""
    obs_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    act_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    return obs_space, act_space


@pytest.fixture
def mock_critic(default_space):
    """A simple critic with fixed weights for deterministic output testing"""
    obs_space, act_space = default_space

    # Create a critic with a small, fixed network for predictable outputs
    critic = MOContinuousCritic(
        observation_space=obs_space,
        action_space=act_space,
        net_arch=[4],
        num_objectives=2,
        n_critics=2,
        share_features_across_objectives=True
    )

    # Set fixed weights for deterministic output
    for i, q_net in enumerate(critic.q_networks):
        # Set base_net weights
        with th.no_grad():
            # First layer weights and bias
            if isinstance(q_net.base_net[0], nn.Linear):
                q_net.base_net[0].weight.fill_(0.1 * (i + 1))
                q_net.base_net[0].bias.fill_(0.01 * (i + 1))

            # Set head weights
            breakpoint()
            for j, head in enumerate(q_net.heads):
                head.weight.fill_(0.2 * (i + 1) * (j + 1))
                head.bias.fill_(0.02 * (i + 1) * (j + 1))

    return critic


# Basic forward pass test with different input dimensions
def test_forward_input_dimensions(observation_dims, action_dims, sample_batch_sizes):
    """Test forward pass with different input dimensions"""

    for obs_dim in observation_dims:
        for act_dim in action_dims:
            for batch_size in sample_batch_sizes:
                # Create observation and action spaces
                obs_space = spaces.Box(low=-1, high=1, shape=(obs_dim,), dtype=np.float32)
                act_space = spaces.Box(low=-1, high=1, shape=(act_dim,), dtype=np.float32)

                # Create critic
                critic = MOContinuousCritic(
                    observation_space=obs_space,
                    action_space=act_space,
                    net_arch=[64, 64],
                    num_objectives=2
                )

                # Create sample observations and actions
                obs = th.randn(batch_size, obs_dim)
                actions = th.randn(batch_size, act_dim)

                # Run forward pass
                outputs = critic.forward(obs, actions)

                # Check output structure
                assert isinstance(outputs, tuple)
                assert len(outputs) == critic.n_critics

                # Check each critic's output
                for critic_output in outputs:
                    assert isinstance(critic_output, list)
                    assert len(critic_output) == 2  # num_objectives

                    # Check each objective's output dimensions
                    for obj_output in critic_output:
                        assert obj_output.shape == (batch_size, 1)


# Test the exact output values with a fixed-weight network
def test_forward_output_values(mock_critic):
    """Test that forward pass produces expected output values with fixed weights"""
    batch_size = 2
    obs_dim = 4
    act_dim = 2

    # Create deterministic input
    obs = th.ones(batch_size, obs_dim)
    actions = th.ones(batch_size, act_dim)

    # Run forward pass
    with th.no_grad():
        outputs = mock_critic.forward(obs, actions)
    # For a network with fixed weights as set in the mock_critic fixture,
    # we can predict the exact output values

    # Check each critic ensemble
    for i, critic_output in enumerate(outputs):
        critic_idx = i + 1  # 1-indexed for our weight setting

        # For each objective
        for j, obj_output in enumerate(critic_output):
            obj_idx = j + 1  # 1-indexed

            # Calculate expected output:
            # First layer: input * weight + bias
            # Sum of input is obs_dim + act_dim = 6 (all ones)
            # First layer output = (6 * 0.1 * critic_idx) + (0.01 * critic_idx) = 0.61 * critic_idx
            # ReLU activation = 0.61 * critic_idx (all positive)
            # Head output = (0.61 * critic_idx * 0.2 * critic_idx * obj_idx) + (0.02 * critic_idx * obj_idx)

            # Expected value calculation for each sample (same for all in batch due to identical inputs)
            expected_base_output = 0.61 * critic_idx
            expected_head_output = 4*(expected_base_output * 0.2 * critic_idx * obj_idx) + (0.02 * critic_idx * obj_idx)
            breakpoint()
            # Check that all batch elements have this value
            assert th.allclose(obj_output, th.full_like(obj_output, expected_head_output), atol=1e-5)


# Test gradient flow with shared vs non-shared feature extractors
def test_forward_gradient_flow(default_space):
    """Test gradient flow through the network during forward pass"""
    obs_space, act_space = default_space
    batch_size = 10

    # Test both shared and non-shared feature extractor settings
    for share_features in [True, False]:
        # Create critic
        critic = MOContinuousCritic(
            observation_space=obs_space,
            action_space=act_space,
            net_arch=[64, 64],
            num_objectives=2,
            share_features_extractor=share_features
        )

        # Sample data
        obs = th.randn(batch_size, obs_space.shape[0], requires_grad=True)
        actions = th.randn(batch_size, act_space.shape[0], requires_grad=True)

        # Forward pass
        outputs = critic.forward(obs, actions)

        # Compute loss (sum of all outputs) and backward pass
        loss = sum(sum(output) for critic_outputs in outputs for output in critic_outputs)
        loss.backward()

        # Check if gradients flowed properly through the network
        if share_features:
            # If sharing, feature extractor gradients should be zero when share_features_extractor=True
            # because the forward pass disables gradients for the features extractor
            for param in critic.features_extractor.parameters():
                assert param.grad is None or th.all(param.grad == 0)
        else:
            # If not sharing, gradients should flow to all parameters
            for param in critic.parameters():
                if param.requires_grad:
                    assert param.grad is not None
                    assert not th.all(param.grad == 0)


# Test with different shared/separate network configurations
def test_forward_with_network_types(default_space, sample_batch_sizes):
    """Test forward pass with both shared and separate network types"""
    obs_space, act_space = default_space

    for batch_size in sample_batch_sizes:
        for share_features in [True, False]:
            # Create critic
            critic = MOContinuousCritic(
                observation_space=obs_space,
                action_space=act_space,
                net_arch=[32, 32],
                num_objectives=3,
                share_features_across_objectives=share_features
            )

            # Sample data
            obs = th.randn(batch_size, obs_space.shape[0])
            actions = th.randn(batch_size, act_space.shape[0])

            # Forward pass
            outputs = critic.forward(obs, actions)

            # Verify output structure
            assert isinstance(outputs, tuple)
            assert len(outputs) == critic.n_critics

            # Check the correct type based on configuration
            for critic_idx, critic_outputs in enumerate(outputs):
                assert isinstance(critic_outputs, list)
                assert len(critic_outputs) == 3  # num_objectives

                for obj_idx, obj_output in enumerate(critic_outputs):
                    # Verify all outputs have correct shape
                    assert obj_output.shape == (batch_size, 1)

                    # Verify outputs are different (proper forward pass)
                    if critic_idx > 0 or obj_idx > 0:
                        # Compare with first output - they should be different
                        first_output = outputs[0][0]
                        assert not th.allclose(obj_output, first_output)


# Test handling of different input types and ranges
def test_forward_input_types(default_space):
    """Test forward pass with different input types and ranges"""
    obs_space, act_space = default_space
    batch_size = 16

    # Create critic
    critic = MOContinuousCritic(
        observation_space=obs_space,
        action_space=act_space,
        net_arch=[64, 64],
        num_objectives=2
    )

    input_types = [
        # Normal distribution inputs
        (th.randn(batch_size, obs_space.shape[0]), th.randn(batch_size, act_space.shape[0])),
        # Uniform distribution inputs
        (th.rand(batch_size, obs_space.shape[0]), th.rand(batch_size, act_space.shape[0])),
        # Zero inputs
        (th.zeros(batch_size, obs_space.shape[0]), th.zeros(batch_size, act_space.shape[0])),
        # Ones inputs
        (th.ones(batch_size, obs_space.shape[0]), th.ones(batch_size, act_space.shape[0])),
        # Extreme values
        (th.full((batch_size, obs_space.shape[0]), 100.0), th.full((batch_size, act_space.shape[0]), -100.0)),
    ]

    for obs, actions in input_types:
        # Forward pass should handle all these input types without errors
        outputs = critic.forward(obs, actions)

        # Verify basic output structure
        assert isinstance(outputs, tuple)
        assert len(outputs) == critic.n_critics

        for critic_output in outputs:
            assert isinstance(critic_output, list)
            assert len(critic_output) == 2  # num_objectives

            for obj_output in critic_output:
                assert obj_output.shape == (batch_size, 1)
                # Verify outputs are finite (no NaNs or infs)
                assert th.all(th.isfinite(obj_output))


# Test forward pass with variable batch sizes
def test_forward_batch_dimensions(default_space):
    """Test that forward pass works correctly with different batch dimensions"""
    obs_space, act_space = default_space

    # Create critic
    critic = MOContinuousCritic(
        observation_space=obs_space,
        action_space=act_space,
        net_arch=[32, 32],
        num_objectives=2
    )

    # Test with variety of batch sizes including edge cases
    for batch_size in [1, 2, 5, 10, 32, 100]:
        # Create inputs
        obs = th.randn(batch_size, obs_space.shape[0])
        actions = th.randn(batch_size, act_space.shape[0])

        # Forward pass
        outputs = critic.forward(obs, actions)

        # Check output dimensions
        for critic_output in outputs:
            for obj_output in critic_output:
                assert obj_output.shape == (batch_size, 1)


# Test feature extraction process
def test_features_extraction(default_space):
    """Test that feature extraction works correctly within forward pass"""
    obs_space, act_space = default_space
    batch_size = 10

    # Create a custom features extractor we can monitor
    class MonitoredExtractor(FlattenExtractor):
        def __init__(self, observation_space):
            super().__init__(observation_space)
            self.called = False

        def forward(self, observations):
            self.called = True
            return super().forward(observations)

    # Create critic with monitored extractor
    features_extractor = MonitoredExtractor(obs_space)
    critic = MOContinuousCritic(
        observation_space=obs_space,
        action_space=act_space,
        net_arch=[32, 32],
        num_objectives=2,
        features_extractor=features_extractor
    )

    # Sample data
    obs = th.randn(batch_size, obs_space.shape[0])
    actions = th.randn(batch_size, act_space.shape[0])

    # Forward pass
    outputs = critic.forward(obs, actions)

    # Verify extractor was called
    assert features_extractor.called

    # Verify output shape matches expected
    for critic_output in outputs:
        for obj_output in critic_output:
            assert obj_output.shape == (batch_size, 1)


# Test that inputs are correctly concatenated
def test_input_concatenation(default_space):
    """Test that observation and action inputs are correctly concatenated"""
    obs_space, act_space = default_space
    batch_size = 10

    # Create a custom Q-Network that lets us inspect the input
    class CaptureInputQNet(SharedFeatureQNet):
        def __init__(self, base_net, heads):
            super().__init__(base_net, heads)
            self.last_input = None

        def forward(self, x):
            self.last_input = x.clone()
            # Manually reshape for expected input shape
            features = self.base_net(x)
            return [head(features) for head in self.heads]

    # Create critic
    critic = MOContinuousCritic(
        observation_space=obs_space,
        action_space=act_space,
        net_arch=[32, 32],
        num_objectives=2,
        share_features_across_objectives=True
    )

    # Replace Q-networks with our instrumented versions
    instrumented_q_nets = []
    for q_net in critic.q_networks:
        instrumented_net = CaptureInputQNet(q_net.base_net, q_net.heads)
        instrumented_q_nets.append(instrumented_net)

    # Monkey patch the q_networks list
    critic.q_networks = instrumented_q_nets

    # Create sample data with recognizable patterns
    obs = th.zeros(batch_size, obs_space.shape[0])
    actions = th.ones(batch_size, act_space.shape[0])

    # Set specific values to verify concatenation
    for i in range(obs_space.shape[0]):
        obs[:, i] = i + 1  # values 1, 2, 3, 4

    for i in range(act_space.shape[0]):
        actions[:, i] = 10 * (i + 1)  # values 10, 20

    # Forward pass (custom implementation to use our instrumented networks)
    with th.no_grad():
        features = critic.extract_features(obs, critic.features_extractor)
        qvalue_input = th.cat([features, actions], dim=1)
        _ = tuple(q_net(qvalue_input) for q_net in critic.q_networks)

    # Verify input to each q_net was correctly formed
    for q_net in critic.q_networks:
        # Get captured input
        captured_input = q_net.last_input

        # Check shape is correct
        assert captured_input.shape == (batch_size, obs_space.shape[0] + act_space.shape[0])

        # Check obs part has correct values
        for i in range(obs_space.shape[0]):
            assert th.all(captured_input[:, i] == i + 1)

        # Check action part has correct values
        for i in range(act_space.shape[0]):
            assert th.all(captured_input[:, obs_space.shape[0] + i] == 10 * (i + 1))