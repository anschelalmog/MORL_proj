import pytest
import torch as th
import numpy as np
from scalarize_algorithm.mosac_scalarized  import MOContinuousCritic, MOSACPolicy,  MOReplayBuffer
import gymnasium as gym

def test_mo_continuous_critic_forward_shapes():
    obs_dim = 8
    action_dim = 3
    reward_dim = 2
    hidden_dim = 64
    batch_size = 4

    # Use numpy float32 dtype for Gym spaces
    observation_space = gym.spaces.Box(low=-1, high=1, shape=(obs_dim,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)

    # Initialize critic with explicit features_dim since we have no extractor
    critic = MOContinuousCritic(
        observation_space=observation_space,
        action_space=action_space,
        net_arch=[hidden_dim, hidden_dim],
        num_objectives=reward_dim,
        features_extractor_class=None,
        features_extractor_kwargs=None,
        share_features_extractor=True,
        n_critics=2,
        activation_fn=th.nn.ReLU,
        normalize_images=False,
        share_features_across_objectives=True,
        #features_dim=obs_dim  # Explicitly pass feature dimension
    )

    # Dummy input tensors
    obs = th.rand((batch_size, obs_dim))
    actions = th.rand((batch_size, action_dim))

    # Check output shape of each Q-network
    for q_net in critic.q_networks:
        q_vals = q_net(obs, actions)
        assert isinstance(q_vals, list)
        assert len(q_vals) == reward_dim
        for q in q_vals:
            assert q.shape == (batch_size, 1)



