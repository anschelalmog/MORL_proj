import pytest
import torch as th
import numpy as np
from scalarize_algorithm.mosac_scalarized import MOContinuousCritic
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

# -----------------------------
# Test MOSACPolicy
# -----------------------------

def test_mosac_policy_action_sampling():
    actor = MOSACPolicy(
        observation_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=LATENT_DIM,
    )
    obs = th.rand(BATCH_SIZE, OBS_DIM)
    action, log_prob = actor(obs)
    assert action.shape == (BATCH_SIZE, ACTION_DIM)
    assert log_prob.shape == (BATCH_SIZE,)

# -----------------------------
# Test MOReplayBuffer
# -----------------------------

def test_moreplay_buffer_store_and_sample():
    buffer = MOReplayBuffer(buffer_size=10, observation_dim=OBS_DIM, action_dim=ACTION_DIM, reward_dim=REWARD_DIM)
    for _ in range(10):
        buffer.add(
            obs=np.random.rand(OBS_DIM),
            action=np.random.rand(ACTION_DIM),
            reward=np.random.rand(REWARD_DIM),
            next_obs=np.random.rand(OBS_DIM),
            done=np.random.choice([True, False])
        )
    sample = buffer.sample(batch_size=5)
    assert all(key in sample for key in ["observations", "actions", "rewards", "next_observations", "dones"])
    assert sample["observations"].shape == (5, OBS_DIM)
    assert sample["actions"].shape == (5, ACTION_DIM)
    assert sample["rewards"].shape == (5, REWARD_DIM)
    assert sample["next_observations"].shape == (5, OBS_DIM)
    assert sample["dones"].shape == (5,)

# -----------------------------
# Scalarized reward check (if applicable)
# -----------------------------

#def test_scalarized_reward():
#    from your_module import scalarize_reward
#    reward = np.array([[1.0, 2.0], [3.0, 4.0]])
#    weights = np.array([0.3, 0.7])
#    scalar = scalarize_reward(reward, weights)
#    expected = reward @ weights
#    assert np.allclose(scalar, expected)

