import pytest
import torch as th
import numpy as np
import gymnasium as gym
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.preprocessing import preprocess_obs

from algorithms.mosac import MOSAC, MOSACPolicy

@pytest.fixture
def dummy_env():
    return gym.make("Pendulum-v1")  # A continuous control env suitable for SAC

@pytest.fixture
def mosac_policy(dummy_env):
    observation_space = dummy_env.observation_space
    action_space = dummy_env.action_space

    policy = MOSACPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _: 1e-3,
        num_objectives=3,
        net_arch=[64, 64],
        activation_fn=th.nn.ReLU,
        use_sde=False,
        log_std_init=-2.0,
        normalize_images=False,
    )
    policy._build(lr_schedule=lambda _: 3e-4)
    return policy

def test_policy_initialization(mosac_policy):
    assert isinstance(mosac_policy.actor, th.nn.Module)
    assert isinstance(mosac_policy.critic, th.nn.Module)
    assert mosac_policy.num_objectives == 3


def test_actor_output_shape(mosac_policy, dummy_env):
    obs, _ = dummy_env.reset()
    obs_tensor = th.as_tensor(obs, dtype=th.float32).unsqueeze(0).to(get_device("auto"))

    # Manually build the policy if not already built
    if not hasattr(mosac_policy, "actor") or mosac_policy.actor is None:
        mosac_policy._build(lambda _: 1e-3)


    # Sample an action using the policy's predict method
    action, _ = mosac_policy.predict(obs, deterministic=False)

    # Check shape
    assert action.shape == (dummy_env.action_space.shape[0],)

def test_actor_output_shape_old(mosac_policy, dummy_env):


    obs, _ = dummy_env.reset()
    obs_tensor = th.as_tensor(obs, dtype=th.float32).unsqueeze(0).to(get_device("auto"))
    # Manually build the policy if it hasn’t been built already
    if not hasattr(mosac_policy, "features_extractor") or mosac_policy.features_extractor is None:
        mosac_policy._build(lambda _: 1e-3)

    # This uses the SACPolicy's logic to compute the distribution#features_extractor(obs)
    features = mosac_policy.extract_features(obs_tensor,mosac_policy.features_extractor)
    latent_pi, _ = mosac_policy.mlp_extractor(features)
    distribution = mosac_policy._get_action_dist_from_latent(latent_pi)

    action_sample = distribution.sample()

    assert action_sample.shape == (1, dummy_env.action_space.shape[0])

def test_critic_output_shapes(mosac_policy, dummy_env):
    obs, _ = dummy_env.reset()
    obs_tensor = th.as_tensor(obs, dtype=th.float32).unsqueeze(0).to(get_device("auto"))
    action = dummy_env.action_space.sample()
    action_tensor = th.as_tensor(action, dtype=th.float32).unsqueeze(0).to(get_device("auto"))

    critic_output = mosac_policy.critic(obs_tensor, action_tensor)
    assert isinstance(critic_output, list)
    assert all(isinstance(q_values, list) for q_values in critic_output)

    for critic in critic_output:
        assert len(critic) == mosac_policy.num_objectives
        for q in critic:
            assert q.shape == (1, 1)
