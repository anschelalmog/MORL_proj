# test_mosac.py

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env
from scalarize_algorithm.mosac_scalarized import MOSAC, MOSACPolicy


def test_mosac_initialization_and_prediction():
    # Dummy continuous environment
    env = gym.make("Pendulum-v1")

    # Create MOSAC model
    model = MOSAC(
        policy=MOSACPolicy,
        env=env,
        learning_rate=1e-3,
        buffer_size=1000,
        batch_size=32,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        learning_starts=100,
        policy_kwargs=dict(
            net_arch=[64, 64],
            num_objectives=2,
            share_features_across_objectives=True,
        ),
        verbose=1
    )

    # Check model is on correct device
    assert model.device == torch.device("cpu") or model.device.type == "cuda"

    # Collect one step of experience
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert env.action_space.contains(action), "Predicted action not in action space"

    # Perform one training step after filling buffer minimally
    for _ in range(110):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            obs, _ = env.reset()
        model.replay_buffer.add(obs, action, reward, obs, done)

    model.train(gradient_steps=1)
