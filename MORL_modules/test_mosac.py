# test_mosac.py
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env
from scalarize_algorithm.mosac_scalarized import MOSAC, MOSACPolicy

import gym
import numpy as np
from gym import spaces
import gym
import torch
from stable_baselines3.common.utils import get_device

class VectorRewardWrapper(gym.Wrapper):
    def __init__(self, env, num_objectives=3):
        super().__init__(env)
        self.num_objectives = num_objectives
        self.reward_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_objectives,), dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Simulate vector reward: here just duplicating scalar reward for demonstration
        vector_reward = np.array([reward + i for i in range(self.num_objectives)], dtype=np.float32)

        return obs, vector_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)



def test_mosac_initialization_and_prediction():


    env = gym.make("Pendulum-v1")

    model = MOSAC(
        policy=MOSACPolicy,
        env=env,
        num_objectives=3,
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
            num_objectives=3,
            share_features_across_objectives=True,
        ),
        verbose=1
    )

    # Check device
    assert model.device == torch.device("cpu") or model.device.type == "cuda"

    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert env.action_space.contains(action), "Predicted action not in action space"

    for _ in range(110):
        # Save current obs
        current_obs = obs
        action, _ = model.predict(current_obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        model.replay_buffer.add(current_obs, action, reward, next_obs, done, {})

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

    model.train(gradient_steps=1)


