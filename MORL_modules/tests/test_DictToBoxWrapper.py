import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
import os
from typing import Dict, Tuple, Any, Optional, Union
from MORL_modules.wrappers.mo_pcs_wrapper import MOPCSWrapper
import pdb
# Import the wrapper (or include it directly in this file)
from MORL_modules.wrappers.dict_to_box_wrapper import DictToBoxWrapper


# Create a custom environment with Dict spaces for testing
class DictEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            'velocity': spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32),
            'sensors': spaces.Box(low=0, high=100, shape=(10,), dtype=np.float32)
        })

        self.action_space = spaces.Dict({
            'motor': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            'brake': spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32)
        })

        self.state = {
            'position': np.zeros(3, dtype=np.float32),
            'velocity': np.zeros(3, dtype=np.float32),
            'sensors': np.zeros(10, dtype=np.float32)
        }
        self.steps = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        # Initialize with random values
        self.state = {
            'position': np.random.uniform(-0.5, 0.5, size=(3,)).astype(np.float32),
            'velocity': np.random.uniform(-1.0, 1.0, size=(3,)).astype(np.float32),
            'sensors': np.random.uniform(40, 60, size=(10,)).astype(np.float32)
        }

        return self.state.copy(), {}

    def step(self, action):
        # Update state based on action
        motor = action['motor']
        brake = action['brake']

        # Simple dynamics (for demonstration)
        acceleration = motor.sum() - brake.sum() * 0.5

        # Update position and velocity
        self.state['velocity'] += acceleration * 0.1
        self.state['velocity'] = np.clip(self.state['velocity'], -2.0, 2.0)
        self.state['position'] += self.state['velocity'] * 0.1
        self.state['position'] = np.clip(self.state['position'], -1.0, 1.0)

        # Update sensors (random for simplicity)
        self.state['sensors'] = np.random.uniform(40, 60, size=(10,)).astype(np.float32)

        # Calculate reward (simple example - try to reach position [0.5, 0.5, 0.5])
        target_position = np.array([0.5, 0.5, 0.5])
        distance = np.linalg.norm(self.state['position'] - target_position)
        reward = 1.0 - distance  # Higher reward for getting closer to target

        # Check if episode is done
        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_steps

        return self.state.copy(), reward, terminated, truncated, {}


def test_sac_with_dict_wrapper():
    breakpoint()
    """Test training and evaluation of SAC on a wrapped environment."""
    print("Creating environment...")
    env = DictEnv()

    # Wrap the environment to convert Dict spaces to Box spaces
    print("Wrapping environment with DictToBoxWrapper...")
    wrapped_env = DictToBoxWrapper(env)

    # Check the spaces
    print(f"Original observation space: {env.observation_space}")
    print(f"Wrapped observation space: {wrapped_env.observation_space}")
    print(f"Original action space: {env.action_space}")
    print(f"Wrapped action space: {wrapped_env.action_space}")

    # Create the SAC agent
    print("Creating SAC agent...")
    model = SAC(
        "MlpPolicy",
        wrapped_env,
        learning_rate=3e-4,
        buffer_size=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        verbose=1
    )

    # Train the agent
    print("Training SAC agent...")
    model.learn(total_timesteps=10)

    # Save the model
    model_path = "sac_dict_env"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Evaluate the trained agent
    print("Evaluating the trained agent...")
    mean_reward, std_reward = evaluate_policy(model, wrapped_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Test the model with a few episodes
    print("\nRunning 3 test episodes with trained model:")
    for i in range(3):
        obs, _ = wrapped_env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = wrapped_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        print(f"Episode {i + 1} reward: {episode_reward:.2f}")


if __name__ == "__main__":
    test_sac_with_dict_wrapper()