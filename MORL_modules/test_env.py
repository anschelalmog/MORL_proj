import unittest
import numpy as np
import gym
from gym import spaces
from numpy.testing import assert_equal


# Including the environment class so the test file is self-contained
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


class TestDummyMultiObjectiveEnv(unittest.TestCase):

    def setUp(self):
        self.env = DummyMultiObjectiveEnv(num_objectives=3)

    def test_reset(self):
        # Test with default parameters
        self.env.current_step = 50  # Set a non-zero step count
        observation, info = self.env.reset()

        # Check if step counter was reset
        self.assertEqual(self.env.current_step, 0,
                         "reset() should set current_step to 0")

        # Check if observation has the correct shape and type
        self.assertEqual(observation.shape, (4,),
                         "Observation should have shape (4,)")
        self.assertEqual(observation.dtype, np.float32,
                         "Observation should have dtype np.float32")

        # Check if observation values are within the defined space bounds
        self.assertTrue(np.all(observation >= -10) and np.all(observation <= 10),
                        "Observation values should be within bounds [-10, 10]")

        # Check if info is an empty dictionary
        self.assertEqual(info, {}, "Info should be an empty dictionary")

    def test_reset_with_seed(self):
        # Test with seed parameter for reproducibility
        seed = 42
        obs1, _ = self.env.reset(seed=seed)

        # Reset again with the same seed
        self.env = DummyMultiObjectiveEnv(num_objectives=3)  # Create
