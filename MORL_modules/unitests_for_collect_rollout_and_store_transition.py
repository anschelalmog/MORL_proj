import numpy as np
import torch as th
import pytest
import gymnasium as gym
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List, Optional, Tuple, Union

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit, RolloutReturn

from agents.mobuffers import MOReplayBuffer
from agents.mosac import MOSAC
from agents.mo_env_wrappers import MODummyVecEnv, MultiObjectiveWrapper
from stable_baselines3.common.noise import ActionNoise, NormalActionNoise

class MultiObjectiveTestEnv(gym.Env):
    """Simple multi-objective test environment that returns vector rewards."""

    def __init__(self, num_objectives=4):
        super().__init__()
        self.num_objectives = num_objectives
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.current_step = 0
        self.max_steps = 10

    def reset(self, **kwargs):
        self.current_step = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1
        # Create a vector reward with different values for each objective
        reward = np.array([0.1 * i * (action[0] + 1) for i in range(1, self.num_objectives + 1)], dtype=np.float32)
        done = self.current_step >= self.max_steps

        # Return an observation, vector reward, done flag, and empty info dict
        return np.ones(4, dtype=np.float32) * self.current_step, reward, done, False, {}


class TestStoreTransition:
    """Tests for the _store_transition method in MOSAC."""

    def setup_method(self):
        """Setup for each test."""
        # Create environment and MOSAC instance
        env = MultiObjectiveTestEnv(num_objectives=4)
        env = MODummyVecEnv([lambda: env])

        self.num_objectives = 4
        self.mosac = MOSAC(
            policy="MOSACPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=1000,
            num_objectives=self.num_objectives,
            verbose=0,
            seed=0,
        )

        # Use the actual replay buffer from MOSAC
        self.replay_buffer = self.mosac.replay_buffer

    def test_store_transition_basic(self):
        """Test basic storage of a transition with vector rewards."""
        # Setup test data
        buffer_action = np.array([[0.5, -0.5]],  dtype=np.float32)
        new_obs = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        reward = np.array([0.1, 0.2, 0.3, 0.4],  dtype=np.float32)
        dones = np.array([False])
        infos = [{}]

        # Set last observation
        self.mosac._last_obs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.mosac._last_original_obs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Record initial position in buffer
        initial_pos = self.replay_buffer.pos

        # Call the method
        self.mosac._store_transition(
            self.replay_buffer, buffer_action, new_obs, reward, dones, infos
        )

        # Verify data was added to buffer (position should have incremented)
        assert self.replay_buffer.pos == (initial_pos + 1) % self.replay_buffer.buffer_size

        # Get the stored data
        idx = initial_pos
        stored_obs = self.replay_buffer.observations[idx]
        stored_next_obs = self.replay_buffer.next_observations[idx] if not self.replay_buffer.optimize_memory_usage else \
        self.replay_buffer.observations[(idx + 1) % self.replay_buffer.buffer_size]
        stored_action = self.replay_buffer.actions[idx]
        stored_reward = self.replay_buffer.rewards[idx]
        stored_done = self.replay_buffer.dones[idx]

        # Verify stored data
        breakpoint()
        np.testing.assert_array_equal(stored_obs[0], self.mosac._last_original_obs[0])
        np.testing.assert_array_equal(stored_next_obs[0], new_obs[0])
        np.testing.assert_array_equal(stored_action[0], buffer_action[0])
        np.testing.assert_array_equal(stored_reward[0], reward)
        np.testing.assert_array_equal(stored_done[0], dones[0])

        # Check that _last_obs was updated
        np.testing.assert_array_equal(self.mosac._last_obs, new_obs)

    def test_store_transition_with_terminal_obs(self):
        """Test storing a transition with terminal observation."""
        # Setup test data
        buffer_action = np.array([[0.5, -0.5]])
        new_obs = np.array([[1.0, 2.0, 3.0, 4.0]])
        reward = np.array([[0.1, 0.2, 0.3, 0.4]])
        dones = np.array([True])
        terminal_obs = np.array([5.0, 6.0, 7.0, 8.0])
        infos = [{"terminal_observation": terminal_obs}]

        # Set last observation
        self.mosac._last_obs = np.array([[0.0, 0.0, 0.0, 0.0]])
        self.mosac._last_original_obs = np.array([[0.0, 0.0, 0.0, 0.0]])

        # Record initial position in buffer
        initial_pos = self.replay_buffer.pos

        # Call the method
        self.mosac._store_transition(
            self.replay_buffer, buffer_action, new_obs, reward, dones, infos
        )

        # Get the stored data
        idx = initial_pos
        stored_next_obs = self.replay_buffer.next_observations[idx] if not self.replay_buffer.optimize_memory_usage else \
        self.replay_buffer.observations[(idx + 1) % self.replay_buffer.buffer_size]

        # Verify that the next_obs is the terminal observation
        np.testing.assert_array_equal(stored_next_obs[0], terminal_obs)

    def test_store_transition_with_vec_normalize(self):
        """Test storing a transition with VecNormalize wrapper."""
        # Create a VecNormalize environment
        env = MultiObjectiveTestEnv(num_objectives=4)
        env = MODummyVecEnv([lambda: env])
        vec_normalize_env = VecNormalize(env)

        # Setup MOSAC with VecNormalize
        self.mosac._vec_normalize_env = vec_normalize_env

        # Mock the get_original methods
        original_obs = np.array([[10.0, 20.0, 30.0, 40.0]])
        original_reward = np.array([[1.0, 2.0, 3.0, 4.0]])

        with patch.object(vec_normalize_env, 'get_original_obs', return_value=original_obs), \
                patch.object(vec_normalize_env, 'get_original_reward', return_value=original_reward):
            # Setup test data
            buffer_action = np.array([[0.5, -0.5]])
            new_obs = np.array([[1.0, 2.0, 3.0, 4.0]])  # Normalized observation
            reward = np.array([[0.1, 0.2, 0.3, 0.4]])  # Normalized reward
            dones = np.array([False])
            infos = [{}]

            # Set last observation
            self.mosac._last_obs = np.array([[0.0, 0.0, 0.0, 0.0]])
            self.mosac._last_original_obs = np.array([[0.0, 0.0, 0.0, 0.0]])

            # Record initial position in buffer
            initial_pos = self.replay_buffer.pos

            # Call the method
            self.mosac._store_transition(
                self.replay_buffer, buffer_action, new_obs, reward, dones, infos
            )

            # Get the stored data
            idx = initial_pos
            stored_next_obs = self.replay_buffer.next_observations[
                idx] if not self.replay_buffer.optimize_memory_usage else self.replay_buffer.observations[
                (idx + 1) % self.replay_buffer.buffer_size]
            stored_reward = self.replay_buffer.rewards[idx]

            # Verify original values were stored
            np.testing.assert_array_equal(stored_next_obs[0], original_obs[0])
            np.testing.assert_array_equal(stored_reward[0], original_reward[0])

            # Check that _last_obs was updated with normalized obs
            np.testing.assert_array_equal(self.mosac._last_obs, new_obs)
            # And _last_original_obs with original
            np.testing.assert_array_equal(self.mosac._last_original_obs, original_obs)


class TrackingCallback(BaseCallback):
    """Callback that tracks calls for testing purposes."""

    def __init__(self, verbose=0, env: Optional[gym.Env] = None):
        super().__init__(verbose)
        self.on_step_calls = 0
        self.on_rollout_start_calls = 0
        self.on_rollout_end_calls = 0
        self.stop_after_steps = None

    def _on_step(self) -> bool:
        self.on_step_calls += 1
        return False if self.stop_after_steps and self.on_step_calls >= self.stop_after_steps else True

    def on_rollout_start(self):
        super().on_rollout_start()
        self.on_rollout_start_calls += 1

    def on_rollout_end(self):
        super().on_rollout_end()
        self.on_rollout_end_calls += 1


class TestCollectRollouts:
    """Tests for the collect_rollouts method in MOSAC."""

    def setup_method(self):
        """Setup for each test."""
        # Create environment and MOSAC instance
        self.num_objectives = 4
        env = MultiObjectiveTestEnv(num_objectives=self.num_objectives)
        env = MODummyVecEnv([lambda: env])

        self.mosac = MOSAC(
            policy="MOSACPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=1000,
            num_objectives=self.num_objectives,
            verbose=0,
            seed=0,
        )

        # Use the actual replay buffer
        self.replay_buffer = self.mosac.replay_buffer

        # Create a tracking callback
        self.callback = TrackingCallback()
        self.mosac._setup_learn(10,self.callback)

        # Create a mock for _sample_action method
        self.orig_sample_action = self.mosac._sample_action

    def teardown_method(self):
        """Cleanup after each test."""
        # Restore original methods
        self.mosac._sample_action = self.orig_sample_action

    def test_collect_single_step(self):
        """Test collecting a single step of experience."""
        # Mock the _sample_action method to return predictable actions
        actions = np.array([[0.5, -0.5]])
        buffer_actions = np.array([[0.5, -0.5]])

        def mock_sample_action(learning_starts, action_noise, n_envs):
            return actions, buffer_actions

        self.mosac._sample_action = mock_sample_action

        # Set training mode to False (happens in collect_rollouts)
        # Record initial buffer position
        initial_pos = self.replay_buffer.pos

        # Call collect_rollouts for 1 step
        train_freq = TrainFreq(1, TrainFrequencyUnit.STEP)
        result = self.mosac.collect_rollouts(
            env=self.mosac.env,
            callback=self.callback,
            train_freq=train_freq,
            replay_buffer=self.replay_buffer,
            action_noise=None,
            learning_starts=0,
        )

        # Verify callback methods were called
        assert self.callback.on_rollout_start_calls == 1
        assert self.callback.on_step_calls == 1
        assert self.callback.on_rollout_end_calls == 1

        # Verify result

        assert result[0] == 1  # num_collected_steps * env.num_envs
        assert result[1] == 0  # num_collected_episodes (no episodes completed)
        assert result[2] == True  # continue_training

        # Verify data was added to buffer
        assert self.replay_buffer.pos == (initial_pos + 1) % self.replay_buffer.buffer_size

        # Get the stored reward
        stored_reward = self.replay_buffer.rewards[initial_pos]
        # Verify it's a vector reward with correct dimensions
        assert stored_reward[0].shape == (self.num_objectives,)

    def test_collect_full_episode(self):
        """Test collecting a full episode of experience."""
        # Mock the _sample_action method to return predictable actions
        actions = np.array([[0.5, -0.5]])
        buffer_actions = np.array([[0.5, -0.5]])

        def mock_sample_action(learning_starts, action_noise, n_envs):
            return actions, buffer_actions

        self.mosac._sample_action = mock_sample_action

        # Record initial buffer position and episode count
        initial_pos = self.replay_buffer.pos
        initial_episodes = self.mosac._episode_num

        # Call collect_rollouts for enough steps to complete an episode
        # (Our test env terminates after 10 steps)
        train_freq = TrainFreq(15, TrainFrequencyUnit.STEP)  # More than episode length
        result = self.mosac.collect_rollouts(
            env=self.mosac.env,
            callback=self.callback,
            train_freq=train_freq,
            replay_buffer=self.replay_buffer,
            action_noise=None,
            learning_starts=0,
        )

        # Verify callback methods were called
        assert self.callback.on_rollout_start_calls == 1
        assert self.callback.on_step_calls == 15
        assert self.callback.on_rollout_end_calls == 1

        # Verify result shows at least one episode completed

        assert result[0] == 15  # num_collected_steps * env.num_envs
        assert result[1] >= 1  # num_collected_episodes (no episodes completed)
        assert result[2] == result.continue_training # continue_training

        # Verify episode counter incremented
        assert self.mosac._episode_num > initial_episodes

        # Verify buffer was filled with vector rewards
        for i in range(15):
            idx = (initial_pos + i) % self.replay_buffer.buffer_size
            stored_reward = self.replay_buffer.rewards[idx]
            assert stored_reward[0].shape == (self.num_objectives,)

    def test_collect_with_action_noise(self):
        """Test collecting with action noise."""
        # Create a real action noise object
        action_noise = NormalActionNoise(
            mean=np.zeros(self.mosac.action_space.shape),
            sigma=0.1 * np.ones(self.mosac.action_space.shape)
        )

        # Spy on the reset method
        original_reset = action_noise.reset
        reset_calls = []

        def spy_reset(**kwargs):
            reset_calls.append(kwargs)
            return original_reset(**kwargs)

        action_noise.reset = spy_reset

        # Call collect_rollouts with action noise for full episode
        train_freq = TrainFreq(20, TrainFrequencyUnit.STEP)  # Enough for 2 episodes
        self.mosac.collect_rollouts(
            env=self.mosac.env,
            callback=self.callback,
            train_freq=train_freq,
            replay_buffer=self.replay_buffer,
            action_noise=action_noise,
            learning_starts=0,
        )

        # Verify noise reset was called when episodes completed
        assert len(reset_calls) >= 1

    def test_collect_with_early_stop(self):
        """Test early stopping via callback."""
        # Set callback to stop after 5 steps
        self.callback.stop_after_steps = 5

        # Call collect_rollouts for more steps than callback allows
        train_freq = TrainFreq(10, TrainFrequencyUnit.STEP)
        result = self.mosac.collect_rollouts(
            env=self.mosac.env,
            callback=self.callback,
            train_freq=train_freq,
            replay_buffer=self.replay_buffer,
            action_noise=None,
            learning_starts=0,
        )

        # Verify early stopping
        assert self.callback.on_step_calls <= 5
        assert not result[2]

    def test_collect_with_learning_starts(self):
        """Test collection with learning_starts > 0."""
        # Track _sample_action calls
        sample_calls = []

        def track_sample(learning_starts, action_noise, n_envs):
            sample_calls.append(learning_starts)
            return np.array([[0.5, -0.5]]), np.array([[0.5, -0.5]])

        self.mosac._sample_action = track_sample

        # Call collect_rollouts with learning_starts > 0
        learning_starts = 100
        train_freq = TrainFreq(5, TrainFrequencyUnit.STEP)
        self.mosac.collect_rollouts(
            env=self.mosac.env,
            callback=self.callback,
            train_freq=train_freq,
            replay_buffer=self.replay_buffer,
            action_noise=None,
            learning_starts=learning_starts,
        )

        # Verify learning_starts was passed to _sample_action
        for ls in sample_calls:
            assert ls == learning_starts


def test_integration():
    """Integration test combining both methods."""
    # Create environment and MOSAC instance
    num_objectives = 4
    env = MultiObjectiveTestEnv(num_objectives=num_objectives)
    env = MODummyVecEnv([lambda: env])

    mosac = MOSAC(
        policy="MOSACPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=1000,
        num_objectives=num_objectives,
        verbose=0,
        seed=0,
    )

    # Create a proper SB3 callback
    callback = CallbackList([
        CheckpointCallback(save_freq=100, save_path="./logs/", name_prefix="mosac_model"),
        TrackingCallback()
    ])

    # Initialize the callback
    callback.init_callback(mosac)

    # Collect some experience
    train_freq = TrainFreq(20, TrainFrequencyUnit.STEP)
    mosac.collect_rollouts(
        env=mosac.env,
        callback=callback,
        train_freq=train_freq,
        replay_buffer=mosac.replay_buffer,
        action_noise=None,
        learning_starts=0,
    )

    # Check the replay buffer has data with correct shapes
    assert mosac.replay_buffer.full or mosac.replay_buffer.pos > 0

    # Sample from the buffer to check the vector rewards
    if mosac.replay_buffer.pos > 0:  # Ensure there's data to sample
        sample = mosac.replay_buffer.sample(batch_size=5)

        # Verify shapes of sample components
        assert sample.rewards.shape[1] == num_objectives  # Should be (batch_size, num_objectives)

        # Try training on the collected data
        mosac.train(gradient_steps=5, batch_size=5)