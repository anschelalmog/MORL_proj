import unittest
import numpy as np
import torch as th
from unittest.mock import MagicMock


# Import your MOSAC and dependencies here
from your_module import MOSAC, MOReplayBuffer

class DummyVecEnv:
    def __init__(self, num_envs, obs_shape, action_shape, num_objectives):
        self.num_envs = num_envs
        self.observation_space = MagicMock()
        self.action_space = MagicMock()
        self.action_space.sample = lambda: np.zeros(action_shape)
        self.num_objectives = num_objectives
        self._step_count = 0
        self.max_steps = 5
        self.rewards = np.ones((num_envs, num_objectives))
        self.observations = np.zeros((num_envs, obs_shape))
        self._terminated = np.zeros(num_envs, dtype=bool)
        self._truncated = np.zeros(num_envs, dtype=bool)

    def step(self, actions):
        # Always return the same obs and reward, terminate after max_steps
        self._step_count += 1
        terminated = self._terminated.copy()
        truncated = self._truncated.copy()
        if self._step_count >= self.max_steps:
            terminated[:] = True
        infos = [{} for _ in range(self.num_envs)]
        return self.observations, self.rewards, terminated, truncated, infos

    def reset(self):
        self._step_count = 0
        return self.observations


class DummyReplayBuffer:
    def __init__(self):
        self.added = []

    def add(self, *args, **kwargs):
        self.added.append((args, kwargs))


class DummyCallback:
    def on_rollout_start(self): pass

    def update_locals(self, locals_): pass

    def on_step(self): return True

    def on_rollout_end(self): return True


class MOSACMock(MOSAC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Minimal setup for testing
        self.policy = MagicMock()
        self.policy.set_training_mode = MagicMock()
        self._last_obs = np.zeros((kwargs['env'].num_envs, 4))
        self.action_space = MagicMock()
        self.action_space.sample = lambda: np.zeros(2)
        self.num_envs = kwargs['env'].num_envs
        self._total_timesteps = 100
        self._vec_normalize_env = None
        self._last_original_obs = self._last_obs
        self.device = 'cpu'
        # For test, just return random actions
        self.predict = MagicMock(return_value=(np.zeros((self.num_envs, 2)), None))
        self._prepare_action = lambda a: a
        self._add_noise_to_action = lambda a, n, e: a
        self._extract_mo_rewards = lambda tup: np.ones((self.num_envs, self.num_objectives))
        self._update_info_buffer = lambda infos, terminated: None
        self._store_transition = MagicMock()
        self._get_episode_rewards_timesteps = lambda t, tr, i: ([1.0], [1])
        self._update_pareto_front = lambda r: None
        self._update_current_progress_remaining = lambda now, total: None
        self._on_step = lambda: None
        self.logger = MagicMock()
        self.logger.record = MagicMock()


def should_collect_more(train_freq, num_collected_steps, num_collected_episodes):
    # Stop after 3 steps for test
    return num_collected_steps < 3


class TestMOSACRollout(unittest.TestCase):
    def test_collect_rollouts_multi_objective(self):
        num_envs = 2
        num_objectives = 3
        obs_shape = 4
        action_shape = 2
        env = DummyVecEnv(num_envs, obs_shape, action_shape, num_objectives)
        replay_buffer = DummyReplayBuffer()
        callback = DummyCallback()
        mosac = MOSACMock(
            env=env,
            policy='MlpPolicy',
            learning_rate=0.01,
            buffer_size=100,
            batch_size=8,
            num_objectives=num_objectives,
            preference_weights=[1, 1, 1],
        )
        # Patch should_collect_more inside the method's scope
        mosac.collect_rollouts.__globals__['should_collect_more'] = should_collect_more

        result = mosac.collect_rollouts(
            env=env,
            callback=callback,
            train_freq=1,
            replay_buffer=replay_buffer,
            action_noise=None,
            learning_starts=0,
            log_interval=1,
        )

        # Check that _store_transition was called and vector rewards handled
        self.assertTrue(mosac._store_transition.called)
        # Check the result type and fields
        self.assertTrue(hasattr(result, 'reward'))
        self.assertTrue(hasattr(result, 'steps'))
        self.assertTrue(hasattr(result, 'episodes'))
        self.assertTrue(hasattr(result, 'continue_training'))

        # Check that rewards per objective were collected
        for episode_rewards in mosac._episode_mo_rewards:
            self.assertTrue(len(episode_rewards) > 0)

        # Check logger was called for each objective
        for i in range(num_objectives):
            mosac.logger.record.assert_any_call(f"metrics/objective_{i}_mean_reward", unittest.mock.ANY)


if __name__ == "__main__":
    unittest.main()