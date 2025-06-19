# M O R L _modules/tests/test_MOPCSWrapper.py

import sys
import os
import logging

import pytest
import numpy as np
from gymnasium import spaces

# make sure we can import the wrapper
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wrappers.mo_pcs_wrapper import MOPCSWrapper


class DummyController:
    """
    Minimal controller stub to satisfy MOPCSWrapper._get_environment_components
    and reward computations.
    """
    def __init__(self):
        # pretend PCSUnit and BatteryManager are the same object here
        self.pcsunit = self
        self.battery = self
        self.battery_manager = self

        # battery bounds
        self.battery_min = 0.0
        self.battery_max = 100.0

    def get_level(self):
        # fixed battery state of charge
        return 50.0

    def get_self_production(self):
        # stub production for autonomy tests
        return 5.0

    def get_self_consumption(self):
        # stub consumption for autonomy tests
        return 10.0


class DummyEnv(spaces.Space):
    """
    A minimal Gym-style env that MOPCSWrapper can wrap.
    - unwrapped.controller must exist
    - reset() returns ([iso_obs, pcs_obs], info)
    - step() returns ([iso_obs, pcs_obs], reward, done, truncated, info)
    """
    def __init__(self):
        # make our dummy controller visible at env.unwrapped.controller
        self.unwrapped = self
        self.controller = DummyController()

        # define action and observation spaces so .sample() works
        box = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Dict({"iso": box, "pcs": box})
        self.observation_space = spaces.Dict({
            "iso": spaces.Box(0.0, 1.0, (1,), np.float32),
            "pcs": spaces.Box(0.0, 100.0, (1,), np.float32),
        })

    def reset(self, **kwargs):
        # iso sees [time], pcs sees [level]
        return [np.array([0.0]), np.array([50.0])], {}

    def step(self, action):
        # always return reward=1.0, no termination, and stub info for grid
        info = {
            "net_exchange": 2.0,
            "iso_buy_price": 3.0,
            "iso_sell_price": 1.0
        }
        return [np.array([0.0]), np.array([50.0])], 1.0, False, False, info


@pytest.fixture
def dummy_env():
    return DummyEnv()

@pytest.fixture
def wrapper(dummy_env):
    # no normalization, uniform weights
    return MOPCSWrapper(dummy_env,
                       num_objectives=4,
                       reward_weights=np.ones(4)/4,
                       normalize_rewards=False,
                       log_level=logging.INFO)


def test_initialization_and_configuration(wrapper):
    # 1. Initialization & Configuration
    assert wrapper.num_objectives == 4
    assert np.allclose(wrapper.reward_weights, [0.25, 0.25, 0.25, 0.25])
    assert wrapper.normalize_rewards is False

    # logger at INFO and has at least one handler
    assert wrapper.logger.level == logging.INFO
    assert len(wrapper.logger.handlers) > 0


def test_environment_component_extraction(wrapper):
    # 2. Environment Component Extraction
    wrapper._get_environment_components()
    # ensure we pulled from dummy_env.controller
    assert wrapper.controller is wrapper.env.unwrapped.controller
    assert wrapper.pcsunit is wrapper.controller.pcsunit
    assert wrapper.battery is wrapper.controller.battery
    assert wrapper.battery_manager is wrapper.controller.battery_manager

    # simulate missing components
    # remove controller and re-run
    del wrapper.env.unwrapped.controller
    # should not raise
    wrapper._get_environment_components()
    # controller now None
    assert wrapper.controller is None


def test_multi_objective_reward_computation(wrapper):
    # 3. Multi-Objective Reward Computation (raw, no normalization)
    obs, info = wrapper.reset()
    action = {"iso": np.array([0.0]), "pcs": np.array([0.0])}
    _, mo_rewards, terminated, truncated, info2 = wrapper.step(action)

    # economic = base reward = 1.0
    # battery health = +0.1 (in optimal 30–70%) + 0 cycling penalty
    # grid support = 0.0 (history too short to compute)
    # autonomy = 0.5 (5/10)
    expected = np.array([1.0, 0.1, 0.0, 0.5])
    assert np.allclose(mo_rewards, expected, atol=1e-6)


def test_battery_health_edge_cases(wrapper):
    # 4. Battery Health Objective Specifically
    wrapper._get_environment_components()

    # edge: no prev level
    wrapper.prev_battery_level = None
    assert wrapper._compute_battery_health_reward({}) == 0.0

    # cycling penalty + optimal range reward
    wrapper.prev_battery_level = 40.0
    # monkey‐patch get_level() to simulate a jump
    orig = wrapper.battery_manager.get_level
    wrapper.battery_manager.get_level = lambda: 50.0
    raw = wrapper._compute_battery_health_reward({})
    # state_change=10 -> penalty = -0.1*(10/10) = -0.1; range_reward=+0.1 => total 0.0
    assert pytest.approx(raw, abs=1e-6) == 0.0
    wrapper.battery_manager.get_level = orig


def test_grid_support_objective(wrapper):
    # 5. Grid Support Objective
    wrapper._get_environment_components()

    # need at least 10 history entries
    wrapper.price_history = [2.0] * 10
    info = {"iso_buy_price": 4.0, "iso_sell_price": 2.0, "net_exchange": 5.0}
    raw = wrapper._compute_grid_support_reward(info)
    # avg_price=3, baseline=2 => dev=0.5 => raw = -5*0.5/5 = -0.5
    assert pytest.approx(raw, abs=1e-6) == -0.5

    # missing keys => zero
    assert wrapper._compute_grid_support_reward({}) == 0.0


def test_energy_autonomy_objective(wrapper):
    # 6. Energy Autonomy
    wrapper._get_environment_components()

    # normal case: production < consumption
    wrapper.pcsunit.get_self_production = lambda: 15.0
    wrapper.pcsunit.get_self_consumption = lambda: 10.0
    # ratio=1.0 + excess_bonus=min(0.2,5/10*0.1=0.05)=0.05 => 1.05 clipped to 1.0
    assert pytest.approx(wrapper._compute_autonomy_reward({}), abs=1e-6) == 1.0

    # zero consumption & production
    wrapper.pcsunit.get_self_production = lambda: 0.0
    wrapper.pcsunit.get_self_consumption = lambda: 0.0
    assert wrapper._compute_autonomy_reward({}) == 1.0

    # missing methods => fallback 0.0
    del wrapper.pcsunit.get_self_production
    del wrapper.pcsunit.get_self_consumption
    assert wrapper._compute_autonomy_reward({}) == 0.0


def test_episode_tracking_and_statistics(wrapper):
    # 7. Episode Tracking & Statistics
    wrapper._get_environment_components()

    # run one episode
    wrapper.reset()
    action = {"iso": np.array([0.0]), "pcs": np.array([0.0])}
    wrapper.step(action)

    # wrap up that episode
    wrapper.reset()

    # should have recorded 1 episode
    stats = wrapper.get_episode_statistics()
    assert "economic" in stats
    assert stats["economic"]["episodes"] == 1
    # mean == the single recorded value
    rec = wrapper.episode_rewards["economic"][0]
    assert stats["economic"]["mean"] == pytest.approx(rec, abs=1e-6)


def test_integration_reset_and_step_api(wrapper):
    # 8. Integration with Environment
    obs, info = wrapper.reset()
    # obs is a dict of arrays
    assert set(obs.keys()) == {"iso", "pcs"}

    # a valid sample action
    iso_act = wrapper.env.action_space["iso"].sample()
    pcs_act = wrapper.env.action_space["pcs"].sample()
    obs2, rewards, done, trunc, info2 = wrapper.step({"iso": iso_act, "pcs": pcs_act})
    # all four objective rewards
    assert isinstance(rewards, np.ndarray) and rewards.shape == (4,)
    # terminated/truncated are bools
    assert isinstance(done, bool)
    assert isinstance(trunc, bool)
    # info2 inherited from DummyEnv.step plus MO info
    assert "mo_rewards" in info2
    assert len(info2["mo_rewards"]) == 4


def test_error_handling_and_edge_cases(wrapper):
    # 9. Error Handling & Edge Cases

    # missing battery_manager entirely
    wrapper._get_environment_components()
    wrapper.battery_manager = None
    wrapper.prev_battery_level = 50.0
    # should bail to fallback and not exception
    assert wrapper._compute_battery_health_reward({}) == 0.0

    # division-by-zero safety in autonomy
    # stub consumption = 0
    class C:
        pass
    # give pcsunit but only production stub
    wrapper.pcsunit = C()
    wrapper.pcsunit.get_self_production = lambda: 100.0
    # no get_self_consumption
    assert wrapper._compute_autonomy_reward({}) == 0.0

