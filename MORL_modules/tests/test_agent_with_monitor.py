import sys
from typing import Dict, Any, List, Tuple
import pdb
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'MORL_modules'))
print(project_root)
print(current_dir)
from agents.mosac import MOSAC, MOSACPolicy, MOContinuousCritic
from agents.mobuffers import MOReplayBuffer as MOReplayBuffer
from agents.monets import SharedFeatureQNet, SeparateQNet
from agents.mo_env_wrappers import MODummyVecEnv, MultiObjectiveWrapper
from wrappers.mo_pcs_wrapper import MOPCSWrapper
from wrappers.scalarized_mo_pcs_wrapper import ScalarizedMOPCSWrapper
from wrappers.dict_to_box_wrapper import DictToBoxWrapper
from energy_net.envs.energy_net_v0 import EnergyNetV0

from energy_net.market.pricing.pricing_policy import PricingPolicy
from energy_net.market.pricing.cost_types import CostType
from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern

from utils.utils import moving_average, plot_results_scalarized
from utils.callbacks import SaveOnBestTrainingRewardCallback

def create_energynet_env(**kwargs):
    """Create EnergyNet environment."""
    from energy_net.envs.energy_net_v0 import EnergyNetV0

    default_kwargs = {
        'pricing_policy': PricingPolicy.QUADRATIC,
        'demand_pattern': DemandPattern.SINUSOIDAL,
        'cost_type': CostType.CONSTANT,
    }

    default_kwargs.update(kwargs)

    return DictToBoxWrapper(EnergyNetV0(**default_kwargs))

def test_learn_with_mo_environment():
    """Test learning with multi-objective environment."""
    base_env = create_energynet_env()
    mo_env = MOPCSWrapper(base_env, num_objectives=4)

    model = MOSAC(
        policy="MOSACPolicy",
        env=mo_env,
        num_objectives=4,
        learning_starts=10,
        buffer_size=1000,
        batch_size=64,
        verbose=0,
        policy_kwargs={"share_features_across_objectives": False},
        train_freq=(1, "episode"),
        gradient_steps=1,
    )

    # Short learning run
    model.learn(total_timesteps=500000, log_interval=1, callback=
                SaveOnBestTrainingRewardCallback(check_freq = 50, log_dir= "MORL_modules/logs/mosac_monitor/"))

    plot_results_scalarized("MORL_modules/logs/mosac_monitor/", title="Learning Curve")
    # Check that model has learned something
    assert model._n_updates >= 0
    assert model.replay_buffer.size() > 0
