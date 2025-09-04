import sys
import numpy as np
from typing import Dict, Any, List, Tuple
import pdb
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'MORL_modules'))
#sys.path.append(os.path.join(project_root, 'energy-net'))
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

#!/usr/bin/env python3
"""
Wrapper script that calls train_mosac.py with a specific set of arguments,
now including --learning-rate.
"""
import subprocess
import sys
import os
from datetime import datetime

#THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(project_root, "MORL_modules/run_algos/train_mosac.py")

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"MORL_modules/logs/mosac_monitor_2_objective_run_{timestamp}"
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--total-timesteps", "500000",
        "--weights", "1,1,0,0",
        "--pricing-policy", "QUADRATIC",
        "--demand-pattern", "SINUSOIDAL",
        "--cost-type", "CONSTANT",
        "--buffer-size", "1000",
        "--batch-size", "64",
        "--learning-starts", "10",
        "--gradient-steps", "1",
        "--train-freq-n", "1",
        "--train-freq-unit", "episode",
        "--learning-rate", "3e-4",
        "--seed", "none",
        "--log-dir", log_dir,
        "--plot-title", f"MOSAC Learning Curve ({timestamp})",
        "--save-check-freq", "500",
        "--calc-mse-before-scalarization",
        "--share-features"
    ]

    print("Running command:\n", " ".join(cmd), "\n")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"Training script exited with non-zero status: {result.returncode}")
        sys.exit(result.returncode)
    else:
        print("Experiment finished successfully.")

if __name__ == "__main__":
    main()