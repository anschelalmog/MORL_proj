#!/usr/bin/env python3
"""
Train a MOSAC agent on the EnergyNet multi-objective environment.

Update:
    --seed now accepts an integer or the keyword 'none' (case-insensitive) to disable seeding.
    Example: --seed none  (will pass seed=None to MOSAC)

Example:
    python train_mosac.py --total-timesteps 300000 --weights 1,0.5,1.5,1 --learning-rate 1e-4 --seed none
"""
import argparse
import sys
import os
from typing import Dict, Optional
import numpy as np
import shlex
# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'MORL_modules'))

print("Project root:", project_root)
print("Current dir :", current_dir)

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from agents.mosac import MOSAC
from agents.mobuffers import MOReplayBuffer  # noqa: F401
from wrappers.mo_pcs_wrapper import MOPCSWrapper
from wrappers.dict_to_box_wrapper import DictToBoxWrapper
from energy_net.envs.energy_net_v0 import EnergyNetV0

from energy_net.market.pricing.pricing_policy import PricingPolicy
from energy_net.market.pricing.cost_types import CostType
from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern

from utils.utils import plot_results_scalarized
from utils.callbacks import SaveOnBestTrainingRewardCallback

# -----------------------------------------------------------------------------
# Preset configurations for rewards normalization stats
# -----------------------------------------------------------------------------
# Canonical "MOSAC" preset (original script values)
MOSAC_PRESET = {
    "economic":       {"min": -50.0, "max": 50.0, "std": 10.0},
    "battery_health": {"min": -2.0,  "max": 1.0,  "std": 0.5},
    "grid_support":   {"min": -1.0,  "max": 1.0,  "std": 0.3},
    "autonomy":       {"min": 0.0,   "max": 1.0,  "std": 0.3},
}

# Example alternative "Hyper-Morl" preset (hypothetical, broader ranges).
# Adjust these if you have domain-specific scaling.
HYPER_MORL_PRESET =   {
                'economic': {'min': -10, 'max': 10, 'mean': -8.1, 'std': 0.1},
                'battery_health': {'min': -5, 'max': 5, 'mean': -0.057, 'std': 0.001},
                'grid_support': {'min': -0.02, 'max': 0.02, 'mean': -0.057, 'std': 0.001},
                'autonomy': {'min': -1.0, 'max': 1.0, 'mean': 0.5, 'std': 0.3}
 }


PRESET_CONFIGS = {
    "mosac": MOSAC_PRESET,
    "hyper-morl": HYPER_MORL_PRESET
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def str_to_list_floats(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    try:
        arr = np.array([float(p) for p in parts], dtype=float)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Could not parse weights '{s}': {e}")
    return arr


def parse_enum(enum_cls, value: str):
    try:
        return enum_cls[value.upper()]
    except KeyError:
        valid = ", ".join([m.name for m in enum_cls])
        raise argparse.ArgumentTypeError(
            f"Invalid value '{value}' for {enum_cls.__name__}. Valid: {valid}"
        )


def parse_seed(v: str) -> Optional[int]:
    """
    Parse --seed argument allowing 'none' (case-insensitive) to produce None.
    """
    if isinstance(v, str):
        vl = v.strip().lower()
        if vl in ("none", "null", ""):
            return None
    try:
        return int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Seed must be an integer or 'none', got '{v}'."
        )


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_args_file(args: argparse.Namespace,
                   reward_stats: Dict[str, Dict[str, float]],
                   file_path: str):
    """
    Save command line, resolved arguments, reward stats, and override info to args.txt
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# ================== MOSAC Training Arguments ==================\n")
            f.write("Command Line:\n")
            f.write("python " + os.path.basename(__file__) + " " +
                    " ".join(shlex.quote(a) for a in sys.argv[1:]) + "\n\n")


            f.write("Resolved Arguments:\n")
            # Convert enums to their names for readability
            for k, v in sorted(vars(args).items()):
                if isinstance(v, (PricingPolicy, DemandPattern, CostType)):
                    v_repr = v.name
                else:
                    v_repr = v
                f.write(f"  {k}: {v_repr}\n")
            f.write("\nReward Stats (effective):\n")
            for name, rs in reward_stats.items():
                f.write(f"  {name}:\n")
                for key in ["min", "max", "mean", "std"]:
                    f.write(f"    {key}: {rs[key]}\n")

    except Exception as e:
        print(f"WARNING: Failed to write args file at {file_path}: {e}")


# -----------------------------------------------------------------------------
# Reward stats builder from individual min/max args
# -----------------------------------------------------------------------------

def flags_provided(flag_names):
    """
    Return True if ANY of the provided flag names (list of str) appears
    verbatim in sys.argv.
    """
    argv_set = set(sys.argv[1:])  # skip program name
    return any(flag in argv_set for flag in flag_names)


def build_reward_stats(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    # Start from preset
    preset_key = args.reward_stats.lower()
    if preset_key in ("none", "null", ""):
            return None
    base = PRESET_CONFIGS[preset_key]

    # Clone to avoid mutation
    stats = {
        name: {
            "min": cfg["min"],
            "max": cfg["max"],
            "std": cfg["std"]
        } for name, cfg in base.items()
    }

    # Determine overrides (only apply if the flag appears on the command line)
    # Mapping: objective -> (min_flag, max_flag, std_flag, attr_prefix)
    override_map = {
        "economic":       (["--economic-min"], ["--economic-max"], ["--economic-std"], "economic"),
        "battery_health": (["--battery-health-min"], ["--battery-health-max"], ["--battery-health-std"], "battery_health"),
        "grid_support":   (["--grid-support-min"], ["--grid-support-max"], ["--grid-support-std"], "grid_support"),
        "autonomy":       (["--autonomy-min"], ["--autonomy-max"], ["--autonomy-std"], "autonomy"),
    }

    for obj, (min_flags, max_flags, std_flags, prefix) in override_map.items():
        if flags_provided(min_flags):
            stats[obj]["min"] = getattr(args, f"{prefix.replace('-', '_')}_min")
        if flags_provided(max_flags):
            stats[obj]["max"] = getattr(args, f"{prefix.replace('-', '_')}_max")
        if flags_provided(std_flags):
            stats[obj]["std"] = getattr(args, f"{prefix.replace('-', '_')}_std")

        # Validate
        if stats[obj]["max"] < stats[obj]["min"]:
            raise ValueError(f"For objective '{obj}' max < min after overrides.")

        # Recompute mean every time
        stats[obj]["mean"] = 0.5 * (stats[obj]["min"] + stats[obj]["max"])

        # Guard against zero or negative std (if user sets something odd)
        if stats[obj]["std"] <= 0:
            raise ValueError(f"Std must be positive for objective '{obj}', got {stats[obj]['std']}.")

    return stats


# -----------------------------------------------------------------------------
# Environment factory
# -----------------------------------------------------------------------------
def create_energynet_env(pricing_policy: PricingPolicy,
                         demand_pattern: DemandPattern,
                         cost_type: CostType,
                         pcs_unit_config_path: str,
                         **kwargs):
    env = EnergyNetV0(
        pricing_policy=pricing_policy,
        demand_pattern=demand_pattern,
        cost_type=cost_type,
        pcs_unit_config_path=pcs_unit_config_path,
        **kwargs
    )
    return DictToBoxWrapper(env)


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
def get_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train MOSAC on EnergyNet multi-objective environment."
    )

    # Environment / domain
    p.add_argument("--pricing-policy", type=lambda s: parse_enum(PricingPolicy, s),
                   default="QUADRATIC", help="PricingPolicy enum name.")
    p.add_argument("--demand-pattern", type=lambda s: parse_enum(DemandPattern, s),
                   default="SINUSOIDAL", help="DemandPattern enum name.")
    p.add_argument("--cost-type", type=lambda s: parse_enum(CostType, s),
                   default="CONSTANT", help="CostType enum name.")
    p.add_argument("--pcs-unit-config-path", type=str,
                   default="MORL_modules/configs/pcs_unit_config.yaml",
                   help="Path to PCS unit config YAML.")
    p.add_argument("--reward-scale", type=float, default=1.0,
                   help="Uniform scaling factor applied to predefined reward stats.")

    # MOSAC / training hyper-parameters
    p.add_argument("--total-timesteps", type=int, default=500_000,
                   help="Total timesteps for learning.")
    p.add_argument("--learning-starts", type=int, default=10,
                   help="Timesteps before learning starts.")
    p.add_argument("--buffer-size", type=int, default=1000,
                   help="Replay buffer size.")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size.")
    p.add_argument("--gradient-steps", type=int, default=1,
                   help="Gradient steps per training iteration.")
    p.add_argument("--train-freq-n", type=int, default=1,
                   help="Numeric part of train_freq tuple.")
    p.add_argument("--train-freq-unit", type=str, choices=["step", "episode"], default="episode",
                   help="Unit for train frequency.")
    p.add_argument("--learning-rate", type=float, default=3e-5,
                   help="Optimizer learning rate (default 3e-5).")

    p.add_argument("--seed", type=parse_seed, default=42,
                   help="Random seed integer or 'none' to disable deterministic seeding (default 42).")
    p.add_argument("--verbose", type=int, default=1, choices=[0, 1],
                   help="Verbosity level.")


    # Preset selection
    p.add_argument("--reward_stats", type=str, choices=["MOSAC", "Hyper-Morl", "mosac", "hyper-morl"],
                   default="MOSAC",
                   help="Preset reward stats configuration to use (case-insensitive).")

    # Per-objective min/max/std arguments (defaults set to MOSAC preset for convenience)
    p.add_argument("--economic-min", type=float, default=MOSAC_PRESET["economic"]["min"],
                   help="Min economic reward.")
    p.add_argument("--economic-max", type=float, default=MOSAC_PRESET["economic"]["max"],
                   help="Max economic reward.")
    p.add_argument("--economic-std", type=float, default=MOSAC_PRESET["economic"]["std"],
                   help="Std economic reward (override).")

    p.add_argument("--battery-health-min", type=float, default=MOSAC_PRESET["battery_health"]["min"],
                   help="Min battery_health reward.")
    p.add_argument("--battery-health-max", type=float, default=MOSAC_PRESET["battery_health"]["max"],
                   help="Max battery_health reward.")
    p.add_argument("--battery-health-std", type=float, default=MOSAC_PRESET["battery_health"]["std"],
                   help="Std battery_health reward (override).")

    p.add_argument("--grid-support-min", type=float, default=MOSAC_PRESET["grid_support"]["min"],
                   help="Min grid_support reward.")
    p.add_argument("--grid-support-max", type=float, default=MOSAC_PRESET["grid_support"]["max"],
                   help="Max grid_support reward.")
    p.add_argument("--grid-support-std", type=float, default=MOSAC_PRESET["grid_support"]["std"],
                   help="Std grid_support reward (override).")

    p.add_argument("--autonomy-min", type=float, default=MOSAC_PRESET["autonomy"]["min"],
                   help="Min autonomy reward.")
    p.add_argument("--autonomy-max", type=float, default=MOSAC_PRESET["autonomy"]["max"],
                   help="Max autonomy reward.")
    p.add_argument("--autonomy-std", type=float, default=MOSAC_PRESET["autonomy"]["std"],
                   help="Std autonomy reward (override).")


    p.add_argument("--share-features", action="store_true",
                   help="Share features across objectives (policy_kwargs).")
    p.add_argument("--no-share-features", action="store_false", dest="share_features",
                   help="Disable shared features.")
    p.set_defaults(share_features=True)

    # Multi-objective specifics
    p.add_argument("--weights", type=str_to_list_floats, default="1,1,1,1",
                   help="Comma-separated preference weights.")
    p.add_argument("--num-objectives", type=int, default=4,
                   help="Number of objectives.")
    p.add_argument("--calc-mse-before-scalarization", action="store_true",
                   help="Use MSE before scalarization.")
    p.add_argument("--no-calc-mse-before-scalarization", action="store_false",
                   dest="calc_mse_before_scalarization",
                   help="Use MSE after scalarization.")
    p.set_defaults(calc_mse_before_scalarization=False)

    # Logging / output
    p.add_argument("--log-dir", type=str,
                   default="MORL_modules/logs/mosac_monitor/",
                   help="Directory to store logs & monitor.")
    p.add_argument("--plot-title", type=str, default="Learning Curve",
                   help="Title for the results plot.")
    p.add_argument("--save-check-freq", type=int, default=500,
                   help="Callback frequency (timesteps) to evaluate & save best model.")

    # Misc
    p.add_argument("--dry-run", action="store_true",
                   help="Initialize everything but skip training loop.")

    return p


# -----------------------------------------------------------------------------
# Main training logic
# -----------------------------------------------------------------------------
def main(args: argparse.Namespace):
    if args.num_objectives != len(args.weights):
        raise ValueError(
            f"num-objectives ({args.num_objectives}) does not match weights length ({len(args.weights)})"
        )
    reward_stats = build_reward_stats(args)

    base_env = create_energynet_env(
        pricing_policy=args.pricing_policy,
        demand_pattern=args.demand_pattern,
        cost_type=args.cost_type,
        pcs_unit_config_path=args.pcs_unit_config_path
    )

    mo_env = MOPCSWrapper(
        base_env,
        num_objectives=args.num_objectives,
        reward_stats=reward_stats
    )

    ensure_dir(args.log_dir)

    # Save args + reward stats BEFORE training for reproducibility
    args_file_path = os.path.join(args.log_dir, "args.txt")
    save_args_file(args, reward_stats,  args_file_path)
    print(f"Arguments saved to: {args_file_path}")

    policy_kwargs = {
        "share_features_across_objectives": args.share_features
    }

    train_freq = (args.train_freq_n, args.train_freq_unit)

    print("\n=== Configuration Summary ===")
    print("Pricing policy        :", args.pricing_policy.name)
    print("Demand pattern        :", args.demand_pattern.name)
    print("Cost type             :", args.cost_type.name)
    print("PCS config path       :", args.pcs_unit_config_path)
    print("Num objectives        :", args.num_objectives)
    print("Weights               :", args.weights)
    print("Train freq            :", train_freq)
    print("Learning rate         :", args.learning_rate)
    print("Buffer size           :", args.buffer_size)
    print("Batch size            :", args.batch_size)
    print("Learning starts       :", args.learning_starts)
    print("Gradient steps        :", args.gradient_steps)
    print("Calc MSE before scal. :", args.calc_mse_before_scalarization)
    print("Share features        :", args.share_features)
    print("Total timesteps       :", args.total_timesteps)
    print("Log dir               :", args.log_dir)
    print("Seed                  :", args.seed if args.seed is not None else "None (not setting)")
    print("Dry run               :", args.dry_run)
    print("==============================\n")

    mosac_kwargs = dict(
        policy="MOSACPolicy",
        env=mo_env,
        num_objectives=args.num_objectives,
        learning_starts=args.learning_starts,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        verbose=args.verbose,
        policy_kwargs=policy_kwargs,
        train_freq=train_freq,
        gradient_steps=args.gradient_steps,
        preference_weights=args.weights,
        calculate_mse_before_scalarization=args.calc_mse_before_scalarization,
        learning_rate=args.learning_rate,
        log_folder= args.log_dir
    )
    # Only include seed if not None (in case MOSAC behaves differently when omitted)
    if args.seed is not None:
        mosac_kwargs["seed"] = args.seed

    model = MOSAC(**mosac_kwargs)

    if not args.dry_run:
        callback = SaveOnBestTrainingRewardCallback(
            check_freq=args.save_check_freq,
            log_dir=args.log_dir
        )
        model.learn(
            total_timesteps=args.total_timesteps,
            log_interval=1,
            callback=callback
        )

        plot_results_scalarized(
            args.log_dir,
            title=args.plot_title,
            preference_weights=args.weights
        )

        assert model._n_updates >= 0
        assert model.replay_buffer.size() > 0
        print("Training completed successfully.")
    else:
        print("Dry run: Skipping training loop.")


if __name__ == "__main__":
    parser = get_arg_parser()
    parsed_args = parser.parse_args()
    main(parsed_args)