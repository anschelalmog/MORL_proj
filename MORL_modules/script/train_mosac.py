import gymnasium as gym
import numpy as np
import argparse
import os
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from MORL_modules.algorithms.mosac import MOSAC, register_mosac
from MORL_modules.wrappers.MOwrapper import MOEnergyNetWrapper

# Import energy_net environment
import energy_net.env.register_envs
from energy_net.env import EnergyNetV0

# Make sure alternating_wrappers is available
from tmp.alternating_wrappers import PCSEnvWrapper  # Assuming this exists

def make_mo_pcs_env(
        log_dir="logs",
        monitor=True,
        num_objectives=4,
        preference_weights=None,
        **env_kwargs
):
    """Create a multi-objective PCS environment."""
    # Create base environment with passed kwargs
    base_env = EnergyNetV0(**env_kwargs)

    # Wrap for PCS-only training
    pcs_env = PCSEnvWrapper(base_env)

    # Apply monitoring
    if monitor:
        os.makedirs(log_dir, exist_ok=True)
        pcs_env = Monitor(pcs_env, log_dir)

    # Apply multi-objective wrapper
    mo_env = MOEnergyNetWrapper(
        pcs_env,
        num_objectives=num_objectives,
        reward_weights=preference_weights
    )

    return mo_env

def train_mosac(args):
    """Train MOSAC agent on EnergyNet environment."""
    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Prepare log directories
    log_dir = os.path.join(args.log_folder, "mosac")
    os.makedirs(log_dir, exist_ok=True)

    tensorboard_log = os.path.join(args.log_folder, "tensorboard")
    os.makedirs(tensorboard_log, exist_ok=True)

    # Create environment
    env_kwargs = {
        "pricing_policy": args.pricing_policy,
        "demand_pattern": args.demand_pattern,
        "cost_type": args.cost_type,
        "dispatch_config": {
            "use_dispatch_action": args.use_dispatch_action,
            "default_strategy": args.dispatch_strategy
        }
    }

    # Create environment
    env = make_mo_pcs_env(
        log_dir=os.path.join(log_dir, "train_monitor"),
        num_objectives=args.num_objectives,
        **env_kwargs
    )

    # Vectorize environment (required by SB3)
    env = DummyVecEnv([lambda: env])

    # Create evaluation environment
    eval_env = make_mo_pcs_env(
        log_dir=os.path.join(log_dir, "eval_monitor"),
        num_objectives=args.num_objectives,
        **env_kwargs
    )
    eval_env = DummyVecEnv([lambda: eval_env])

    # Set preference weights if specified
    if args.preference_weights:
        weights = [float(w) for w in args.preference_weights.split(',')]
        assert len(weights) == args.num_objectives, "Number of weights must match number of objectives"
    else:
        # Default equal weights
        weights = np.ones(args.num_objectives) / args.num_objectives

    # Create policy_kwargs
    policy_kwargs = {
        "net_arch": {
            "pi": [64, 64],  # Actor network
            "qf": [64, 64]   # Critic network
        },
        "share_features_across_objectives": True
    }

    # Initialize MOSAC agent
    model = MOSAC(
        policy="MlpPolicy",
        env=env,
        num_objectives=args.num_objectives,
        preference_weights=weights,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        verbose=1,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        device=args.device
    )

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    if args.save_freq > 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.save_freq // env.num_envs, 1),
            save_path=os.path.join(log_dir, "checkpoints"),
            name_prefix="mosac_model",
            verbose=1
        )
        callbacks.append(checkpoint_callback)

    # Evaluation callback
    if args.eval_freq > 0:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(log_dir, "best_model"),
            log_path=os.path.join(log_dir, "eval_results"),
            eval_freq=max(args.eval_freq // env.num_envs, 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            verbose=1
        )
        callbacks.append(eval_callback)

    # Train model
    model.learn(
        total_timesteps=args.n_timesteps,
        callback=callbacks,
        log_interval=10
    )

    # Save final model
    model.save(os.path.join(log_dir, "final_model"))

    print(f"Training complete! Model saved to {os.path.join(log_dir, 'final_model')}")

    return model

if __name__ == "__main__":
    # Register MOSAC with rl-baselines3-zoo
    register_mosac()

    parser = argparse.ArgumentParser(description="Train MOSAC on EnergyNet")

    # Algorithm parameters
    parser.add_argument("--num-objectives", type=int, default=4, help="Number of objectives")
    parser.add_argument("--preference-weights", type=str, default=None, help="Comma-separated list of preference weights")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=1000000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    # Environment parameters
    parser.add_argument("--pricing-policy", type=str, default="ONLINE", help="Pricing policy")
    parser.add_argument("--demand-pattern", type=str, default="SINUSOIDAL", help="Demand pattern")
    parser.add_argument("--cost-type", type=str, default="CONSTANT", help="Cost type")
    parser.add_argument("--use-dispatch-action", action="store_true", help="Use dispatch action")
    parser.add_argument("--dispatch-strategy", type=str, default="PROPORTIONAL", help="Dispatch strategy")

    # Training parameters
    parser.add_argument("--n-timesteps", type=int, default=100000, help="Number of timesteps to train")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluate every n steps")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of episodes for evaluation")
    parser.add_argument("--save-freq", type=int, default=10000, help="Save checkpoint every n steps")
    parser.add_argument("--log-folder", type=str, default="logs", help="Log folder")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, ...)")

    args = parser.parse_args()

    # Train MOSAC
    model = train_mosac(args)