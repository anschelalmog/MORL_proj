import os
import sys
import numpy as np
import torch as th
import gymnasium as gym
from datetime import datetime
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from MORL_modules.algorithms.mosac import MOSAC, register_mosac
from MORL_modules.wrappers.MOwrapper import MOEnergyNetWrapper

# Import energy_net environment
import energy_net.env.register_envs
from energy_net.env import EnergyNetV0
from alternating_wrappers import PCSEnvWrapper


class MORewardLogger(BaseCallback):
    """Custom callback for logging multi-objective rewards."""

    def __init__(self, num_objectives=4, log_freq=100, verbose=0):
        super().__init__(verbose)
        self.num_objectives = num_objectives
        self.log_freq = log_freq
        self.episode_rewards = [[] for _ in range(num_objectives)]
        self.current_rewards = np.zeros((1, num_objectives))  # For single env
        self.episode_count = 0

        # For plotting
        self.episode_numbers = []
        self.mean_rewards_per_obj = [[] for _ in range(num_objectives)]

    def _on_step(self) -> bool:
        # Get info from the environment
        if len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]

            # Extract multi-objective rewards
            if 'mo_rewards' in info:
                mo_rewards = info['mo_rewards']
                self.current_rewards[0] += mo_rewards

            # Check if episode ended
            if self.locals['dones'][0]:
                # Log episode rewards for each objective
                for obj_idx in range(self.num_objectives):
                    self.episode_rewards[obj_idx].append(self.current_rewards[0, obj_idx])

                    # Record to tensorboard
                    self.logger.record(f'rollout/ep_reward_obj_{obj_idx}', self.current_rewards[0, obj_idx])

                # Record scalarized reward
                scalarized = np.dot(self.current_rewards[0], self.model.preference_weights)
                self.logger.record('rollout/ep_reward_scalarized', scalarized)

                # Reset current rewards
                self.current_rewards[0] = 0
                self.episode_count += 1

                # Calculate and store means for plotting
                if self.episode_count % self.log_freq == 0:
                    self.episode_numbers.append(self.episode_count)
                    for obj_idx in range(self.num_objectives):
                        if len(self.episode_rewards[obj_idx]) > 0:
                            mean_reward = np.mean(self.episode_rewards[obj_idx][-self.log_freq:])
                            self.mean_rewards_per_obj[obj_idx].append(mean_reward)

        return True

    def plot_rewards(self, save_path):
        """Plot the learning curves for each objective."""
        if len(self.episode_numbers) == 0:
            print("No episodes completed yet, cannot plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        objective_names = ['Economic', 'Battery Health', 'Grid Support', 'Energy Autonomy']

        for obj_idx in range(self.num_objectives):
            ax = axes[obj_idx]
            if len(self.mean_rewards_per_obj[obj_idx]) > 0:
                ax.plot(self.episode_numbers, self.mean_rewards_per_obj[obj_idx],
                        label=f'Objective {obj_idx + 1}')
                ax.set_xlabel('Episode')
                ax.set_ylabel('Cumulative Reward')
                ax.set_title(f'{objective_names[obj_idx]} Objective')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data yet', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_title(f'{objective_names[obj_idx]} Objective')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved learning curves to {save_path}")


def make_mo_pcs_env(seed=0, **env_kwargs):
    """Create a multi-objective PCS environment."""
    # Create base environment
    base_env = EnergyNetV0(**env_kwargs)

    # Wrap for PCS-only training
    pcs_env = PCSEnvWrapper(base_env)

    # Set seed
    pcs_env.seed(seed)

    # Apply multi-objective wrapper
    mo_env = MOEnergyNetWrapper(pcs_env, num_objectives=4)

    return mo_env


def train_mosac():
    """Train MOSAC agent with proper logging."""
    # Register MOSAC
    register_mosac()

    # Configuration
    config = {
        'num_objectives': 4,
        'n_timesteps': 100000,
        'seed': 42,
        'log_dir': 'logs/mosac',
        'preference_weights': [0.4, 0.2, 0.2, 0.2],  # Economic, Battery, Grid, Autonomy
        'learning_rate': 3e-4,
        'batch_size': 256,
        'buffer_size': 100000,
        'learning_starts': 1000,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'eval_freq': 5000,
        'save_freq': 10000,
    }

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config['log_dir'], timestamp)
    os.makedirs(log_dir, exist_ok=True)

    # Environment configuration
    env_kwargs = {
        'pricing_policy': 'ONLINE',
        'demand_pattern': 'SINUSOIDAL',
        'cost_type': 'CONSTANT',
        'dispatch_config': {
            'use_dispatch_action': True,
            'default_strategy': 'PROPORTIONAL'
        }
    }

    # Create environments
    print("Creating training environment...")
    train_env = make_mo_pcs_env(seed=config['seed'], **env_kwargs)
    train_env = Monitor(train_env, os.path.join(log_dir, "train"))
    train_env = DummyVecEnv([lambda: train_env])

    print("Creating evaluation environment...")
    eval_env = make_mo_pcs_env(seed=config['seed'] + 100, **env_kwargs)
    eval_env = Monitor(eval_env, os.path.join(log_dir, "eval"))
    eval_env = DummyVecEnv([lambda: eval_env])

    # Initialize MOSAC
    print("Initializing MOSAC agent...")
    model = MOSAC(
        policy="MlpPolicy",
        env=train_env,
        num_objectives=config['num_objectives'],
        preference_weights=config['preference_weights'],
        learning_rate=config['learning_rate'],
        buffer_size=config['buffer_size'],
        learning_starts=config['learning_starts'],
        batch_size=config['batch_size'],
        tau=config['tau'],
        gamma=config['gamma'],
        train_freq=config['train_freq'],
        gradient_steps=config['gradient_steps'],
        tensorboard_log=os.path.join(log_dir, 'tensorboard'),
        verbose=1,
        seed=config['seed'],
        device='cuda' if th.cuda.is_available() else 'cpu',
        policy_kwargs={
            'net_arch': {'pi': [64, 64], 'qf': [64, 64]},
            'num_objectives': config['num_objectives']
        },
        replay_buffer_class=MOReplayBuffer,
        replay_buffer_kwargs={'num_objectives': config['num_objectives']}
    )

    # Callbacks
    mo_logger = MORewardLogger(num_objectives=config['num_objectives'], log_freq=10)

    checkpoint_callback = CheckpointCallback(
        save_freq=config['save_freq'],
        save_path=os.path.join(log_dir, 'checkpoints'),
        name_prefix='mosac',
        save_replay_buffer=True,
        save_vecnormalize=True
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, 'best_model'),
        log_path=os.path.join(log_dir, 'eval'),
        eval_freq=config['eval_freq'],
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    callbacks = [mo_logger, checkpoint_callback, eval_callback]

    # Train
    print(f"Starting training for {config['n_timesteps']} timesteps...")
    print(f"Preference weights: {config['preference_weights']}")
    print(f"Logging to: {log_dir}")

    try:
        model.learn(
            total_timesteps=config['n_timesteps'],
            callback=callbacks,
            log_interval=10,
            tb_log_name=f"mosac_{timestamp}",
            reset_num_timesteps=True,
            progress_bar=True
        )

        # Save final model
        model.save(os.path.join(log_dir, 'final_model'))

        # Plot final results
        mo_logger.plot_rewards(os.path.join(log_dir, 'mosac_learning_curves.png'))

        print(f"\nTraining completed successfully!")
        print(f"Results saved to: {log_dir}")

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

        # Try to save partial results
        try:
            mo_logger.plot_rewards(os.path.join(log_dir, 'mosac_partial_results.png'))
            model.save(os.path.join(log_dir, 'partial_model'))
        except:
            pass


if __name__ == "__main__":
    train_mosac()