import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import torch as th
import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Simple Multi-Objective Environment (Mock PCS Environment)
class MockPCSEnv(gym.Env):
    """Simplified mock PCS environment with 4 objectives"""
    
    def __init__(self):
        super().__init__()
        
        # Action space: battery charge/discharge (-1 to 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space: [battery_level, time, price_buy, price_sell]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]), 
            high=np.array([100.0, 1.0, 100.0, 100.0]), 
            dtype=np.float32
        )
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.battery_level = 50.0  # Start at 50% battery
        self.time = 0.0
        self.step_count = 0
        self.max_steps = 48  # One day simulation
        
        # Reset episode tracking
        self.episode_rewards = np.zeros(4)  # Track 4 objectives
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        # Simple price model (varies with time)
        price_buy = 20 + 10 * np.sin(2 * np.pi * self.time)
        price_sell = price_buy * 0.9  # Sell price is 90% of buy price
        
        return np.array([
            self.battery_level,
            self.time,
            price_buy,
            price_sell
        ], dtype=np.float32)
    
    def step(self, action):
        action = np.clip(action[0], -1.0, 1.0)
        
        # Update battery level
        old_battery = self.battery_level
        self.battery_level = np.clip(self.battery_level + action * 10, 0.0, 100.0)
        actual_action = (self.battery_level - old_battery) / 10.0
        
        # Calculate 4 objectives
        obs = self._get_obs()
        price_buy, price_sell = obs[2], obs[3]
        
        # Objective 1: Economic (profit/loss from energy trading)
        if actual_action > 0:  # Charging (buying energy)
            economic_reward = -abs(actual_action) * price_buy * 0.1
        else:  # Discharging (selling energy)
            economic_reward = abs(actual_action) * price_sell * 0.1
            
        # Objective 2: Battery health (prefer middle range, avoid cycling)
        battery_health_reward = 1.0 - 2.0 * abs(self.battery_level - 50.0) / 100.0
        battery_health_reward -= abs(actual_action) * 0.5  # Penalize large changes
        
        # Objective 3: Grid support (help when prices are extreme)
        price_signal = (price_buy - 30) / 20.0  # Normalize around average price
        grid_support_reward = -actual_action * price_signal * 0.5
        
        # Objective 4: Energy autonomy (maintain charge for independence)
        autonomy_reward = self.battery_level / 100.0
        
        # Store individual objectives
        mo_rewards = np.array([
            economic_reward,
            battery_health_reward, 
            grid_support_reward,
            autonomy_reward
        ])
        
        self.episode_rewards += mo_rewards
        
        # Update time
        self.step_count += 1
        self.time = self.step_count / self.max_steps
        
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        info = {
            'mo_rewards': mo_rewards,
            'battery_level': self.battery_level,
            'economic_reward': economic_reward,
            'battery_health_reward': battery_health_reward,
            'grid_support_reward': grid_support_reward,
            'autonomy_reward': autonomy_reward
        }
        
        if terminated:
            info['episode'] = {
                'total_mo_rewards': self.episode_rewards.copy(),
                'r': np.sum(self.episode_rewards)  # For monitoring
            }
        
        return self._get_obs(), 0.0, terminated, truncated, info

# Multi-Objective Wrapper that scalarizes rewards
class MOWrapper(gym.Wrapper):
    def __init__(self, env, preference_weights=None):
        super().__init__(env)
        self.preference_weights = preference_weights if preference_weights is not None else np.array([0.25, 0.25, 0.25, 0.25])
        self.episode_mo_rewards = []
        
    def reset(self, **kwargs):
        self.episode_mo_rewards = []
        return self.env.reset(**kwargs)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if 'mo_rewards' in info:
            mo_rewards = info['mo_rewards']
            # Scalarize using preference weights
            scalarized_reward = np.dot(mo_rewards, self.preference_weights)
            self.episode_mo_rewards.append(mo_rewards)
            
            if terminated:
                info['episode_mo_rewards'] = np.array(self.episode_mo_rewards)
                
            return obs, scalarized_reward, terminated, truncated, info
        
        return obs, reward, terminated, truncated, info

# Callback to track multi-objective rewards
class MOCallback(BaseCallback):
    def __init__(self, preference_name="", verbose=0):
        super().__init__(verbose)
        self.preference_name = preference_name
        self.mo_rewards_log = []
        
    def _on_step(self) -> bool:
        info = self.locals.get('infos', [{}])[0]
        if 'episode' in info and 'total_mo_rewards' in info['episode']:
            mo_rewards = info['episode']['total_mo_rewards']
            self.mo_rewards_log.append(mo_rewards)
            
            if len(self.mo_rewards_log) % 10 == 0:
                print(f"{self.preference_name} - Episode {len(self.mo_rewards_log)}: "
                      f"Economic: {mo_rewards[0]:.2f}, Battery: {mo_rewards[1]:.2f}, "
                      f"Grid: {mo_rewards[2]:.2f}, Autonomy: {mo_rewards[3]:.2f}")
        
        return True

def train_mosac_experiment(preference_weights, name, n_timesteps=50000):
    """Train SAC with specific preference weights"""
    
    print(f"\n=== Training {name} ===")
    print(f"Preference weights: {preference_weights}")
    
    # Create environment
    env = MockPCSEnv()
    env = MOWrapper(env, preference_weights)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # Create callback
    callback = MOCallback(preference_name=name)
    
    # Train SAC (as proxy for MOSAC)
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        device="cuda"
    )
    
    model.learn(total_timesteps=n_timesteps, callback=callback)
    
    return model, callback.mo_rewards_log

def plot_results(results_balanced, results_economic):
    """Plot comparison of the two training runs"""
    
    plt.figure(figsize=(15, 10))
    
    # Convert to numpy arrays for easier plotting
    balanced_rewards = np.array(results_balanced)
    economic_rewards = np.array(results_economic)
    
    objective_names = ['Economic', 'Battery Health', 'Grid Support', 'Energy Autonomy']
    
    # Plot each objective
    for i, obj_name in enumerate(objective_names):
        plt.subplot(2, 2, i+1)
        
        if len(balanced_rewards) > 0:
            plt.plot(balanced_rewards[:, i], label='Balanced (1/4,1/4,1/4,1/4)', alpha=0.7)
        if len(economic_rewards) > 0:
            plt.plot(economic_rewards[:, i], label='Economic Focus (1,0,0,0)', alpha=0.7)
            
        plt.title(f'{obj_name} Objective')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mosac_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== Results Summary ===")
    if len(balanced_rewards) > 0:
        print("Balanced weights - Final rewards:")
        final_balanced = balanced_rewards[-1]
        for i, obj_name in enumerate(objective_names):
            print(f"  {obj_name}: {final_balanced[i]:.2f}")
    
    if len(economic_rewards) > 0:
        print("Economic focus - Final rewards:")
        final_economic = economic_rewards[-1]
        for i, obj_name in enumerate(objective_names):
            print(f"  {obj_name}: {final_economic[i]:.2f}")

if __name__ == "__main__":
    # Training parameters
    n_timesteps = 25000  # Reduced for faster testing
    
    # Experiment 1: Balanced preferences
    model1, results1 = train_mosac_experiment(
        preference_weights=np.array([0.25, 0.25, 0.25, 0.25]),
        name="Balanced",
        n_timesteps=n_timesteps
    )
    
    # Experiment 2: Economic focus
    model2, results2 = train_mosac_experiment(
        preference_weights=np.array([1.0, 0.0, 0.0, 0.0]),
        name="Economic Focus", 
        n_timesteps=n_timesteps
    )
    
    # Plot comparison
    plot_results(results1, results2)
    
    print(f"\nTraining complete! Results saved to 'mosac_comparison.png'")
