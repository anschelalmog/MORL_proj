import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_mosac_fixed import train_mosac

# Override config for quick test
import train_mosac_fixed
train_mosac_fixed.config = {
    'num_objectives': 4,
    'n_timesteps': 1000,  # Very short for testing
    'seed': 42,
    'log_dir': 'logs/mosac_test',
    'preference_weights': [0.4, 0.2, 0.2, 0.2],
    'learning_rate': 3e-4,
    'batch_size': 64,
    'buffer_size': 1000,
    'learning_starts': 100,
    'tau': 0.005,
    'gamma': 0.99,
    'train_freq': 1,
    'gradient_steps': 1,
    'eval_freq': 500,
    'save_freq': 500,
}

if __name__ == "__main__":
    train_mosac()