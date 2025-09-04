# utils/env_diagnostic.py

import numpy as np
import sys
import os
from gymnasium import spaces

# Add energy_net to path - adjust this path to where energy_net is located
sys.path.append('/home/anschelalmog/MORL_PROJ/energy_net')
sys.path.append('/home/anschelalmog/MORL_PROJ')

try:
    from energy_net.envs.energy_net_v0 import EnergyNetV0
    from energy_net.market.pricing.cost_types import CostType
    from energy_net.market.pricing.pricing_policy import PricingPolicy
    from energy_net.dynamics.consumption_dynamics.demand_patterns import DemandPattern

    print("✓ Successfully imported energy_net modules")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please check the energy_net path in your project")
    sys.exit(1)


def diagnose_environment():
    """Diagnose the EnergyNet environment structure and action spaces."""

    print("=== EnergyNet Environment Diagnostic ===")

    try:
        # Create environment
        env = EnergyNetV0(
            controller_name="EnergyNetController",
            controller_module="energy_net.controllers",
            env_config_path='energy_net/configs/environment_config.yaml',
            iso_config_path='energy_net/configs/iso_config.yaml',
            pcs_unit_config_path='energy_net/configs/pcs_unit_config.yaml',
            cost_type=CostType.CONSTANT,
            pricing_policy=PricingPolicy.QUADRATIC,
            demand_pattern=DemandPattern.SINUSOIDAL,
        )

        print("✓ Environment created successfully")

        # Check action space
        print(f"\nAction Space Type: {type(env.action_space)}")

        if isinstance(env.action_space, spaces.Dict):
            print("Action Space Structure:")
            for key, space in env.action_space.spaces.items():
                print(f"  {key}: {space}")
                if hasattr(space, 'shape'):
                    print(f"    Shape: {space.shape}")
                if hasattr(space, 'low') and hasattr(space, 'high'):
                    print(f"    Range: [{space.low}, {space.high}]")
        else:
            print(f"Action Space: {env.action_space}")
            if hasattr(env.action_space, 'shape'):
                print(f"Shape: {env.action_space.shape}")

        # Check observation space
        print(f"\nObservation Space Type: {type(env.observation_space)}")

        if isinstance(env.observation_space, spaces.Dict):
            print("Observation Space Structure:")
            for key, space in env.observation_space.spaces.items():
                print(f"  {key}: {space}")
                if hasattr(space, 'shape'):
                    print(f"    Shape: {space.shape}")
        else:
            print(f"Observation Space: {env.observation_space}")
            if hasattr(env.observation_space, 'shape'):
                print(f"Shape: {env.observation_space.shape}")

        # Test reset
        print("\n=== Testing Reset ===")
        obs, info = env.reset()
        print(f"Reset successful")
        print(f"Observation type: {type(obs)}")
        if isinstance(obs, dict):
            for key, value in obs.items():
                print(f"  {key}: shape {np.array(value).shape}")
        else:
            print(f"Observation shape: {np.array(obs).shape}")

        # Test sample action
        print("\n=== Testing Sample Action ===")
        if isinstance(env.action_space, spaces.Dict):
            sample_action = {}
            for key, space in env.action_space.spaces.items():
                sample_action[key] = space.sample()
                print(f"Sample {key} action: {sample_action[key]} (shape: {np.array(sample_action[key]).shape})")
        else:
            sample_action = env.action_space.sample()
            print(f"Sample action: {sample_action} (shape: {np.array(sample_action).shape})")

        # Test step with sample action
        print("\n=== Testing Step ===")
        try:
            obs2, reward, terminated, truncated, info2 = env.step(sample_action)
            print("✓ Step successful")
            print(f"Reward type: {type(reward)}")
            if isinstance(reward, dict):
                for key, value in reward.items():
                    print(f"  {key}: {value}")
            else:
                print(f"Reward: {reward}")
        except Exception as e:
            print(f"✗ Step failed: {e}")

            # Try with zero action
            print("\nTrying with zero action...")
            if isinstance(env.action_space, spaces.Dict):
                zero_action = {}
                for key, space in env.action_space.spaces.items():
                    if hasattr(space, 'shape'):
                        zero_action[key] = np.zeros(space.shape)
                    else:
                        zero_action[key] = 0.0
            else:
                zero_action = np.zeros(env.action_space.shape)

            print(f"Zero action: {zero_action}")
            try:
                obs2, reward, terminated, truncated, info2 = env.step(zero_action)
                print("✓ Zero action step successful")
            except Exception as e2:
                print(f"✗ Zero action step failed: {e2}")

        # Check if we need PCS-only wrapper
        print("\n=== Environment Analysis ===")
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'controller'):
            controller = env.unwrapped.controller
            print(f"Controller type: {type(controller)}")

            if hasattr(controller, 'pcs_unit'):
                print("✓ PCS Unit available")
            if hasattr(controller, 'battery_manager'):
                print("✓ Battery Manager available")

        env.close()

    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()


def get_correct_action_format(env):
    """Get the correct action format for the environment."""
    if isinstance(env.action_space, spaces.Dict):
        action = {}
        for key, space in env.action_space.spaces.items():
            if hasattr(space, 'shape') and len(space.shape) > 0:
                action[key] = np.zeros(space.shape)
            else:
                action[key] = np.array([0.0])
    else:
        if hasattr(env.action_space, 'shape'):
            action = np.zeros(env.action_space.shape)
        else:
            action = np.array([0.0])

    return action


if __name__ == "__main__":
    diagnose_environment()