"""
PCS Environment Factory for RL-Baselines3-Zoo integration.

This module provides factory functions that create PCS-focused environments
wrapped appropriately for training with RL-Baselines3-Zoo.
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from energy_net.envs import EnergyNetV0


def make_pcs_env_zoo(
    norm_path=None,
    iso_policy_path=None,
    log_dir="logs",
    use_dispatch_action=False,
    dispatch_strategy="PROPORTIONAL",
    monitor=True,
    seed=None,
    controller_name="EnergyNetController",
    controller_module="energy_net.controllers",
    **kwargs
):
    """
    Factory function for PCS environment compatible with RL-Baselines3-Zoo.
    
    Args:
        norm_path: Path to saved normalization statistics
        iso_policy_path: Path to a trained ISO policy to use during PCS training
        log_dir: Directory for saving logs
        use_dispatch_action: Whether to include dispatch in ISO action space
        dispatch_strategy: Strategy for dispatch when not controlled by agent
        monitor: Whether to wrap with Monitor for episode stats
        seed: Random seed
        controller_name: Name of the controller class to use
        controller_module: Python module path where the controller is defined
        **kwargs: Additional arguments to pass to EnergyNetV0
        
    Returns:
        A PCS-focused environment ready for training with RL-Baselines3-Zoo
    """
    # Create monitor directory if it doesn't exist
    if monitor:
        monitor_dir = os.path.join(log_dir, "pcs_monitor")
        os.makedirs(monitor_dir, exist_ok=True)
    
    # Create base environment
    env_kwargs = {
        "controller_name": controller_name,
        "controller_module": controller_module,
        "dispatch_config": {
            "use_dispatch_action": use_dispatch_action,
            "default_strategy": dispatch_strategy
        }
    }
    env_kwargs.update(kwargs)
    
    env = EnergyNetV0(**env_kwargs)
    
    # Load ISO policy if provided
    iso_policy = None
    if iso_policy_path:
        try:
            print(f"Loading ISO policy from {iso_policy_path}")
            iso_policy = PPO.load(iso_policy_path)
        except Exception as e:
            print(f"Error loading ISO policy: {e}")
    
    # Import here to avoid circular imports
    from tmp.alternating_wrappers import PCSEnvWrapper
    
    # Apply PCS wrapper
    env = PCSEnvWrapper(env, iso_policy=iso_policy)
    
    # Apply monitor wrapper if requested
    if monitor:
        env = Monitor(env, monitor_dir, allow_early_resets=True)
    
    # Set random seed if provided
    if seed is not None:
        env.seed(seed)
    
    return env 