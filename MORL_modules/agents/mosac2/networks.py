# MORL_modules/agents/mosac/networks.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import BasePolicy
from gymnasium import spaces
from typing import Dict, List, Tuple, Type, Union
import numpy as np


class MOSACCriticNetwork(nn.Module):
    """
    Individual critic network for one objective.
    Completely separate from other critics.
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            hidden_dims: List[int] = [256, 256],
            activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]

        # Input layer combines observation and action
        input_dim = obs_dim + action_dim

        # Build hidden layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
            ])
            prev_dim = hidden_dim

        # Output layer - single Q-value for this objective
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Q(s,a) for this specific objective

        Args:
            obs: Batch of observations [batch_size, obs_dim]
            action: Batch of actions [batch_size, action_dim]

        Returns:
            Q-values: [batch_size, 1]
        """
        # Concatenate observation and action
        x = torch.cat([obs, action], dim=1)
        return self.network(x)


class MOSACActorNetwork(nn.Module):
    """
    Actor network for multi-objective SAC.
    Outputs mean and log_std for action distribution.
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            hidden_dims: List[int] = [256, 256],
            activation_fn: Type[nn.Module] = nn.ReLU,
            log_std_min: float = -20,
            log_std_max: float = 2,
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]

        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Build hidden layers
        layers = []
        prev_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
            ])
            prev_dim = hidden_dim

        self.feature_network = nn.Sequential(*layers)

        # Output layers
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: outputs action distribution parameters

        Args:
            obs: Batch of observations [batch_size, obs_dim]

        Returns:
            mean: Action means [batch_size, action_dim]
            log_std: Action log standard deviations [batch_size, action_dim]
        """
        features = self.feature_network(obs)

        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)

        # Clip log_std to prevent numerical instability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from the policy

        Args:
            obs: Batch of observations [batch_size, obs_dim]
            deterministic: If True, return mean action (no noise)

        Returns:
            action: Sampled actions [batch_size, action_dim]
            log_prob: Log probabilities [batch_size, 1]
        """
        mean, log_std = self.forward(obs)

        if deterministic:
            # Return mean action for evaluation
            action = torch.tanh(mean)
            # For deterministic actions, log_prob is not meaningful
            log_prob = torch.zeros((obs.shape[0], 1), device=obs.device)
        else:
            # Sample from Gaussian and apply tanh
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)

            # Reparameterization trick
            x_t = normal.rsample()
            action = torch.tanh(x_t)

            # Compute log probability with tanh correction
            log_prob = normal.log_prob(x_t)
            # Tanh correction
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob


class MOSACMultiCritic(nn.Module):
    """
    Multi-objective critic consisting of 4 separate critic networks.
    Each critic evaluates a different objective.
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            num_objectives: int = 4,
            hidden_dims: List[int] = [256, 256],
            activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()

        self.num_objectives = num_objectives

        # Create separate critic networks for each objective
        self.critics = nn.ModuleList([
            MOSACCriticNetwork(
                observation_space=observation_space,
                action_space=action_space,
                hidden_dims=hidden_dims,
                activation_fn=activation_fn,
            )
            for _ in range(num_objectives)
        ])

        # Objective names for tracking
        self.objective_names = [
            "economic", "battery_health", "grid_support", "autonomy"
        ]

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through all critics

        Args:
            obs: Batch of observations [batch_size, obs_dim]
            action: Batch of actions [batch_size, action_dim]

        Returns:
            List of Q-values from each critic [batch_size, 1]
        """
        q_values = []
        for critic in self.critics:
            q_val = critic(obs, action)
            q_values.append(q_val)
        return q_values

    def get_critic_loss(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            target_q_values: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for all critics

        Args:
            obs: Current observations
            action: Current actions
            target_q_values: Target Q-values for each objective

        Returns:
            total_loss: Average loss across all critics
            loss_dict: Individual losses for logging
        """
        current_q_values = self.forward(obs, action)

        losses = []
        loss_dict = {}

        for i, (current_q, target_q) in enumerate(zip(current_q_values, target_q_values)):
            loss = F.mse_loss(current_q, target_q)
            losses.append(loss)
            loss_dict[f"critic_{self.objective_names[i]}_loss"] = loss.item()

        # Simple average of all critic losses
        total_loss = torch.mean(torch.stack(losses))
        loss_dict["total_critic_loss"] = total_loss.item()

        return total_loss, loss_dict


class MOSACPolicy(BasePolicy):
    """
    Multi-Objective SAC Policy
    Contains actor and critics for multi-objective learning
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule,
            num_objectives: int = 4,
            actor_hidden_dims: List[int] = [256, 256],
            critic_hidden_dims: List[int] = [256, 256],
            activation_fn: Type[nn.Module] = nn.ReLU,
            **kwargs
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=None,  # We handle features internally
            **kwargs
        )

        self.num_objectives = num_objectives

        # Create actor network
        self.actor = MOSACActorNetwork(
            observation_space=observation_space,
            action_space=action_space,
            hidden_dims=actor_hidden_dims,
            activation_fn=activation_fn,
        )

        # Create multi-critic (4 separate critics)
        self.critic = MOSACMultiCritic(
            observation_space=observation_space,
            action_space=action_space,
            num_objectives=num_objectives,
            hidden_dims=critic_hidden_dims,
            activation_fn=activation_fn,
        )

        # Create target critics (copy of main critics)
        self.critic_target = MOSACMultiCritic(
            observation_space=observation_space,
            action_space=action_space,
            num_objectives=num_objectives,
            hidden_dims=critic_hidden_dims,
            activation_fn=activation_fn,
        )

        # Initialize target networks with same weights
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Freeze target networks
        for param in self.critic_target.parameters():
            param.requires_grad = False

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Predict action for given observation

        Args:
            observation: Input observation
            deterministic: Whether to use deterministic policy

        Returns:
            action: Predicted action
        """
        action, _ = self.actor.sample_action(observation, deterministic=deterministic)
        return action

    def forward(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Forward pass (same as _predict for compatibility)"""
        return self._predict(observation, deterministic)