# MORL_modules/agents/mosac/critics.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Type
import numpy as np
from gymnasium import spaces


class EconomicCritic(nn.Module):
    """
    Specialized critic for economic objective.
    Focuses on profit maximization and cost minimization.
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
        input_dim = obs_dim + action_dim

        # Build network with economic-specific architecture
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
            ])
            prev_dim = hidden_dim

        # Add economic-specific layer that processes price-related features
        self.price_processor = nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2)

        # Final Q-value layer
        self.q_layer = nn.Linear(hidden_dims[-1] + hidden_dims[-1] // 2, 1)

        self.feature_network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass with economic-specific processing"""
        x = torch.cat([obs, action], dim=1)
        features = self.feature_network(x)

        # Process price-related information
        price_features = torch.relu(self.price_processor(features))

        # Combine features
        combined_features = torch.cat([features, price_features], dim=1)

        return self.q_layer(combined_features)


class BatteryHealthCritic(nn.Module):
    """
    Specialized critic for battery health objective.
    Focuses on minimizing battery degradation and optimizing charge cycles.
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
        input_dim = obs_dim + action_dim

        # Build network with battery-specific architecture
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
            ])
            prev_dim = hidden_dim

        # Battery-specific processing layers
        self.soc_processor = nn.Linear(hidden_dims[-1], hidden_dims[-1] // 4)
        self.cycle_processor = nn.Linear(hidden_dims[-1], hidden_dims[-1] // 4)

        # Final Q-value layer
        combined_dim = hidden_dims[-1] + 2 * (hidden_dims[-1] // 4)
        self.q_layer = nn.Linear(combined_dim, 1)

        self.feature_network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass with battery health-specific processing"""
        x = torch.cat([obs, action], dim=1)
        features = self.feature_network(x)

        # Process state-of-charge related features
        soc_features = torch.relu(self.soc_processor(features))

        # Process charge cycle related features
        cycle_features = torch.relu(self.cycle_processor(features))

        # Combine all features
        combined_features = torch.cat([features, soc_features, cycle_features], dim=1)

        return self.q_layer(combined_features)


class GridSupportCritic(nn.Module):
    """
    Specialized critic for grid support objective.
    Focuses on grid stability and demand response.
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
        input_dim = obs_dim + action_dim

        # Build network with grid-specific architecture
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
            ])
            prev_dim = hidden_dim

        # Grid-specific processing layers
        self.demand_processor = nn.Linear(hidden_dims[-1], hidden_dims[-1] // 3)
        self.frequency_processor = nn.Linear(hidden_dims[-1], hidden_dims[-1] // 3)

        # Final Q-value layer
        combined_dim = hidden_dims[-1] + 2 * (hidden_dims[-1] // 3)
        self.q_layer = nn.Linear(combined_dim, 1)

        self.feature_network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass with grid support-specific processing"""
        x = torch.cat([obs, action], dim=1)
        features = self.feature_network(x)

        # Process demand-related features
        demand_features = torch.relu(self.demand_processor(features))

        # Process frequency regulation features
        freq_features = torch.relu(self.frequency_processor(features))

        # Combine features
        combined_features = torch.cat([features, demand_features, freq_features], dim=1)

        return self.q_layer(combined_features)


class AutonomyCritic(nn.Module):
    """
    Specialized critic for energy autonomy objective.
    Focuses on self-sufficiency and energy independence.
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
        input_dim = obs_dim + action_dim

        # Build network with autonomy-specific architecture
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
            ])
            prev_dim = hidden_dim

        # Autonomy-specific processing layers
        self.storage_processor = nn.Linear(hidden_dims[-1], hidden_dims[-1] // 3)
        self.generation_processor = nn.Linear(hidden_dims[-1], hidden_dims[-1] // 3)

        # Final Q-value layer
        combined_dim = hidden_dims[-1] + 2 * (hidden_dims[-1] // 3)
        self.q_layer = nn.Linear(combined_dim, 1)

        self.feature_network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass with autonomy-specific processing"""
        x = torch.cat([obs, action], dim=1)
        features = self.feature_network(x)

        # Process energy storage features
        storage_features = torch.relu(self.storage_processor(features))

        # Process generation/consumption balance features
        generation_features = torch.relu(self.generation_processor(features))

        # Combine features
        combined_features = torch.cat([features, storage_features, generation_features], dim=1)

        return self.q_layer(combined_features)


class SpecializedMOSACMultiCritic(nn.Module):
    """
    Multi-objective critic using specialized critics for each objective.
    Each critic is designed with domain knowledge for its specific objective.
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            hidden_dims: List[int] = [256, 256],
            activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()

        self.num_objectives = 4
        self.objective_names = ["economic", "battery_health", "grid_support", "autonomy"]

        # Create specialized critics
        self.economic_critic = EconomicCritic(
            observation_space, action_space, hidden_dims, activation_fn
        )

        self.battery_health_critic = BatteryHealthCritic(
            observation_space, action_space, hidden_dims, activation_fn
        )

        self.grid_support_critic = GridSupportCritic(
            observation_space, action_space, hidden_dims, activation_fn
        )

        self.autonomy_critic = AutonomyCritic(
            observation_space, action_space, hidden_dims, activation_fn
        )

        # Store critics in a list for easier iteration
        self.critics = [
            self.economic_critic,
            self.battery_health_critic,
            self.grid_support_critic,
            self.autonomy_critic
        ]

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through all specialized critics"""
        q_values = []
        for critic in self.critics:
            q_val = critic(obs, action)
            q_values.append(q_val)
        return q_values

    def get_critic_loss(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            target_q_values: List[torch.Tensor],
            preference_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute adaptive loss for all critics"""
        current_q_values = self.forward(obs, action, preference_weights)

        losses = []
        loss_dict = {}

        for i, (current_q, target_q) in enumerate(zip(current_q_values, target_q_values)):
            loss = F.mse_loss(current_q, target_q)
            losses.append(loss)
            loss_dict[f"critic_{self.objective_names[i]}_loss"] = loss.item()

        # Adaptive loss weighting based on preferences
        if preference_weights is not None and self.use_attention:
            # Weight losses by preference
            pref_weights = preference_weights.mean(dim=0)  # Average across batch
            weighted_losses = [w * loss for w, loss in zip(pref_weights, losses)]
            total_loss = sum(weighted_losses)
        else:
            total_loss = torch.mean(torch.stack(losses))

        loss_dict["total_critic_loss"] = total_loss.item()

        return total_loss, loss_dict


class HierarchicalMOSACCritic(nn.Module):
    """
    Hierarchical multi-objective critic that first learns a shared representation
    and then specializes for each objective.
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            num_objectives: int = 4,
            shared_hidden_dims: List[int] = [512, 256],
            specialist_hidden_dims: List[int] = [128, 64],
            activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()

        self.num_objectives = num_objectives
        self.objective_names = ["economic", "battery_health", "grid_support", "autonomy"]

        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        input_dim = obs_dim + action_dim

        # Shared representation learning
        shared_layers = []
        prev_dim = input_dim

        for hidden_dim in shared_hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
                nn.Dropout(0.1),  # Regularization
            ])
            prev_dim = hidden_dim

        self.shared_network = nn.Sequential(*shared_layers)

        # Specialist networks for each objective
        self.specialist_networks = nn.ModuleList()

        for _ in range(num_objectives):
            specialist_layers = []
            specialist_prev_dim = prev_dim

            for hidden_dim in specialist_hidden_dims:
                specialist_layers.extend([
                    nn.Linear(specialist_prev_dim, hidden_dim),
                    activation_fn(),
                ])
                specialist_prev_dim = hidden_dim

            # Output layer
            specialist_layers.append(nn.Linear(specialist_prev_dim, 1))

            self.specialist_networks.append(nn.Sequential(*specialist_layers))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> List[torch.Tensor]:
        """Hierarchical forward pass"""
        x = torch.cat([obs, action], dim=1)

        # Get shared representation
        shared_features = self.shared_network(x)

        # Process through specialist networks
        q_values = []
        for specialist in self.specialist_networks:
            q_val = specialist(shared_features)
            q_values.append(q_val)

        return q_values

    def get_shared_features(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get shared feature representation"""
        x = torch.cat([obs, action], dim=1)
        return self.shared_network(x)

    def get_critic_loss(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            target_q_values: List[torch.Tensor],
            regularization_weight: float = 0.01,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute hierarchical loss with regularization"""
        current_q_values = self.forward(obs, action)

        losses = []
        loss_dict = {}

        # Compute Q-value losses
        for i, (current_q, target_q) in enumerate(zip(current_q_values, target_q_values)):
            loss = F.mse_loss(current_q, target_q)
            losses.append(loss)
            loss_dict[f"critic_{self.objective_names[i]}_loss"] = loss.item()

        # Main loss
        main_loss = torch.mean(torch.stack(losses))

        # Regularization: encourage diverse specialist representations
        if regularization_weight > 0:
            shared_features = self.get_shared_features(obs, action)
            specialist_features = []

            for specialist in self.specialist_networks:
                # Get intermediate representation from each specialist
                for layer in specialist[:-1]:  # Exclude output layer
                    shared_features = layer(shared_features)
                specialist_features.append(shared_features)

            # Diversity regularization
            diversity_loss = 0.0
            for i in range(len(specialist_features)):
                for j in range(i + 1, len(specialist_features)):
                    # Penalize high correlation between specialist features
                    correlation = F.cosine_similarity(
                        specialist_features[i], specialist_features[j], dim=1
                    ).mean()
                    diversity_loss += correlation.abs()

            total_loss = main_loss + regularization_weight * diversity_loss
            loss_dict["diversity_loss"] = diversity_loss.item()
        else:
            total_loss = main_loss

        loss_dict["total_critic_loss"] = total_loss.item()
        loss_dict["main_loss"] = main_loss.item()

        return total_loss, loss_dict


def create_mosac_critic(
        critic_type: str,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        **kwargs
) -> nn.Module:
    """
    Factory function to create different types of MOSAC critics

    Args:
        critic_type: Type of critic ("standard", "specialized", "adaptive", "hierarchical")
        observation_space: Environment observation space
        action_space: Environment action space
        **kwargs: Additional arguments for critic initialization

    Returns:
        MOSAC critic network
    """
    if critic_type == "standard":
        from .networks import MOSACMultiCritic
        return MOSACMultiCritic(observation_space, action_space, **kwargs)

    elif critic_type == "specialized":
        return SpecializedMOSACMultiCritic(observation_space, action_space, **kwargs)

    elif critic_type == "adaptive":
        return AdaptiveMOSACMultiCritic(observation_space, action_space, **kwargs)

    elif critic_type == "hierarchical":
        return HierarchicalMOSACCritic(observation_space, action_space, **kwargs)

    else:
        raise ValueError(f"Unknown critic type: {critic_type}")


class CriticEnsemble(nn.Module):
    """
    Ensemble of different critic types for robust multi-objective learning
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            critic_types: List[str] = ["standard", "specialized"],
            ensemble_method: str = "average",  # "average", "weighted", "voting"
            **kwargs
    ):
        super().__init__()

        self.num_objectives = 4
        self.objective_names = ["economic", "battery_health", "grid_support", "autonomy"]
        self.ensemble_method = ensemble_method

        # Create ensemble of critics
        self.critics = nn.ModuleList([
            create_mosac_critic(critic_type, observation_space, action_space, **kwargs)
            for critic_type in critic_types
        ])

        # Learnable ensemble weights
        if ensemble_method == "weighted":
            self.ensemble_weights = nn.Parameter(torch.ones(len(self.critics)) / len(self.critics))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> List[torch.Tensor]:
        """Ensemble forward pass"""
        # Get predictions from all critics
        all_predictions = []
        for critic in self.critics:
            q_values = critic(obs, action)
            all_predictions.append(q_values)

        # Combine predictions
        if self.ensemble_method == "average":
            # Simple average
            ensemble_q_values = []
            for obj_idx in range(self.num_objectives):
                obj_predictions = [pred[obj_idx] for pred in all_predictions]
                avg_q = torch.mean(torch.stack(obj_predictions), dim=0)
                ensemble_q_values.append(avg_q)

        elif self.ensemble_method == "weighted":
            # Weighted average
            weights = F.softmax(self.ensemble_weights, dim=0)
            ensemble_q_values = []
            for obj_idx in range(self.num_objectives):
                obj_predictions = [pred[obj_idx] for pred in all_predictions]
                weighted_q = sum(w * pred for w, pred in zip(weights, obj_predictions))
                ensemble_q_values.append(weighted_q)

        else:  # voting
            # Majority voting (for discrete actions) or median (for continuous)
            ensemble_q_values = []
            for obj_idx in range(self.num_objectives):
                obj_predictions = torch.stack([pred[obj_idx] for pred in all_predictions])
                median_q = torch.median(obj_predictions, dim=0)[0]
                ensemble_q_values.append(median_q)

        return ensemble_q_values

    def get_critic_loss(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            target_q_values: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute ensemble loss"""
        # Get individual critic losses
        individual_losses = []
        loss_dict = {}

        for i, critic in enumerate(self.critics):
            if hasattr(critic, 'get_critic_loss'):
                loss, critic_loss_dict = critic.get_critic_loss(obs, action, target_q_values)
                individual_losses.append(loss)

                # Add critic-specific losses to dict
                for key, value in critic_loss_dict.items():
                    loss_dict[f"critic_{i}_{key}"] = value

        # Ensemble loss
        if individual_losses:
            total_loss = torch.mean(torch.stack(individual_losses))
        else:
            # Fallback: compute ensemble loss directly
            ensemble_q_values = self.forward(obs, action)
            losses = []
            for current_q, target_q in zip(ensemble_q_values, target_q_values):
                loss = F.mse_loss(current_q, target_q)
                losses.append(loss)
            total_loss = torch.mean(torch.stack(losses))

        loss_dict["ensemble_total_loss"] = total_loss.item()

        return total_loss, loss_dict
        loss_weights: Optional[torch.Tensor] = None,

    ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute loss for all specialized critics"""
    current_q_values = self.forward(obs, action)

    losses = []
    loss_dict = {}

    for i, (current_q, target_q) in enumerate(zip(current_q_values, target_q_values)):
        loss = F.mse_loss(current_q, target_q)
    losses.append(loss)
    loss_dict[f"critic_{self.objective_names[i]}_loss"] = loss.item()

    # Compute total loss
    if loss_weights is not None:
        weighted_losses = [w * loss for w, loss in zip(loss_weights, losses)]
    total_loss = sum(weighted_losses)
    else:
    total_loss = torch.mean(torch.stack(losses))

    loss_dict["total_critic_loss"] = total_loss.item()

    return total_loss, loss_dict

    def get_objective_q_values(self, obs: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get Q-values with objective names"""
        q_values = self.forward(obs, action)
        return {name: q_val for name, q_val in zip(self.objective_names, q_values)}


class AdaptiveMOSACMultiCritic(nn.Module):
    """
    Adaptive multi-objective critic that can adjust attention to different objectives
    based on current preference weights or training phase.
    """

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            num_objectives: int = 4,
            hidden_dims: List[int] = [256, 256],
            activation_fn: Type[nn.Module] = nn.ReLU,
            use_attention: bool = True,
    ):
        super().__init__()

        self.num_objectives = num_objectives
        self.use_attention = use_attention
        self.objective_names = ["economic", "battery_health", "grid_support", "autonomy"]

        obs_dim = observation_space.shape[0]
        action_dim = action_space.shape[0]
        input_dim = obs_dim + action_dim

        # Shared feature extraction
        shared_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims[:-1]:  # All but last layer
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn(),
            ])
            prev_dim = hidden_dim

        self.shared_network = nn.Sequential(*shared_layers)

        # Individual critic heads
        self.critic_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1]),
                activation_fn(),
                nn.Linear(hidden_dims[-1], 1)
            )
            for _ in range(num_objectives)
        ])

        # Attention mechanism for objective weighting
        if use_attention:
            self.attention_network = nn.Sequential(
                nn.Linear(prev_dim, hidden_dims[-1] // 2),
                nn.ReLU(),
                nn.Linear(hidden_dims[-1] // 2, num_objectives),
                nn.Softmax(dim=-1)
            )

    def forward(self, obs: torch.Tensor, action: torch.Tensor,
                preference_weights: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Forward pass with optional preference-aware attention

        Args:
            obs: Observations
            action: Actions
            preference_weights: Optional preference weights [batch_size, num_objectives]

        Returns:
            List of Q-values for each objective
        """
        x = torch.cat([obs, action], dim=1)
        shared_features = self.shared_network(x)

        # Get Q-values from each head
        q_values = []
        for head in self.critic_heads:
            q_val = head(shared_features)
            q_values.append(q_val)

        # Apply attention if enabled
        if self.use_attention:
            if preference_weights is not None:
                # Use provided preference weights
                attention_weights = preference_weights
            else:
                # Learn attention weights
                attention_weights = self.attention_network(shared_features)

            # Apply attention to Q-values (optional, for interpretation)
            # Note: This doesn't change the Q-values themselves, just provides attention info
            self.last_attention_weights = attention_weights

        return q_values

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get last computed attention weights"""
        return getattr(self, 'last_attention_weights', None)

    def get_critic_loss(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            target_q_values: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute ensemble loss"""
        # Get individual critic losses
        individual_losses = []
        loss_dict = {}

        for i, critic in enumerate(self.critics):
            if hasattr(critic, 'get_critic_loss'):
                loss, critic_loss_dict = critic.get_critic_loss(obs, action, target_q_values)
                individual_losses.append(loss)

                # Add critic-specific losses to dict
                for key, value in critic_loss_dict.items():
                    loss_dict[f"critic_{i}_{key}"] = value

        # Ensemble loss
        if individual_losses:
            total_loss = torch.mean(torch.stack(individual_losses))
        else:
            # Fallback: compute ensemble loss directly
            ensemble_q_values = self.forward(obs, action)
            losses = []
            for current_q, target_q in zip(ensemble_q_values, target_q_values):
                loss = F.mse_loss(current_q, target_q)
                losses.append(loss)
            total_loss = torch.mean(torch.stack(losses))

        loss_dict["ensemble_total_loss"] = total_loss.item()

        return total_loss, loss_dict