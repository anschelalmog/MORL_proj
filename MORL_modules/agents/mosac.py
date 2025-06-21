import numpy as np
import torch as th
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.buffers import ReplayBufferSamples
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.sac.sac import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.type_aliases import GymEnv, Schedule, TensorDict
import gymnasium as gym
from stable_baselines3.common.torch_layers import FlattenExtractor
import pdb
from stable_baselines3.common.type_aliases import GymEnv, Schedule, TensorDict, RolloutReturn
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import polyak_update, should_collect_more_steps
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.preprocessing import get_action_dim
from agents.monets import SharedFeatureQNet, SeparateQNet


class MOContinuousCritic(ContinuousCritic):
    """
    Multi-objective critic network that extends SB3's ContinuousCritic.
    Instead of creating our own neural network implementation, we'll use
    SB3's built-in network creation functions.
    """

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space,
            net_arch: List[int],  num_objectives: int = 2,
            features_extractor_class = FlattenExtractor, features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            share_features_extractor: bool = True, n_critics: int = 2,
            activation_fn: Type[th.nn.Module] = th.nn.ReLU, normalize_images: bool = True,
            share_features_across_objectives: bool = True, features_extractor: Optional[BaseFeaturesExtractor] = None,
                 features_dim: Optional[int] = 2):

        if features_extractor_class is None:
            # Use default extractor (e.g., FlattenExtractor) if none is providedr
            features_extractor_class = FlattenExtractor
        if features_extractor is None:
            # Create the features extractor using the specified class and kwargs
            features_extractor = features_extractor_class(observation_space, **( features_extractor_kwargs or {}))

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            activation_fn=activation_fn,
            n_critics=n_critics,
            features_extractor= features_extractor,
            features_dim = features_extractor.features_dim,
            share_features_extractor = share_features_extractor
        )

        self.features_dim = self.features_extractor.features_dim
        self.action_dim = get_action_dim(action_space)
        self.num_objectives = num_objectives
        self.share_features_across_objectives = share_features_across_objectives
        self.q_networks: list[th.nn.Module] = []


        # Create separate q-networks for each critic ensemble and each objective
        for idx in range(self.n_critics):
            # For each critic ensemble, create multiple heads (one per objective)
            if share_features_across_objectives:
                # Shared features extractor for all objectives
                # Using SB3's create_mlp function rather than custom implementation
                q_net = th.nn.Sequential(
                    *create_mlp(
                        self.features_dim + self.action_dim,
                        net_arch[-1],  # Last layer of shared features
                        net_arch[:-1],  # Hidden layers
                        activation_fn
                    )
                )

                # Create separate output heads for each objective
                q_heads = th.nn.ModuleList([
                    th.nn.Linear(net_arch[-1], 1) for _ in range(num_objectives)
                ])

                shared_q_net = SharedFeatureQNet(q_net, q_heads)
                self.add_module(f"qf{idx}", shared_q_net)
                self.q_networks.append(shared_q_net)
            else:
                # Separate networks for each objective
                objective_nets = th.nn.ModuleList([
                    th.nn.Sequential(
                        *create_mlp(
                            self.features_dim + self.action_dim,
                            1,  # Single output
                            net_arch,
                            activation_fn
                        )
                    ) for _ in range(num_objectives)
                ])
                # Each critic ensemble has its own set of objective networks
                saperate_q_net = SeparateQNet(objective_nets)
                self.add_module(f"qf{idx}", saperate_q_net )
                self.q_networks.append(saperate_q_net)

        # Convert list to ModuleList
        #self.q_networks = th.nn.ModuleList(self.q_networks)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor]:
        """
        Forward pass of the multi-objective critic.
        Returns Q-values for each critic ensemble and each objective.

        Args:
            obs: Observation tensor
            actions: Action tensor

        Returns:
            List of lists: [critic_ensemble][objective] -> q_value tensor
        """
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        return tuple(q_net(features, actions) for q_net in self.q_networks)

    def q_value(self, obs: th.Tensor, actions: th.Tensor, preference_weights: th.Tensor) -> th.Tensor:
        """
        Get scalarized Q-values using preference weights.

        Args:
            obs: Observation tensor
            actions: Action tensor
            preference_weights: Weights for objectives, shape (batch_size, num_objectives) or (num_objectives,)

        Returns:
            List of scalarized Q-values, one for each critic ensemble
        """
        all_critic_values = self.forward(obs, actions)

        # Ensure proper shape for preference weights
        if preference_weights.dim() == 1:
            preference_weights = preference_weights.unsqueeze(0).expand(obs.shape[0], -1)

        # Scalarize Q-values for each critic
        scalarized_q_values = []
        for critic_values in all_critic_values:
            # Stack objective values for easier computation
            # Shape: (batch_size, num_objectives)
            stacked_q_values = th.cat([q_val for q_val in critic_values], dim=1)
            # Compute weighted sum along objective dimension
            # Shape: (batch_size, 1)
            scalarized = th.sum(stacked_q_values * preference_weights, dim=1, keepdim=True)
            scalarized_q_values.append(scalarized)
        # Return list of scalarized Q-values for each critic ensemble
        return scalarized_q_values


class MOSACPolicy(SACPolicy):
    """
    Policy class for MOSAC adapted to leverage SB3's policy infrastructure.
    """
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            num_objectives: int = 4,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[th.nn.Module] = th.nn.ReLU,
            use_sde: bool = False,
            log_std_init: float = -3,
            sde_net_arch: Optional[List[int]] = FlattenExtractor,
            use_expln: bool = False,
            clip_mean: float = 2.0,
            features_extractor_class = None,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            n_critics: int = 2,
            share_features_extractor: bool = True,
            share_features_across_objectives: bool = True,
    ):
        self.num_objectives = num_objectives
        self.share_features_across_objectives = share_features_across_objectives
        if features_extractor_class is None:
            # Use default extractor (e.g., FlattenExtractor) if none is provided
            features_extractor_class = FlattenExtractor




        super().__init__(
            observation_space = observation_space,
            action_space = action_space,
            lr_schedule = lr_schedule,
            net_arch = net_arch,
            activation_fn = activation_fn,
            use_sde = use_sde,
            log_std_init = log_std_init,
            #sde_net_arch = sde_net_arch,
            use_expln = use_expln,
            clip_mean = clip_mean,
            features_extractor_class = features_extractor_class,
            features_extractor_kwargs = features_extractor_kwargs,
            normalize_images = normalize_images,
            optimizer_class = optimizer_class,
            optimizer_kwargs = optimizer_kwargs ,
            n_critics = n_critics,
            share_features_extractor = share_features_extractor,
        )





    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None )  -> MOContinuousCritic:

        """
        Create a multi-objective critic.
        Uses SB3's network structure but with multiple objective heads.

        Returns:
            Multi-objective continuous critic
        """
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs.update({
            "num_objectives": self.num_objectives,
            "share_features_across_objectives": self.share_features_across_objectives,
            "features_extractor_class": self.features_extractor_class,
        })

        return MOContinuousCritic(**critic_kwargs).to(self.device)