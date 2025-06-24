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
from stable_baselines3.common.policies import ContinuousCritic, BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.type_aliases import GymEnv, Schedule, TensorDict
import gymnasium as gym
from stable_baselines3.common.torch_layers import FlattenExtractor
import pdb
from stable_baselines3.common.type_aliases import GymEnv, Schedule, TensorDict, RolloutReturn
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import polyak_update, should_collect_more_steps
from stable_baselines3.common.torch_layers import FlattenExtractor, BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_action_dim
from agents.monets import SharedFeatureQNet, SeparateQNet
from  agents.mobuffers import MOReplayBuffer
from typing import ClassVar
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from stable_baselines3.common.noise import ActionNoise

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

class MOSAC(SAC):
    """
    Multi-Objective Soft Actor-Critic algorithm.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
        "MOSACPolicy": MOSACPolicy,
    }

    def __init__(self,

        env: Union[GymEnv, str],
        policy: Union[str, type[SACPolicy]] = "MOSACPolicy",
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        # Multi-objective specific parameters
        num_objectives=4,
        preference_weights=None,
        ):

        self.num_objectives = num_objectives
        if preference_weights is None:
            self.preference_weights = np.ones(self.num_objectives, dtype=np.float32) / self.num_objectives
        else:
            assert len(preference_weights) == self.num_objectives
            self.preference_weights = np.array(preference_weights, dtype=np.float32) / np.sum(preference_weights)


        # Set default replay buffer class if not specified
        if replay_buffer_class is None:
            replay_buffer_class = MOReplayBuffer

        # Ensure policy_kwargs contain num_objectives
        policy_kwargs = {} if policy_kwargs is None else policy_kwargs.copy()
        policy_kwargs['num_objectives'] = self.num_objectives

        # Ensure replay buffer kwargs contain num_objectives
        replay_buffer_kwargs = {} if replay_buffer_kwargs is None else replay_buffer_kwargs.copy()
        replay_buffer_kwargs['num_objectives'] = self.num_objectives

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model# We'll set up the model ourselves
        )


    def _setup_model(self) -> None:
        """Setup model with multi-objective components."""
        # Set default replay buffer class if not specified
        if self.replay_buffer_class is None:
            self.replay_buffer_class = MOReplayBuffer
        if self.policy_class is None:
            self.policy_class = MOSACPolicy

        super()._setup_model()

        self.preference_weights_tensor = th.FloatTensor(self.preference_weights).to(self.device)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)

        breakpoint()
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []


        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, self._vec_normalize_env)# type: ignore[union-attr]

            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled states
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = self.critic_target(replay_data.next_observations, next_actions)


                # Compute target Q-values for each objective
                # Assuming rewards has shape (batch_size, num_objectives)
                target_q_values = []
                for obj_idx in range(self.num_objectives):
                    # Get Q-values for this objective from all critic ensembles
                    obj_next_q_values = th.cat([
                        ensemble[obj_idx] for ensemble in next_q_values
                    ], dim=1)
                    # Compute target Q-value for this objective
                    obj_next_q_values = obj_next_q_values.min(dim=1, keepdim=True)[0]
                    obj_next_q_values = obj_next_q_values  - ent_coef * next_log_prob.reshape(-1, 1)
                    obj_reward = replay_data.rewards[:, obj_idx].reshape(-1, 1)
                    obj_target = obj_reward + (1 - replay_data.dones) * self.gamma *obj_next_q_values
                    target_q_values.append(obj_target)

                # Scalarize target using preference weights
                #target_q_value = sum(w * tq for w, tq in zip(self.preference_weights, target_q_values))

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss for each objective, use preference weights
            critic_loss = 0.5 * sum( sum(F.mse_loss(obj_current_q, target_q_value[critic_idx][obj_idx])*self.preference_weights[obj_idx] for
                                         obj_idx, obj_current_q in enumerate(current_q_values))
                                    for critic_idx , current_q in enumerate(current_q_values))
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic.q_value(replay_data.observations, actions_pi, self.preference_weights ), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see https://github.com/DLR-RM/stable-baselines3/issues/996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        # Logger is problematic
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coef.cpu().detach().numpy()))
        self.logger.record("train/actor_loss", np.mean(actor_losses))

        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def _store_transition(
            self,
            replay_buffer: MOReplayBuffer,
            buffer_action: np.ndarray,
            new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
            reward: np.ndarray,  # Now vector reward
            dones: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:
        """Store transition in the replay buffer, handling vector rewards."""
        # Store transition
        # self._vec_normalize_env is none if vec normalization is not used
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            new_obs_ = new_obs
            reward_ = reward

        replay_buffer.add(
            self._last_original_obs,
            new_obs_,
            buffer_action,
            reward_,  # Vector reward
            dones,
            infos,
        )
        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_