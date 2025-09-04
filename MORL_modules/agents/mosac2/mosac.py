# MORL_modules/agents/mosac/mosac.py

import torch
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import get_schedule_fn, polyak_update
from stable_baselines3.common.callbacks import BaseCallback
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import logging
from collections import defaultdict

from .networks import MOSACPolicy
from .replay_buffer import MOSACReplayBuffer


class MOSAC(OffPolicyAlgorithm):
    """
    Multi-Objective Soft Actor-Critic (MOSAC)

    SAC with multiple critics for different objectives.
    Uses simple averaging of critic gradients for actor updates.
    """

    def __init__(
            self,
            policy: Union[str, Type[MOSACPolicy]] = MOSACPolicy,
            env: Union[GymEnv, str] = None,
            learning_rate: Union[float, Schedule] = 3e-4,
            buffer_size: int = 1_000_000,
            learning_starts: int = 100,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = 1,
            gradient_steps: int = 1,
            action_noise=None,
            replay_buffer_class: Optional[Type] = None,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            ent_coef: Union[str, float] = "auto",
            target_update_interval: int = 1,
            target_entropy: Union[str, float] = "auto",
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            use_sde_at_warmup: bool = False,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[torch.device, str] = "auto",
            _init_setup_model: bool = True,
            num_objectives: int = 4,
            critic_lr: Optional[float] = None,
            actor_lr: Optional[float] = None,
    ):

        # Set up multi-objective specific parameters
        self.num_objectives = num_objectives
        self.objective_names = ["economic", "battery_health", "grid_support", "autonomy"]

        # Learning rates
        self.critic_lr = critic_lr if critic_lr is not None else learning_rate
        self.actor_lr = actor_lr if actor_lr is not None else learning_rate

        # SAC specific parameters
        self.target_update_interval = target_update_interval
        self.ent_coef = ent_coef
        self.target_entropy = target_entropy
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        self.use_sde_at_warmup = use_sde_at_warmup

        # Set up custom replay buffer
        if replay_buffer_class is None:
            replay_buffer_class = MOSACReplayBuffer

        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {}
        replay_buffer_kwargs.update({"num_objectives": num_objectives})

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
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )

        # Tracking for multi-objective rewards
        self.objective_tracking = defaultdict(list)
        self.episode_rewards = np.zeros(num_objectives)
        self.episode_length = 0

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        """Set up model components"""
        super()._setup_model()

        # Set up entropy coefficient
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Auto tune entropy coefficient
            if self.target_entropy == "auto":
                self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
            else:
                self.target_entropy = float(self.target_entropy)

            # Create learnable entropy coefficient
            self.log_ent_coef = torch.log(torch.ones(1, device=self.device) * 0.1).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.learning_rate)
        else:
            self.ent_coef = float(self.ent_coef)
            self.log_ent_coef = None

        # Set up optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(),
            lr=self.actor_lr
        )

        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(),
            lr=self.critic_lr
        )

    def _store_transition(
            self,
            replay_buffer,
            buffer_action: np.ndarray,
            new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
            reward: np.ndarray,
            dones: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in replay buffer, handling vector rewards
        """
        # Handle vector rewards from environment
        if isinstance(reward, dict) and 'mo_rewards' in reward:
            # Multi-objective reward from wrapper
            vector_reward = reward['mo_rewards']
        elif hasattr(reward, '__len__') and len(reward) == self.num_objectives:
            # Direct vector reward
            vector_reward = np.array(reward, dtype=np.float32)
        else:
            # Scalar reward - convert to vector (put in first objective)
            vector_reward = np.zeros(self.num_objectives, dtype=np.float32)
            vector_reward[0] = float(reward)

        # Update episode tracking
        self.episode_rewards += vector_reward
        self.episode_length += 1

        # Store transition with vector reward
        super()._store_transition(
            replay_buffer, buffer_action, new_obs, vector_reward, dones, infos
        )

        # Log episode completion
        if dones[0]:  # Episode finished
            for i, obj_name in enumerate(self.objective_names):
                self.objective_tracking[obj_name].append(self.episode_rewards[i])
            self.objective_tracking['episode_length'].append(self.episode_length)

            # Reset tracking
            self.episode_rewards = np.zeros(self.num_objectives)
            self.episode_length = 0

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Update policy and critics
        """
        # Switch to train mode
        self.policy.set_training_mode(True)

        # Update optimizers learning rates
        optimizers = [self.actor_optimizer, self.critic_optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Train critics
            critic_loss, critic_loss_dict = self._train_critics(replay_data)
            critic_losses.append(critic_loss_dict)

            # Train actor and entropy coefficient
            if gradient_step % self.target_update_interval == 0:
                actor_loss = self._train_actor(replay_data)
                actor_losses.append(actor_loss)

                # Update target networks
                polyak_update(
                    self.policy.critic.parameters(),
                    self.policy.critic_target.parameters(),
                    self.tau
                )

                # Train entropy coefficient if auto
                if self.log_ent_coef is not None:
                    ent_coef_loss, ent_coef = self._train_entropy_coefficient(replay_data)
                    ent_coef_losses.append(ent_coef_loss)
                    ent_coefs.append(ent_coef)

        # Log training metrics
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(critic_losses) > 0:
            # Average critic losses
            for key in critic_losses[0].keys():
                avg_loss = np.mean([loss[key] for loss in critic_losses])
                self.logger.record(f"train/{key}", avg_loss)

        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))

        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
            self.logger.record("train/ent_coef", np.mean(ent_coefs))

        # Log objective-specific rewards
        for obj_name in self.objective_names:
            if len(self.objective_tracking[obj_name]) > 0:
                recent_rewards = self.objective_tracking[obj_name][-10:]  # Last 10 episodes
                self.logger.record(f"rollout/ep_{obj_name}_reward_mean", np.mean(recent_rewards))

    def _train_critics(self, replay_data) -> Tuple[float, Dict[str, float]]:
        """Train all critics"""
        obs = replay_data["observations"]
        actions = replay_data["actions"]
        next_obs = replay_data["next_observations"]
        rewards = replay_data["rewards"]  # [batch_size, num_objectives]
        dones = replay_data["dones"]

        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs = self.policy.actor.sample_action(next_obs)

            # Compute target Q-values for all objectives
            next_q_values = self.policy.critic_target(next_obs, next_actions)

            # Apply entropy regularization
            ent_coef = self.ent_coef
            if self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef)

            target_q_values = []
            for i, next_q in enumerate(next_q_values):
                target_q = rewards[:, i:i + 1] + (1 - dones) * self.gamma * (next_q - ent_coef * next_log_probs)
                target_q_values.append(target_q)

        # Compute critic losses
        total_loss, loss_dict = self.policy.critic.get_critic_loss(obs, actions, target_q_values)

        # Update critics
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        self.critic_optimizer.step()

        return total_loss.item(), loss_dict

    def _train_actor(self, replay_data) -> float:
        """Train actor using average of critic gradients"""
        obs = replay_data["observations"]

        # Sample actions from current policy
        actions, log_probs = self.policy.actor.sample_action(obs)

        # Get Q-values from all critics
        q_values = self.policy.critic(obs, actions)

        # Simple average of Q-values from all critics
        avg_q_value = torch.mean(torch.stack(q_values), dim=0)

        # Actor loss: maximize Q - entropy
        ent_coef = self.ent_coef
        if self.log_ent_coef is not None:
            ent_coef = torch.exp(self.log_ent_coef)

        actor_loss = -(avg_q_value - ent_coef * log_probs).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def _train_entropy_coefficient(self, replay_data) -> Tuple[float, float]:
        """Train entropy coefficient if auto"""
        if self.log_ent_coef is None:
            return 0.0, self.ent_coef

        obs = replay_data["observations"]

        with torch.no_grad():
            _, log_probs = self.policy.actor.sample_action(obs)

        ent_coef_loss = -(self.log_ent_coef * (log_probs + self.target_entropy)).mean()

        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()

        return ent_coef_loss.item(), torch.exp(self.log_ent_coef).item()

    def learn(
            self,
            total_timesteps: int,
            callback=None,
            log_interval: int = 4,
            tb_log_name: str = "MOSAC",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ):
        """Learn policy using multi-objective SAC"""
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Predict action using trained policy
        """
        return self.policy.predict(observation, state, episode_start, deterministic)

    def get_objective_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each objective"""
        stats = {}
        for obj_name in self.objective_names:
            rewards = self.objective_tracking[obj_name]
            if len(rewards) > 0:
                stats[obj_name] = {
                    'mean': np.mean(rewards),
                    'std': np.std(rewards),
                    'min': np.min(rewards),
                    'max': np.max(rewards),
                    'episodes': len(rewards)
                }
        return stats