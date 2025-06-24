import pytest
import numpy as np
import torch as th
from gymnasium import spaces
from copy import deepcopy
from torch.nn.functional import mse_loss

# Import your custom classes
from agents.mosac import MOSACPolicy, MOContinuousCritic
from agents.mobuffers import MOReplayBuffer

class TestMOSACPolicyTraining:
    """Test suite for MOSACPolicy focusing on actor and critic losses."""

    @pytest.fixture
    def training_setup(self):
        """Create a setup for training tests with all necessary components."""
        # Define spaces
        observation_space = spaces.Box(low=-10, high=10, shape=(5,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Number of objectives
        num_objectives = 3

        # Create policy
        policy = MOSACPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda _: 3e-4,
            num_objectives=num_objectives,
            net_arch=[64, 64],
            n_critics=2,
            share_features_across_objectives=True
        )

        # Create target policy (in SAC, we have target networks for the critics)
        target_policy = deepcopy(policy)

        # Generate mock batch data
        batch_size = 32
        obs = th.FloatTensor(np.random.uniform(-1, 1, size=(batch_size, 5)))
        actions = th.FloatTensor(np.random.uniform(-1, 1, size=(batch_size, 2)))
        rewards = th.FloatTensor(np.random.uniform(-1, 1, size=(batch_size, num_objectives)))
        next_obs = th.FloatTensor(np.random.uniform(-1, 1, size=(batch_size, 5)))
        dones = th.FloatTensor(np.random.choice([0, 1], size=(batch_size, 1), p=[0.9, 0.1]))

        # Generate preference weights (one per sample in batch)
        preference_weights = th.FloatTensor(np.random.uniform(0, 1, size=(batch_size, num_objectives)))
        preference_weights = preference_weights / preference_weights.sum(dim=1, keepdim=True)  # Normalize

        # Common hyperparameters for SAC
        gamma = 0.99
        ent_coef = 0.1
        target_entropy = -float(action_space.shape[0])  # Typical heuristic

        return {
            "policy": policy,
            "target_policy": target_policy,
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones,
            "preference_weights": preference_weights,
            "gamma": gamma,
            "ent_coef": ent_coef,
            "target_entropy": target_entropy,
            "batch_size": batch_size,
            "num_objectives": num_objectives
        }

    def test_actor_loss_calculation(self, training_setup):
        """Test actor loss calculation for multi-objective SAC."""
        policy = training_setup["policy"]
        obs = training_setup["obs"]
        preference_weights = training_setup["preference_weights"]
        ent_coef = training_setup["ent_coef"]
        batch_size = training_setup["batch_size"]

        # 1. Sample actions from the current policy
        actions, log_probs = policy.actor.action_log_prob(obs)

        # 2. Get Q-values from critic - directly using the critic method
        critic_outputs = policy.critic(obs, actions)

        # 3. Manually calculate scalarized Q-values
        # Structure: critic_outputs is a tuple of (critic_ensemble1, critic_ensemble2, ...)
        # Each critic_ensemble is a tuple of (objective1, objective2, ...)
        # Each objective is a tensor of shape (batch_size, 1)

        # Compute scalarized values for each critic ensemble
        scalarized_q_values = []

        for critic_ensemble in critic_outputs:
            # Stack objective values: (batch_size, num_objectives)
            stacked_values = th.cat([obj_val for obj_val in critic_ensemble], dim=1)

            # Compute weighted sum: (batch_size, 1)
            scalarized = th.sum(stacked_values * preference_weights, dim=1, keepdim=True)
            scalarized_q_values.append(scalarized)

        # 4. Compute the minimum Q-value across critic ensembles (SAC uses min for pessimism)
        min_qf_pi = th.cat(scalarized_q_values, dim=1).min(dim=1, keepdim=True)[0]

        # 5. Compute actor loss
        actor_loss = (ent_coef * log_probs - min_qf_pi).mean()

        # Verify loss has reasonable values and requires gradients
        assert not th.isnan(actor_loss), "Actor loss contains NaN values"
        assert not th.isinf(actor_loss), "Actor loss contains infinite values"
        assert actor_loss.requires_grad, "Actor loss doesn't require gradients"

        # 6. Verify gradients flow through the actor network
        actor_loss.backward()

        # Check at least some actor parameters received gradients
        has_grad = False
        for param in policy.actor.parameters():
            if param.grad is not None and th.any(param.grad != 0):
                has_grad = True
                break

        assert has_grad, "No actor parameters received nonzero gradients"

        # Optional: Check magnitude of gradients is reasonable
        grad_norm = sum(
            th.norm(param.grad) ** 2 for param in policy.actor.parameters()
            if param.grad is not None
        ).sqrt()

        assert 1e-6 < grad_norm < 1e6, f"Actor gradient norm {grad_norm} is outside reasonable range"

        # Print debug info
        print(f"Actor loss: {actor_loss.item():.6f}, gradient norm: {grad_norm.item():.6f}")

    def test_critic_loss_calculation(self, training_setup):
        """Test critic loss calculation for multi-objective SAC."""
        policy = training_setup["policy"]
        target_policy = training_setup["target_policy"]
        obs = training_setup["obs"]
        actions = training_setup["actions"]
        rewards = training_setup["rewards"]
        next_obs = training_setup["next_obs"]
        dones = training_setup["dones"]
        preference_weights = training_setup["preference_weights"]
        gamma = training_setup["gamma"]
        ent_coef = training_setup["ent_coef"]
        num_objectives = training_setup["num_objectives"]

        # 1. Get current Q-values from critic - directly using the critic method
        current_q_values = policy.critic(obs, actions)

        # 2. Sample next actions from the current policy (for target calculation)
        next_actions, next_log_probs = policy.actor.action_log_prob(next_obs)

        # 3. Get target Q-values using target critic
        with th.no_grad():
            target_q_values = target_policy.critic(next_obs, next_actions)

            # 3a. Calculate minimum target Q-value for each objective separately
            min_target_q_values = []

            # For each objective
            for obj_idx in range(num_objectives):
                # Get Q-values for this objective from all critic ensembles
                obj_q_values = th.cat([
                    ensemble[obj_idx] for ensemble in target_q_values
                ], dim=1)

                # Min across critic ensembles
                min_obj_q = obj_q_values.min(dim=1, keepdim=True)[0]
                breakpoint()
                min_target_q_values.append(min_obj_q)

            # 3b. Calculate entropy term for target
            next_entropy = -next_log_probs.reshape(-1, 1)

            # 3c. Compute TD targets for each objective
            td_targets = []

            for obj_idx in range(num_objectives):
                # Get rewards for this objective
                obj_rewards = rewards[:, obj_idx].reshape(-1, 1)

                # Compute target with entropy regularization
                # Note: We apply entropy to each objective equally for simplicity
                target = obj_rewards + gamma * (1.0 - dones) * (
                        min_target_q_values[obj_idx] + ent_coef * next_entropy
                )

                td_targets.append(target)

        # 4. Compute critic loss for each critic ensemble and each objective
        critic_losses = []

        for critic_idx, critic_ensemble in enumerate(current_q_values):
            ensemble_losses = []

            for obj_idx, obj_q_values in enumerate(critic_ensemble):
                # MSE loss between current Q-values and targets
                obj_loss = mse_loss(obj_q_values, td_targets[obj_idx])
                ensemble_losses.append(obj_loss)

            # Combine losses across objectives using preference weights
            # Average preference weights across batch for simplicity
            avg_weights = preference_weights.mean(dim=0)
            weighted_loss = sum(loss * weight for loss, weight in zip(ensemble_losses, avg_weights))
            critic_losses.append(weighted_loss)

        # 5. Total critic loss (sum across all critic ensembles)
        critic_loss = sum(critic_losses)

        # Verify loss has reasonable values and requires gradients
        assert not th.isnan(critic_loss), "Critic loss contains NaN values"
        assert not th.isinf(critic_loss), "Critic loss contains infinite values"
        assert critic_loss.requires_grad, "Critic loss doesn't require gradients"

        # 6. Verify gradients flow through the critic network
        critic_loss.backward()

        # Check at least some critic parameters received gradients
        has_grad = False
        for param in policy.critic.parameters():
            if param.grad is not None and th.any(param.grad != 0):
                has_grad = True
                break

        assert has_grad, "No critic parameters received nonzero gradients"

        # Optional: Check magnitude of gradients is reasonable
        grad_norm = sum(
            th.norm(param.grad) ** 2 for param in policy.critic.parameters()
            if param.grad is not None
        ).sqrt()

        assert 1e-6 < grad_norm < 1e6, f"Critic gradient norm {grad_norm} is outside reasonable range"

        # Print debug info
        print(f"Critic loss: {critic_loss.item():.6f}, gradient norm: {grad_norm.item():.6f}")

    def test_full_training_step(self, training_setup):
        """Test a full training step including both actor and critic updates."""
        policy = training_setup["policy"]
        target_policy = training_setup["target_policy"]
        obs = training_setup["obs"]
        actions = training_setup["actions"]
        rewards = training_setup["rewards"]
        next_obs = training_setup["next_obs"]
        dones = training_setup["dones"]
        preference_weights = training_setup["preference_weights"]
        gamma = training_setup["gamma"]
        ent_coef = training_setup["ent_coef"]

        # Create optimizers
        critic_optimizer = th.optim.Adam(policy.critic.parameters(), lr=3e-4)
        actor_optimizer = th.optim.Adam(policy.actor.parameters(), lr=3e-4)

        # Save initial parameters for comparison
        initial_actor_params = {name: param.clone().detach()
                                for name, param in policy.actor.named_parameters()}
        initial_critic_params = {name: param.clone().detach()
                                 for name, param in policy.critic.named_parameters()}

        # 1. Critic update
        # 1a. Zero gradients
        critic_optimizer.zero_grad()

        # 1b. Get current Q-values
        current_q_values = policy.critic(obs, actions)

        # 1c. Sample next actions from the current policy
        next_actions, next_log_probs = policy.actor.action_log_prob(next_obs)

        # 1d. Get target Q-values
        with th.no_grad():
            target_q_values = target_policy.critic(next_obs, next_actions)

            # Calculate minimum target Q-value for each objective separately
            min_target_q_values = []

            # For each objective
            for obj_idx in range(training_setup["num_objectives"]):
                # Get Q-values for this objective from all critic ensembles
                obj_q_values = th.cat([
                    ensemble[obj_idx] for ensemble in target_q_values
                ], dim=1)

                # Min across critic ensembles
                min_obj_q = obj_q_values.min(dim=1, keepdim=True)[0]
                min_target_q_values.append(min_obj_q)

            # Calculate entropy term for target
            next_entropy = -next_log_probs.reshape(-1, 1)

            # Compute TD targets for each objective
            td_targets = []

            for obj_idx in range(training_setup["num_objectives"]):
                # Get rewards for this objective
                obj_rewards = rewards[:, obj_idx].reshape(-1, 1)

                # Compute target with entropy regularization
                target = obj_rewards + gamma * (1.0 - dones) * (
                        min_target_q_values[obj_idx] + ent_coef * next_entropy
                )

                td_targets.append(target)

        # 1e. Compute critic losses
        critic_losses = []

        for critic_idx, critic_ensemble in enumerate(current_q_values):
            ensemble_losses = []

            for obj_idx, obj_q_values in enumerate(critic_ensemble):
                # MSE loss between current Q-values and targets
                obj_loss = mse_loss(obj_q_values, td_targets[obj_idx])
                ensemble_losses.append(obj_loss)

            # Combine losses across objectives using preference weights
            avg_weights = preference_weights.mean(dim=0)
            weighted_loss = sum(loss * weight for loss, weight in zip(ensemble_losses, avg_weights))
            critic_losses.append(weighted_loss)

        # 1f. Total critic loss and backward
        critic_loss = sum(critic_losses)
        critic_loss.backward()

        # 1g. Critic optimizer step
        critic_optimizer.step()

        # 2. Actor update
        # 2a. Zero gradients
        actor_optimizer.zero_grad()

        # 2b. Sample actions from the current policy
        actions_pi, log_probs = policy.actor.action_log_prob(obs)

        # 2c. Get Q-values from critic
        q_values_pi = policy.critic(obs, actions_pi)

        # 2d. Manually calculate scalarized Q-values
        scalarized_q_values = []

        for critic_ensemble in q_values_pi:
            # Stack objective values: (batch_size, num_objectives)
            stacked_values = th.cat([obj_val for obj_val in critic_ensemble], dim=1)

            # Compute weighted sum: (batch_size, 1)
            scalarized = th.sum(stacked_values * preference_weights, dim=1, keepdim=True)
            scalarized_q_values.append(scalarized)

        # 2e. Compute the minimum Q-value across critic ensembles
        min_qf_pi = th.cat(scalarized_q_values, dim=1).min(dim=1, keepdim=True)[0]

        # 2f. Compute actor loss
        actor_loss = (ent_coef * log_probs - min_qf_pi).mean()

        # 2g. Actor backward and optimizer step
        actor_loss.backward()
        actor_optimizer.step()

        # 3. Update target networks (typically done with soft update)
        # For testing, we'll just verify parameters were updated

        # Check that parameters were updated
        actor_params_changed = False
        for name, param in policy.actor.named_parameters():
            if not th.allclose(param, initial_actor_params[name]):
                actor_params_changed = True
                break

        critic_params_changed = False
        for name, param in policy.critic.named_parameters():
            if not th.allclose(param, initial_critic_params[name]):
                critic_params_changed = True
                break

        # Assert parameters were updated
        assert actor_params_changed, "Actor parameters were not updated"
        assert critic_params_changed, "Critic parameters were not updated"

        # Print debug info
        print(
            f"Full training step complete. Actor loss: {actor_loss.item():.6f}, Critic loss: {critic_loss.item():.6f}")