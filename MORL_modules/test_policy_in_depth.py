import pytest
import numpy as np
import torch as th
from gymnasium import spaces
from torch.distributions import Normal


from agents.mosac import MOContinuousCritic, MOSACPolicy
from agents.monets import SharedFeatureQNet, SeparateQNet
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution

class TestMOSACPolicyDetailed:
    """Advanced tests for MOSACPolicy focusing on forward pass and Q-value calculations."""

    @pytest.fixture
    def setup_environment(self):
        """Create a testing environment with fixed parameters."""
        observation_space = spaces.Box(low=-10, high=10, shape=(5,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Create policy with 3 objectives
        policy = MOSACPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda _: 3e-4,
            num_objectives=3,
            net_arch=[64, 64],
            n_critics=2,
            share_features_across_objectives=True
        )

        # Generate test data
        batch_size = 20
        obs = th.FloatTensor(np.random.uniform(-1, 1, size=(batch_size, 5)))
        actions = th.FloatTensor(np.random.uniform(-1, 1, size=(batch_size, 2)))

        return {
            "policy": policy,
            "obs": obs,
            "actions": actions,
            "batch_size": batch_size,
            "action_dim": 2,
            "obs_dim": 5,
            "num_objectives": 3
        }

    def test_actor_distribution_parameters(self, setup_environment):
        """Test the actor network outputs the correct distribution parameters."""
        policy = setup_environment["policy"]
        obs = setup_environment["obs"]
        batch_size = setup_environment["batch_size"]
        action_dim = setup_environment["action_dim"]

        # Get distribution parameters
        mean_actions, log_std, _ = policy.actor.get_action_dist_params(obs)

        # Check shapes
        assert mean_actions.shape == (batch_size, action_dim)
        assert log_std.shape == (batch_size, action_dim)

        # Check log_std is within the expected clipping range
        # SAC typically clips log_std to prevent extreme values
        min_log_std = -20  # Typical lower bound
        max_log_std = 2  # Typical upper bound
        assert th.all(log_std >= min_log_std) and th.all(log_std <= max_log_std)

        # Check means are within action space bounds before squashing
        # (values might exceed [-1,1] because tanh squashing happens later)
        assert th.isfinite(mean_actions).all(), "Mean actions contain NaN or inf values"

    def test_action_sampling_and_log_prob(self, setup_environment):
        """Test action sampling and log probability calculation matching SquashedDiagGaussianDistribution."""
        policy = setup_environment["policy"]
        obs = setup_environment["obs"]
        batch_size = setup_environment["batch_size"]
        action_dim = setup_environment["action_dim"]

        # Get distribution parameters
        mean_actions, log_std, _ = policy.actor.get_action_dist_params(obs)

        # Create normal distribution
        std = th.exp(log_std)
        normal_dist = Normal(mean_actions, std)

        # Sample raw actions (using reparameterization trick)
        noise = th.randn_like(mean_actions)
        raw_actions = mean_actions + noise * std

        # Apply squashing to get bounded actions
        actions = th.tanh(raw_actions)

        # Calculate log probabilities for each dimension
        log_prob_per_dim = normal_dist.log_prob(raw_actions)

        # Sum across action dimensions to get joint log probability (assuming independent dims)
        log_prob = log_prob_per_dim.sum(dim=1)

        # Apply squashing correction
        # log_prob -= sum(log(1 - tanh(raw_actions)^2))
        log_prob -= th.sum(th.log(1 - actions.pow(2) + 1e-6), dim=1)

        # Check shapes
        assert actions.shape == (batch_size, action_dim)
        assert log_prob.shape == (batch_size,)

        # Check actions are within bounds after squashing
        assert th.all(actions >= -1) and th.all(actions <= 1)

        # Now use the policy's own method to get actions and log probs
        policy_actions, policy_log_probs = policy.actor.action_log_prob(obs)

        # Check shapes match
        assert policy_actions.shape == (batch_size, action_dim)
        assert policy_log_probs.shape == (batch_size,)

        # Check actions are within bounds
        assert th.all(policy_actions >= -1) and th.all(policy_actions <= 1)

        # Check log probs are finite
        assert th.isfinite(policy_log_probs).all(), "Log probs contain NaN or inf values"

        # Mock a separate calculation to check if the policy's output matches the expected behavior
        # This simulates what SquashedDiagGaussianDistribution should return
        with th.no_grad():
            # Get a new set of distribution parameters
            mean_actions2, log_std2, _ = policy.actor.get_action_dist_params(obs)

            # Sample actions deterministically for consistent comparison
            actions_deterministic = th.tanh(mean_actions2)

            # For deterministic actions, the log_prob can be calculated as follows:
            gaussian_log_probs = Normal(mean_actions2, th.exp(log_std2)).log_prob(mean_actions2)
            deterministic_log_probs = gaussian_log_probs.sum(dim=1)
            # Apply squashing correction
            deterministic_log_probs -= th.sum(th.log(1 - actions_deterministic.pow(2) + 1e-6), dim=1)
            # Get the policy's deterministic actions and log probs
            policy_det_actions =  policy.actor.forward(obs, deterministic=True)
            policy_det_log_probs =  policy.actor.action_dist.log_prob(policy_det_actions)
            # Check shapes
            assert policy_det_actions.shape == (batch_size, action_dim)
            assert policy_det_log_probs.shape == (batch_size,)

            # Check for consistency in how the calculations are performed
            # Note: We don't expect exact equality since the policy might have its own implementation details
            # but the shapes and ranges should be similar
            assert policy_det_log_probs.shape == deterministic_log_probs.shape
            assert th.equal(policy_det_log_probs, deterministic_log_probs)

    def test_action_deterministic_vs_stochastic(self, setup_environment):
        """Compare deterministic and stochastic action outputs."""
        policy = setup_environment["policy"]
        obs = setup_environment["obs"]

        # Get deterministic actions
        with th.no_grad():
            deterministic_actions = policy.actor.forward(obs, deterministic=True)

        # Get multiple stochastic actions for the same observations
        stochastic_actions_list = []
        for _ in range(10):
            with th.no_grad():
                stochastic_actions = policy.actor.forward(obs, deterministic=False)
                stochastic_actions_list.append(stochastic_actions)

        # Stack stochastic actions for easier analysis
        stochastic_actions_stack = th.stack(stochastic_actions_list, dim=0)

        # Calculate standard deviation across stochastic samples
        stochastic_std = th.std(stochastic_actions_stack, dim=0)

        # Deterministic actions should be within the action space bounds
        assert th.all(deterministic_actions >= -1) and th.all(deterministic_actions <= 1)

        # Stochastic actions should vary (have non-zero standard deviation)
        assert th.mean(stochastic_std) > 0.01, "Stochastic actions don't show enough variation"

        # Mean of stochastic actions should be close to deterministic actions
        # (with some tolerance due to sampling)
        stochastic_mean = th.mean(stochastic_actions_stack, dim=0)
        mean_diff = th.abs(stochastic_mean - deterministic_actions)
        assert th.mean(mean_diff) < 0.5, "Mean of stochastic actions deviates too much from deterministic actions"

    def test_critic_q_value_basic(self, setup_environment):
        """Test basic Q-value calculation with uniform weights."""
        policy = setup_environment["policy"]
        obs = setup_environment["obs"]
        actions = setup_environment["actions"]
        batch_size = setup_environment["batch_size"]
        num_objectives = setup_environment["num_objectives"]

        # Get critic values directly
        critic_values = policy.critic(obs, actions)

        # Check structure
        assert len(critic_values) == policy.critic.n_critics
        for critic_ensemble in critic_values:
            assert len(critic_ensemble) == num_objectives
            for obj_value in critic_ensemble:
                assert obj_value.shape == (batch_size, 1)
                assert th.isfinite(obj_value).all(), "Q-values contain NaN or inf values"

        # Calculate Q-values with uniform weights
        uniform_weights = th.ones(num_objectives) / num_objectives
        q_values = policy.critic.q_value(obs, actions, uniform_weights)

        # Check q_values structure
        assert len(q_values) == policy.critic.n_critics
        for q_val in q_values:
            assert q_val.shape == (batch_size, 1)
            assert th.isfinite(q_val).all(), "Scalarized Q-values contain NaN or inf values"

        # Manually calculate weighted sum and compare
        for i, critic_ensemble in enumerate(critic_values):
            # Stack objective values [batch_size, num_objectives]
            stacked_values = th.cat([obj_val for obj_val in critic_ensemble], dim=1)
            # Calculate weighted sum
            manual_q_value = th.sum(stacked_values * uniform_weights, dim=1, keepdim=True)
            # Compare with the q_value method output
            assert th.allclose(manual_q_value, q_values[i], rtol=1e-5)

    def test_critic_q_value_with_different_weights(self, setup_environment):
        """Test Q-value calculation with different preference weight distributions."""
        policy = setup_environment["policy"]
        obs = setup_environment["obs"]
        actions = setup_environment["actions"]
        batch_size = setup_environment["batch_size"]
        num_objectives = setup_environment["num_objectives"]

        # Get critic values directly
        critic_values = policy.critic(obs, actions)

        # Test cases with different weight distributions
        weight_test_cases = [
            # Uniform weights
            th.ones(num_objectives) / num_objectives,

            # Single objective prioritized
            th.tensor([0.8, 0.1, 0.1]),

            # Two objectives equally prioritized
            th.tensor([0.45, 0.45, 0.1]),

            # Extreme case - only one objective matters
            th.tensor([1.0, 0.0, 0.0]),

            # Batch of weights (different for each observation)
            th.FloatTensor(np.random.uniform(0, 1, size=(batch_size, num_objectives)))
        ]

        # For the batch case, normalize the weights
        weight_test_cases[-1] = weight_test_cases[-1] / weight_test_cases[-1].sum(dim=1, keepdim=True)

        # Test each weight distribution
        for weights in weight_test_cases:
            # Get scalarized Q-values
            q_values = policy.critic.q_value(obs, actions, weights)

            # Check basic properties
            assert len(q_values) == policy.critic.n_critics
            for q_val in q_values:
                assert q_val.shape == (batch_size, 1)
                assert th.isfinite(q_val).all()

            # Manually calculate weighted sum for verification
            for i, critic_ensemble in enumerate(critic_values):
                # Stack objective values [batch_size, num_objectives]
                stacked_values = th.cat([obj_val for obj_val in critic_ensemble], dim=1)

                # Calculate weighted sum based on weight shape
                if weights.dim() == 1:
                    # Expand weights to match batch size
                    expanded_weights = weights.unsqueeze(0).expand(batch_size, -1)
                    manual_q_value = th.sum(stacked_values * expanded_weights, dim=1, keepdim=True)
                else:
                    # Batch-specific weights
                    manual_q_value = th.sum(stacked_values * weights, dim=1, keepdim=True)

                # Compare with the q_value method output
                assert th.allclose(manual_q_value, q_values[i], rtol=1e-5)

    def test_extreme_preference_weights(self, setup_environment):
        """Test Q-value calculation with extreme preference weights."""
        policy = setup_environment["policy"]
        obs = setup_environment["obs"]
        actions = setup_environment["actions"]
        num_objectives = setup_environment["num_objectives"]

        # Get raw critic values
        critic_values = policy.critic(obs, actions)

        # Test with one-hot weights (focus on each objective individually)
        for obj_idx in range(num_objectives):
            # Create one-hot weight vector
            weights = th.zeros(num_objectives)
            weights[obj_idx] = 1.0

            # Get scalarized Q-values
            q_values = policy.critic.q_value(obs, actions, weights)

            # For each critic ensemble, verify the scalarized value
            # equals the value of the selected objective
            for i, critic_ensemble in enumerate(critic_values):
                # Extract the value for the selected objective
                obj_value = critic_ensemble[obj_idx]

                # Compare with the scalarized value
                assert th.allclose(obj_value, q_values[i], rtol=1e-5)

    def test_critic_ensembles_variation(self, setup_environment):
        """Test variation between different critic ensembles."""
        policy = setup_environment["policy"]
        obs = setup_environment["obs"]
        actions = setup_environment["actions"]
        num_objectives = setup_environment["num_objectives"]

        # Get critic values directly
        critic_values = policy.critic(obs, actions)

        # Check there are multiple critic ensembles that produce different outputs
        if len(critic_values) > 1:
            # Compare first two ensembles
            first_ensemble = critic_values[0]
            second_ensemble = critic_values[1]

            # Check if there's variation in at least some of the objective values
            has_variation = False
            for obj_idx in range(num_objectives):
                diff = th.abs(first_ensemble[obj_idx] - second_ensemble[obj_idx])
                if th.mean(diff) > 1e-6:
                    has_variation = True
                    break

            assert has_variation, "Critic ensembles produce identical values, suggesting they might be sharing parameters incorrectly"

        # Test with uniform weights
        uniform_weights = th.ones(num_objectives) / num_objectives
        q_values = policy.critic.q_value(obs, actions, uniform_weights)

        # Check variation in scalarized values
        if len(q_values) > 1:
            diff = th.abs(q_values[0] - q_values[1])
            avg_diff = th.mean(diff).item()

            # Log the average difference for debugging
            print(f"Average difference between critic ensembles: {avg_diff}")

            # Different critic ensembles should produce somewhat different outputs
            assert avg_diff > 1e-6, "Critic ensembles produce identical scalarized values"

    def test_preference_weight_boundary_conditions(self, setup_environment):
        """Test Q-value calculation with boundary condition preference weights."""
        policy = setup_environment["policy"]
        obs = setup_environment["obs"]
        actions = setup_environment["actions"]
        batch_size = setup_environment["batch_size"]
        num_objectives = setup_environment["num_objectives"]

        # Test with small weights that sum to 1
        small_weights = th.ones(num_objectives) * (1.0 / num_objectives * 0.001)
        small_weights[0] = 1.0 - small_weights[1:].sum()  # Ensure they sum to 1

        q_values = policy.critic.q_value(obs, actions, small_weights)

        # Check q_values are finite
        for q_val in q_values:
            assert th.isfinite(q_val).all(), "Q-values with small weights contain NaN or inf values"

        # Test with batch of very skewed weights
        skewed_batch_weights = th.zeros(batch_size, num_objectives)
        # For each sample in batch, have a different objective with weight near 1
        for i in range(batch_size):
            obj_idx = i % num_objectives
            skewed_batch_weights[i, obj_idx] = 0.999
            # Distribute remaining weight to other objectives
            remaining_weight = 0.001 / (num_objectives - 1)
            for j in range(num_objectives):
                if j != obj_idx:
                    skewed_batch_weights[i, j] = remaining_weight

        q_values = policy.critic.q_value(obs, actions, skewed_batch_weights)

        # Check q_values are finite
        for q_val in q_values:
            assert th.isfinite(q_val).all(), "Q-values with skewed batch weights contain NaN or inf values"

    def test_forward_pass_gradient_flow(self, setup_environment):
        """Test gradient flow through the actor during forward pass."""
        policy = setup_environment["policy"]
        obs = setup_environment["obs"]

        # Ensure gradients are being tracked
        obs.requires_grad_(True)

        # Forward pass through actor
        mean_actions, log_std, _ = policy.actor.get_action_dist_params(obs)
        # Check if mean_actions and log_std require gradients
        assert mean_actions.requires_grad, "Mean actions don't require gradients"
        assert log_std.requires_grad, "Log std doesn't require gradients"

        # Create a simple loss based on the mean actions
        loss = mean_actions.mean()

        # Backpropagate
        loss.backward()

        # Check if gradients are computed for observation
        assert obs.grad is not None, "No gradients computed for observations"
        assert not th.all(obs.grad == 0), "All gradients for observations are zero"

        # Check if actor parameters received gradients
        all_params_have_grad = True
        for param in policy.actor.parameters():
            if param.grad is None or th.all(param.grad == 0):
                breakpoint()
                all_params_have_grad = False
                break

        assert all_params_have_grad, "Some actor parameters didn't receive gradients"

    def test_q_value_gradient_flow(self, setup_environment):
        """Test gradient flow through the critic during Q-value calculation."""
        policy = setup_environment["policy"]
        obs = setup_environment["obs"]
        actions = setup_environment["actions"]
        num_objectives = setup_environment["num_objectives"]

        # Ensure gradients are being tracked
        obs.requires_grad_(True)
        actions.requires_grad_(True)

        # Uniform weights
        weights = th.ones(num_objectives) / num_objectives

        # Forward pass through critic
        q_values = policy.critic.q_value(obs, actions, weights)
        # Create a simple loss based on the q_values
        loss = sum(q_val.mean() for q_val in q_values)

        # Backpropagate
        loss.backward()

        # Check if gradients are computed for inputs
        assert obs.grad is not None, "No gradients computed for observations"
        assert actions.grad is not None, "No gradients computed for actions"
        assert not th.all(obs.grad == 0), "All gradients for observations are zero"
        assert not th.all(actions.grad == 0), "All gradients for actions are zero"

        # Check if critic parameters received gradients
        all_params_have_grad = True
        for param in policy.critic.parameters():
            if param.grad is None or th.all(param.grad == 0):
                all_params_have_grad = False
                break

        assert all_params_have_grad, "Some critic parameters didn't receive gradients"

    def test_critic_gradient_flow(self, setup_environment):
        """Test gradient flow through the critic during Q-value calculation."""
        policy = setup_environment["policy"]
        obs = setup_environment["obs"]
        actions = setup_environment["actions"]
        num_objectives = setup_environment["num_objectives"]

        # Ensure gradients are being tracked
        obs.requires_grad_(True)
        actions.requires_grad_(True)

        # Uniform weights
        with th.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = policy.actor.action_log_prob(obs)
            # Compute the next Q values: min over all critics targets
            next_q_values = th.cat(policy.critic(obs, actions), dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            # add entropy term
            next_q_values = next_q_values - 0.1 * next_log_prob.reshape(-1, 1)
            # td error + entropy term
        # Forward pass through critic
        q_values = policy.critic(obs, actions)
        # Create a simple loss based on the q_values
        loss = sum(q_val.mean() for q_val in q_values)

        # Backpropagate
        loss.backward()

        # Check if gradients are computed for inputs
        assert obs.grad is not None, "No gradients computed for observations"
        assert actions.grad is not None, "No gradients computed for actions"
        assert not th.all(obs.grad == 0), "All gradients for observations are zero"
        assert not th.all(actions.grad == 0), "All gradients for actions are zero"

        # Check if critic parameters received gradients
        all_params_have_grad = True
        for param in policy.critic.parameters():
            if param.grad is None or th.all(param.grad == 0):
                all_params_have_grad = False
                break

        assert all_params_have_grad, "Some critic parameters didn't receive gradients"

    def test_numerical_stability(self, setup_environment):
        """Test numerical stability with extreme observation values."""
        policy = setup_environment["policy"]
        batch_size = setup_environment["batch_size"]
        action_dim = setup_environment["action_dim"]
        obs_dim = setup_environment["obs_dim"]
        num_objectives = setup_environment["num_objectives"]

        # Create extreme observations (very large values)
        extreme_obs = th.ones(batch_size, obs_dim) * 1e6

        # Test actor forward pass with extreme values
        with th.no_grad():
            try:
                actions = policy.actor.forward(extreme_obs, deterministic=True)
                # Check actions are within bounds
                assert th.all(actions >= -1) and th.all(actions <= 1)
                assert th.isfinite(actions).all(), "Actions contain NaN or inf values with extreme observations"
            except Exception as e:
                assert False, f"Actor forward pass with extreme observations failed: {str(e)}"

        # Create some valid actions
        actions = th.zeros(batch_size, action_dim)

        # Uniform weights
        weights = th.ones(num_objectives) / num_objectives

        # Test critic forward pass with extreme values
        with th.no_grad():
            try:
                q_values = policy.critic.q_value(extreme_obs, actions, weights)
                # Check q_values are finite
                for q_val in q_values:
                    assert th.isfinite(q_val).all(), "Q-values contain NaN or inf values with extreme observations"
            except Exception as e:
                assert False, f"Critic forward pass with extreme observations failed: {str(e)}"