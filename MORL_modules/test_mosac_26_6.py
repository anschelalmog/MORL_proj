import os
import pytest
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

# Mock the imports for your custom modules if needed
# Uncomment and modify paths as needed
# Import your modules
from agents.monets import SharedFeatureQNet, SeparateQNet
from agents.mobuffers import MOReplayBuffer
from stable_baselines3.common.torch_layers import FlattenExtractor
from agents.mosac import MOSAC, MOSACPolicy, MOContinuousCritic
from stable_baselines3.sac.policies import Actor
# Import your implementation
from agents.monets import SharedFeatureQNet, SeparateQNet
from agents.mobuffers import MOReplayBuffer
from agents.mosac import MOSAC, MOSACPolicy, MOContinuousCritic

from agents.mo_env_wrappers  import MODummyVecEnv, MultiObjectiveWrapper
# Create a simple multi-objective environment for testing
class SimpleMOEnv(gym.Env):
    """
    A simple multi-objective environment with continuous actions.
    """

    def __init__(self, num_objectives=3):
        super().__init__()
        self.observation_space = spaces.Box(low=-10, high=10, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.num_objectives = num_objectives
        self.state = None
        self.max_steps = 100
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.observation_space.sample()
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        self.state = self.state + 0.1 * action.flatten()
        self.state = np.clip(self.state, -10, 10)

        # Calculate multi-objective rewards
        rewards = np.zeros(self.num_objectives)
        # First objective rewards being close to origin
        rewards[0] = -np.sum(np.square(self.state)) * 0.01
        # Second objective rewards high action values
        rewards[1] = np.sum(np.abs(action)) * 0.1
        # Third objective rewards consistency in state values
        if self.num_objectives > 2:
            rewards[2] = -np.std(self.state) * 0.05

        done = self.current_step >= self.max_steps
        truncated = False
        info = {}

        return self.state, rewards, done, truncated, info


# Create a wrapper to make it compatible with the MODummyVecEnv
class MOEnvWrapper(MultiObjectiveWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_objectives = env.num_objectives

    def step(self, action):
        obs, rewards, done, truncated, info = self.env.step(action)
        # Ensure rewards is returned as a vector
        return obs, rewards, done, truncated, info


# Helper function to create a wrapped environment
def make_mo_env(num_objectives=3):
    env = SimpleMOEnv(num_objectives=num_objectives)
    env = MOEnvWrapper(env)
    return env


# Test fixture for creating a MOSAC agent
@pytest.fixture
def mosac_agent():
    # Set seeds for reproducibility
    set_random_seed(0)

    # Create environment
    num_objectives = 3
    env = make_mo_env(num_objectives=num_objectives)
    env = MODummyVecEnv([lambda: env])

    # Create agent with smaller network for faster testing
    policy_kwargs = {
        "net_arch": {
            "pi": [64, 64],
            "qf": [64, 64]
        },
        "share_features_across_objectives": True
    }

    agent = MOSAC(
        env=env,
        policy="MOSACPolicy",
        learning_rate=3e-4,
        buffer_size=1000,  # Small buffer for testing
        learning_starts=10,
        batch_size=4,
        tau=0.01,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        num_objectives=num_objectives,
        preference_weights=[0.4, 0.3, 0.3],
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=0
    )

    return agent


# Test fixture for creating a filled replay buffer
@pytest.fixture
def filled_buffer(mosac_agent):
    # Fill the buffer with some transitions
    env = mosac_agent.env
    obs = env.reset()

    # Fill the buffer with some initial transitions
    for _ in range(20):
        action = np.array([env.action_space.sample() for _ in range(env.num_envs)])
        next_obs, rewards, dones,  infos = env.step(action)

        # Add the transition to the buffer
        mosac_agent.replay_buffer.add(
            obs, next_obs, action, rewards, dones, infos
        )

        obs = next_obs
        if dones.any():
            obs = env.reset()

    return mosac_agent.replay_buffer


def test_mosac_initialization(mosac_agent):
    """Test that MOSAC agent is initialized correctly."""
    # Check that the agent has the correct attributes
    assert isinstance(mosac_agent.policy, MOSACPolicy)
    assert isinstance(mosac_agent.critic, MOContinuousCritic)
    assert isinstance(mosac_agent.critic_target, MOContinuousCritic)
    assert isinstance(mosac_agent.replay_buffer, MOReplayBuffer)

    # Check that the preference weights are set correctly
    assert len(mosac_agent.preference_weights) == 3
    assert np.isclose(mosac_agent.preference_weights.sum(), 1.0)

    # Check that the critic has the correct number of outputs
    q_values = mosac_agent.critic_target.forward(
        th.zeros((1, 4), device=mosac_agent.device),
        th.zeros((1, 2), device=mosac_agent.device)
    )

    # Should return a tuple of critic networks (n_critics)
    assert len(q_values) == 2

    # Each critic should output a list of values (one per objective)
    assert len(q_values[0]) == 3


def test_one_training_step(mosac_agent, filled_buffer):
    """Test that one training step updates the model parameters."""
    # Get initial parameters
    initial_actor_params = [param.clone().detach() for param in mosac_agent.actor.parameters()]
    initial_critic_params = [param.clone().detach() for param in mosac_agent.critic.parameters()]

    mosac_agent._setup_learn(total_timesteps= 1)
    # Perform one training step
    mosac_agent.train(gradient_steps=1, batch_size=4)

    # Check that parameters have been updated
    for i, param in enumerate(mosac_agent.actor.parameters()):
        assert not th.allclose(initial_actor_params[i], param), "Actor parameters should change after training"

    for i, param in enumerate(mosac_agent.critic.parameters()):
        assert not th.allclose(initial_critic_params[i], param), "Critic parameters should change after training"


def test_actor_loss(mosac_agent, filled_buffer):
    """Test that actor loss is calculated correctly and decreases during training."""
    # Enable gradients tracking for loss calculation
    initial_losses = []
    final_losses = []

    # Track actor losses during training
    for _ in range(3):  # Train for 3 iterations
        # Sample batch for consistent comparison
        replay_data = mosac_agent.replay_buffer.sample(mosac_agent.batch_size, None)

        # Get current actions and log probs
        actions_pi, log_prob = mosac_agent.actor.action_log_prob(replay_data.observations)
        log_prob = log_prob.reshape(-1, 1)

        # Get current Q-values
        q_values_pi = th.cat(mosac_agent.critic.q_value(
            replay_data.observations,
            actions_pi,
            mosac_agent.preference_weights_tensor
        ), dim=1)

        min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)

        # Calculate actor loss
        if isinstance(mosac_agent.ent_coef, th.Tensor):
            ent_coef = th.exp(mosac_agent.log_ent_coef.detach())
        else:
            ent_coef = th.tensor(mosac_agent.ent_coef).to(mosac_agent.device)

        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        initial_losses.append(actor_loss.item())

        # Perform one training step
        mosac_agent.train(gradient_steps=1, batch_size=mosac_agent.batch_size)

    # Calculate losses after training
    for _ in range(3):
        replay_data = mosac_agent.replay_buffer.sample(mosac_agent.batch_size, None)

        actions_pi, log_prob = mosac_agent.actor.action_log_prob(replay_data.observations)
        log_prob = log_prob.reshape(-1, 1)

        q_values_pi = th.cat(mosac_agent.critic.q_value(
            replay_data.observations,
            actions_pi,
            mosac_agent.preference_weights_tensor
        ), dim=1)

        min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)

        if isinstance(mosac_agent.ent_coef, th.Tensor):
            ent_coef = th.exp(mosac_agent.log_ent_coef.detach())
        else:
            ent_coef = th.tensor(mosac_agent.ent_coef).to(mosac_agent.device)

        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        final_losses.append(actor_loss.item())

    # Check that loss values are reasonable
    assert all(not np.isnan(loss) for loss in initial_losses), "Initial actor losses should not be NaN"
    assert all(not np.isnan(loss) for loss in final_losses), "Final actor losses should not be NaN"

    # The loss should generally decrease with training
    # We use the average because individual batches might fluctuate
    assert np.mean(final_losses) <= np.mean(initial_losses), "Actor loss should decrease on average during training"


def test_critic_loss(mosac_agent, filled_buffer):
    """Test that critic loss is calculated correctly and decreases during training."""
    # Enable gradients tracking for loss calculation
    initial_losses = []
    final_losses = []

    # Track critic losses during training
    for _ in range(3):  # Train for 3 iterations
        # Sample batch for consistent comparison
        mosac_agent._setup_learn(total_timesteps= 1)
        replay_data = mosac_agent.replay_buffer.sample(mosac_agent.batch_size, mosac_agent._vec_normalize_env)

        # Get current Q-values
        current_q_values = mosac_agent.critic(replay_data.observations, replay_data.actions)

        # Get next Q-values
        with th.no_grad():
            next_actions, next_log_prob = mosac_agent.actor.action_log_prob(replay_data.next_observations)
            next_q_values = mosac_agent.critic_target(replay_data.next_observations, next_actions)

            # Compute target Q-values for each objective
            target_q_values = []
            for obj_idx in range(mosac_agent.num_objectives):
                # Get Q-values for this objective from all critic ensembles
                obj_next_q_values = th.cat([
                    ensemble[obj_idx] for ensemble in next_q_values
                ], dim=1)

                # Compute target Q-value for this objective
                obj_next_q_values = obj_next_q_values.min(dim=1, keepdim=True)[0]

                if isinstance(mosac_agent.ent_coef, th.Tensor):
                    ent_coef = th.exp(mosac_agent.log_ent_coef.detach())
                else:
                    ent_coef = th.tensor(mosac_agent.ent_coef).to(mosac_agent.device)

                obj_next_q_values = obj_next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                obj_reward = replay_data.rewards[:, obj_idx].reshape(-1, 1)
                obj_target = obj_reward + (1 - replay_data.dones) * mosac_agent.gamma * obj_next_q_values
                target_q_values.append(obj_target)

        # Calculate critic loss
        critic_loss = 0.5 * sum(
            sum(
                F.mse_loss(obj_current_q, target_q_values[obj_idx]) * mosac_agent.preference_weights[obj_idx]
                for obj_idx, obj_current_q in enumerate(current_q_ensemble)
            )
            for current_q_ensemble in current_q_values
        )

        initial_losses.append(critic_loss.item())

        # Perform one training step
        mosac_agent.train(gradient_steps=1, batch_size=mosac_agent.batch_size)

    # Calculate losses after training
    for _ in range(3):
        replay_data = mosac_agent.replay_buffer.sample(mosac_agent.batch_size, None)

        current_q_values = mosac_agent.critic(replay_data.observations, replay_data.actions)

        with th.no_grad():
            next_actions, next_log_prob = mosac_agent.actor.action_log_prob(replay_data.next_observations)
            next_q_values = mosac_agent.critic_target(replay_data.next_observations, next_actions)

            target_q_values = []
            for obj_idx in range(mosac_agent.num_objectives):
                obj_next_q_values = th.cat([
                    ensemble[obj_idx] for ensemble in next_q_values
                ], dim=1)

                obj_next_q_values = obj_next_q_values.min(dim=1, keepdim=True)[0]

                if isinstance(mosac_agent.ent_coef, th.Tensor):
                    ent_coef = th.exp(mosac_agent.log_ent_coef.detach())
                else:
                    ent_coef = th.tensor(mosac_agent.ent_coef).to(mosac_agent.device)

                obj_next_q_values = obj_next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                obj_reward = replay_data.rewards[:, obj_idx].reshape(-1, 1)
                obj_target = obj_reward + (1 - replay_data.dones) * mosac_agent.gamma * obj_next_q_values
                target_q_values.append(obj_target)

        critic_loss = 0.5 * sum(
            sum(
                F.mse_loss(obj_current_q, target_q_values[obj_idx]) * mosac_agent.preference_weights[obj_idx]
                for obj_idx, obj_current_q in enumerate(current_q_ensemble)
            )
            for current_q_ensemble in current_q_values
        )

        final_losses.append(critic_loss.item())

    # Check that loss values are reasonable
    assert all(not np.isnan(loss) for loss in initial_losses), "Initial critic losses should not be NaN"
    assert all(not np.isnan(loss) for loss in final_losses), "Final critic losses should not be NaN"

    # The loss should generally decrease with training
    assert np.mean(final_losses) <= np.mean(initial_losses), "Critic loss should decrease on average during training"


def test_replay_buffer_storage(mosac_agent):
    """Test that data is correctly stored in the replay buffer during training."""
    # Reset the environment and buffer
    env = mosac_agent.env
    mosac_agent.replay_buffer = MOReplayBuffer(
        buffer_size=100,
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_objectives=mosac_agent.num_objectives,
        device=mosac_agent.device,
    )

    # Track what should be in the buffer
    observations = []
    actions = []
    rewards = []
    dones = []

    # Collect some data
    obs = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        observations.append(obs.copy())
        actions.append(action.copy())

        next_obs, reward, done,  info = env.step(action)
        rewards.append(np.array(reward,    np.float32).copy())
        dones.append(np.array(done,    np.float32).copy())
        # Add to buffer (happens automatically in learn())
        mosac_agent.replay_buffer.add(obs, next_obs, action, reward, done, info)

        obs = next_obs
        if done.any():
            obs = env.reset()

    # Check buffer contents
    assert mosac_agent.replay_buffer.pos == min(10, mosac_agent.replay_buffer.buffer_size)
    assert not mosac_agent.replay_buffer.full

    # Check that rewards are stored as vectors
    assert mosac_agent.replay_buffer.rewards.shape[2] == mosac_agent.num_objectives

    # Check that the stored rewards match what we expect
    for i in range(len(rewards)):
        assert np.allclose(
            mosac_agent.replay_buffer.rewards[i, 0],
            rewards[i][0]
        ), f"Rewards at position {i} don't match expected values"


def test_complete_learning_loop(mosac_agent):
    """Test a complete learning loop with MOSAC."""
    # Setup the learning parameters
    total_timesteps = 50

    # Use the built-in learn method that sets up everything properly
    mosac_agent.learn(
        total_timesteps=total_timesteps,
        log_interval=None,
    )

    # Check that the replay buffer has been filled
    assert mosac_agent.replay_buffer.pos > 0 or mosac_agent.replay_buffer.full

    # Check that the agent can predict actions
    env = mosac_agent.env
    obs = env.reset()

    action, _ = mosac_agent.predict(obs, deterministic=True)

    # Action should be within the action space bounds
    assert env.action_space.contains(action[0]), "Predicted action should be within action space bounds"

    # Check that the agent's learned Q-values make sense
    obs_tensor = th.FloatTensor(obs).to(mosac_agent.device)
    action_tensor = th.FloatTensor(action).to(mosac_agent.device)

    q_values = mosac_agent.critic.forward(obs_tensor, action_tensor)

    # Each critic ensemble should output values for each objective
    assert len(q_values) == 2, "Should have 2 critic ensembles"
    assert len(q_values[0]) == mosac_agent.num_objectives, "Each critic should output values for each objective"

    # Q-values should be finite (not NaN or inf)
    for critic_ensemble in q_values:
        for obj_values in critic_ensemble:
            assert th.all(th.isfinite(obj_values)), "Q-values should be finite"


def test_preference_weight_scalarization(mosac_agent, filled_buffer):
    """Test that preference weights correctly scalarize the Q-values."""
    # Sample a batch from the buffer
    replay_data = mosac_agent.replay_buffer.sample(mosac_agent.batch_size, None)

    # Get Q-values
    current_q_values = mosac_agent.critic(replay_data.observations, replay_data.actions)

    # Try different preference weights
    weights1 = th.FloatTensor([1.0, 0.0, 0.0]).to(mosac_agent.device)  # Only first objective
    weights2 = th.FloatTensor([0.0, 1.0, 0.0]).to(mosac_agent.device)  # Only second objective

    # Get scalarized Q-values
    scalarized1 = mosac_agent.critic.q_value(replay_data.observations, replay_data.actions, weights1)
    scalarized2 = mosac_agent.critic.q_value(replay_data.observations, replay_data.actions, weights2)

    # The scalarized values should be different for different weights
    # (unless the Q-values for both objectives are identical, which is unlikely)
    for i in range(len(scalarized1)):
        assert not th.allclose(scalarized1[i], scalarized2[i]), \
            "Scalarized Q-values should differ for different preference weights"

    # Check that scalarization is working correctly
    # For weights [1,0,0], the scalarized value should match the first objective's Q-value
    for i, critic_values in enumerate(current_q_values):
        expected = critic_values[0]  # First objective's Q-value
        actual = scalarized1[i]
        assert th.allclose(expected, actual), \
            f"Scalarized Q-value with weights [1,0,0] should match first objective's Q-value for critic {i}"