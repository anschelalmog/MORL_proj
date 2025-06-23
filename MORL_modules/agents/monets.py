import torch as th
import torch.nn as nn

class SharedFeatureQNet(nn.Module):
    """
    A neural network module with a shared feature extractor and multiple output heads
    for multi-objective Q-value estimation.

    This class is used in multi-objective reinforcement learning critic networks where:
    - There are multiple objectives (each requiring a separate Q-value output).
    - The initial (trunk) layers of the network are shared across all objectives, allowing shared feature learning.
    - Each objective has its own output head (typically a linear layer) that produces the Q-value for that objective.

    Args:
        base_net (nn.Module): The shared base network, typically an MLP that processes
                              the concatenated observation and action.
        heads (nn.ModuleList): A list of output heads (one per objective), each producing
                               a single Q-value.

    Forward Inputs:
        obs (Tensor): Observation tensor of shape (batch_size, obs_dim).
        actions (Tensor): Action tensor of shape (batch_size, action_dim).

    Forward Output:
        List[Tensor]: A list of Q-value tensors, one per objective. Each tensor has shape (batch_size, 1).

    Example:
        Given 3 objectives, calling forward(obs, actions) returns [Q1, Q2, Q3],
        where Qi is the predicted Q-value tensor for the i-th objective.
    """
    def __init__(self, base_net: nn.Module, heads: nn.ModuleList):
        super().__init__()
        self.base_net = base_net
        self.heads = heads

    def forward(self, obs: th.Tensor, actions: th.Tensor):
        """
        Forward pass through the shared base and each objective head.

        Args:
            obs (Tensor): Observation tensor of shape (batch_size, obs_dim).
            actions (Tensor): Action tensor of shape (batch_size, action_dim).

        Returns:
            List[Tensor]: List of Q-value outputs, one per objective.
        """
        # Concatenate observation and action along the last dimension
        x = th.cat([obs, actions], dim=1)
        # Pass through the shared base network
        shared_features = self.base_net(x)
        # Pass shared features through each head to get Q-values for each objective
        return [head(shared_features) for head in self.heads]


class SeparateQNet(nn.Module):
    """
    A neural network module consisting of separate networks for each objective,
    used for multi-objective Q-value estimation.

    This class is used in multi-objective reinforcement learning critic networks where:
    - There are multiple objectives (each requiring a separate Q-value output).
    - Each objective has its own independent neural network, with no shared layers between objectives.
    - This can be useful if objectives are diverse or conflicting and may benefit from learning separate representations.

    Args:
        nets (nn.ModuleList): A list of networks (one per objective), each typically an MLP,
                              that processes the concatenated observation and action and produces a single Q-value.

    Forward Inputs:
        obs (Tensor): Observation tensor of shape (batch_size, obs_dim).
        actions (Tensor): Action tensor of shape (batch_size, action_dim).

    Forward Output:
        List[Tensor]: A list of Q-value tensors, one per objective. Each tensor has shape (batch_size, 1).

    Example:
        Given 3 objectives, calling forward(obs, actions) returns [Q1, Q2, Q3],
        where Qi is the predicted Q-value tensor for the i-th objective, computed by its own network.
    """
    def __init__(self, nets: nn.ModuleList):
        super().__init__()
        self.nets = nets

    def forward(self, obs: th.Tensor, actions: th.Tensor):
        """
        Forward pass through each network specific to each objective.

        Args:
            obs (Tensor): Observation tensor of shape (batch_size, obs_dim).
            actions (Tensor): Action tensor of shape (batch_size, action_dim).

        Returns:
            List[Tensor]: List of Q-value outputs, one per objective.
        """
        # Concatenate observation and action along the last dimension
        x = th.cat([obs, actions], dim=1)
        # Pass through each separate network to get Q-values for each objective
        return [net(x) for net in self.nets]