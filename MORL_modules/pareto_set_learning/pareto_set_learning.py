# ALGO LOGIC: initialize agent here:
from morl_baselines.common.networks import layer_init, mlp, polyak_update

class HyperNetwork(nn.Module):
    """Hypernetwork that generates policy network parameters based on preference weights."""

    def __init__(
            self,
            weight_dim: int,
            output_dim: int = 1,
            net_arch: list = [256, 256],
            activation_fn = nn.ReLU
    ):
        """Initialize hypernetwork.

        Args:
            weight_dim: Dimension of preference weights
            target_network_sizes: List of parameter counts for each layer of target network
            hidden_dims: Hidden layer dimensions for hypernetwork
            activation: Activation function name
        """
        super(HyperNetwork, self).__init__()

        self.weight_dim = weight_dim
        self.output_dim = output_dim
        self.critic = mlp(
            input_dim=np.array(self.weight_dim).prod()
            output_dim=self.output_dim,
            net_arch=self.net_arch,
            activation_fn=nn.ReLU,
        )
        self.apply(layer_init)

        # Output layer to generate all target network parameters
        layers.append(nn.Linear(input_dim, self.total_target_params))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, weights: torch.Tensor) -> list:
        """Generate target network parameters from preference weights.

        Args:
            weights: Preference weights of shape (batch_size, weight_dim)

        Returns:
            List of parameter tensors for target network layers
        """
        # Generate all parameters
        all_params = self.network(weights)

        # Split into individual layer parameters
        params = []
        start_idx = 0

        for size in self.target_network_sizes:
            end_idx = start_idx + size
            layer_params = all_params[..., start_idx:end_idx]
            params.append(layer_params)
            start_idx = end_idx

        return params



class MOSoftQNetwork(nn.Module):
    """Soft Q-network: S, A -> ... -> |R| (multi-objective)."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        reward_dim,
        net_arch=[256, 256],
    ):
        """Initialize the soft Q-network."""
        super().__init__()
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.reward_dim = reward_dim
        self.net_arch = net_arch

        # S, A -> ... -> |R| (multi-objective)
        self.critic = mlp(
            input_dim=np.array(self.obs_shape).prod() + np.prod(self.action_shape),
            output_dim=self.reward_dim,
            net_arch=self.net_arch,
            activation_fn=nn.ReLU,
        )
        self.apply(layer_init)

    def forward(self, x, a):
        """Forward pass of the soft Q-network."""
        x = th.cat([x, a], dim=-1)
        x = self.critic(x)
        return x