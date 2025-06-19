class MOReplayBuffer(ReplayBuffer):
    """
    Extended replay buffer that stores vector rewards for multi-objective RL.
    """
    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            num_objectives: int = 4,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
    ):
        """Initialize multi-objective replay buffer."""
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

        self.num_objectives = num_objectives
        
        # Modify rewards buffer to store vectors instead of scalars
        # Shape becomes (buffer_size, n_envs, num_objectives)
        self.rewards = np.zeros((self.buffer_size, self.n_envs, self.num_objectives), dtype=np.float32)

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:
        """Add a new transition to the buffer with vector reward."""
        # Reshape rewards if needed to ensure correct shape
        if reward.ndim == 1:
            reward = reward.reshape(-1, self.num_objectives)

        # Validate reward shape
        assert reward.shape[1] == self.num_objectives, f"Expected reward with {self.num_objectives} objectives, got {reward.shape[1]}"

        # Call parent method but handle reward differently
        super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size: int, env: Optional[GymEnv] = None) -> TensorDict:
        """Sample a batch of transitions with vector rewards."""
        # Sample indices
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)

        # Sample using parent implementation but preserve reward vectors
        data = self._get_samples(batch_inds, env=env)
        return data