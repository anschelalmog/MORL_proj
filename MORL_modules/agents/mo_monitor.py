import csv
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium  as gym
import numpy as np
import pandas

from stable_baselines3.common.monitor import Monitor, ResultsWriter

import csv
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from stable_baselines3.common.monitor import Monitor, ResultsWriter


class MOMonitor(Monitor):
    """
    A monitor wrapper for multi-objective Gym environments.
    This class extends the stable_baselines3 Monitor class to handle vector rewards.

    :param env: The environment to wrap
    :param filename: Path to a log file. If None, no file will be created use
    :param allow_early_resets: Allow reset before the episode is complete
    :param reset_keywords: Keywords for reset that need to be tracked
    :param info_keywords: Keywords for info that need to be tracked
    :param num_objectives: Size of the reward vector (number if objectives to the MORL enviroment)
    :param override_existing: appends to file if ``filename`` exists, otherwise
        override existing files (default)
    """

    def __init__(
            self,
            env: gym.Env,
            filename: Optional[str] = "MORL_modules/logs/mosac_monitor/monitor.csv",
            allow_early_resets: bool = True,
            reset_keywords: Tuple[str, ...] = (),
            info_keywords: Tuple[str, ...] = (),
            num_objectives: int = 4,
            override_existing: bool = True,
    ):
        # Initialize the parent Monitor class
        super().__init__(
            env=env,
            filename=filename,  # We'll handle the file creation ourselves
            allow_early_resets=allow_early_resets,
            reset_keywords=reset_keywords,
            info_keywords=info_keywords,
            override_existing=override_existing,
        )

        self.num_objectives = num_objectives
        # Override the rewards list to store vectors instead of scalars
        self.rewards = []
        # Override episode_returns to store vectors
        self.episode_returns = []

        # Create a custom results writer if filename is provided
        if filename is not None:
            env_id = env.spec.id if env.spec is not None else None
            self.results_writer = ResultsWriter(
                filename,
                header={"t_start": self.t_start, "env_id": str(env_id)},
                extra_keys=reset_keywords + info_keywords,
                override_existing=override_existing
            )

    def step(self, action):
        """
        Step the environment with the given action
        Handles vector rewards

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")

        observation, reward, terminated, truncated, info = self.env.step(action)

        # Handle reward as vector
        if np.isscalar(reward):
            # Convert scalar reward to vector if needed
            reward_vector = np.zeros(self.num_objectives, dtype=np.float32)
            reward_vector[0] = float(reward)
        else:
            # Ensure reward is a numpy array of the correct size
            if len(reward) != self.num_objectives:
                raise ValueError(f"Reward vector size {len(reward)} does not match expected size {self.num_objectives}")
            reward_vector = np.array(reward, dtype=np.float32)

        # Store the reward vector
        self.rewards.append(reward_vector)

        if terminated or truncated:
            self.needs_reset = True

            # Calculate the sum of reward vectors
            ep_rew = np.sum(self.rewards, axis=0)
            ep_rew = np.round(ep_rew, 6)
            ep_len = len(self.rewards)

            ep_info = {
                # Store the full reward vector as a numpy array in "r"
                "r": ep_rew,
                # Also keep individual components for CSV output
                "l": ep_len,
                "t": round(time.time() - self.t_start, 6)
            }
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, terminated, truncated, info





