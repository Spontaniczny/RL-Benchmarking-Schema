import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union, Tuple

import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import safe_mean


class BaseRLSchema(BaseAlgorithm, ABC):
    """
    Parent class for all custom algorithms.
    Enforces strict typing and implementation of train/learn.
    """

    POLICY_MAPPING: Dict[str, Type[BasePolicy]] = {}

    def __init__(
            self,
            policy: Union[str, Type[BasePolicy]],
            env: Union[GymEnv, str],
            learning_rate: float,
            policy_kwargs: Dict[str, Any],
            tensorboard_log: Optional[str],
            verbose: int,
            device: Union[torch.device, str],
            seed: int,
            support_multi_env: bool = True,
            **kwargs
    ):
        if isinstance(policy, str):
            if policy in self.POLICY_MAPPING:
                policy = self.POLICY_MAPPING[policy]
            else:
                raise ValueError(
                    f"Policy '{policy}' is not defined in {self.__class__.__name__}.POLICY_MAPPING.\n"
                    f"Available options: {list(self.POLICY_MAPPING.keys())}"
                )

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            support_multi_env=support_multi_env,
            **kwargs
        )
        self.start_num_timesteps = 0

    def _setup_learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            reset_num_timesteps: bool = True,
            tb_log_name: str = "run",
            progress_bar: bool = False,
    ) -> Tuple[int, MaybeCallback]:
        """
        Overriding _setup_learn to capture the starting timestep.
        This fixes the 'Unresolved attribute' error in dump_logs.
        """
        total_timesteps, callback = super()._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        self.start_num_timesteps = self.num_timesteps
        return total_timesteps, callback

    def dump_logs(self) -> None:
        """
        Write log data to file/console.
        """
        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)

        # Calculate FPS based on steps taken THIS session
        fps = int((self.num_timesteps - self.start_num_timesteps) / time_elapsed)

        # Record standard system metrics
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))

        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

        self.logger.dump(step=self.num_timesteps)

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 100,
            tb_log_name: str = "run",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False
    ) -> "BaseRLSchema":
        pass