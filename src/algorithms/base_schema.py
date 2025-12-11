from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch as th
import numpy as np

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule


class CustomRLSchema(BaseAlgorithm):
    """
    A strict schema for custom RL algorithms.
    inherits from SB3 BaseAlgorithm for compatibility with:
    - Callbacks
    - IO (Save/Load)
    - Vectorized Environments
    """

    def __init__(
            self,
            policy: Union[str, Type[BasePolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule],
            policy_kwargs: Dict[str, Any],
            tensorboard_log: str,
            verbose: int,
            device: Union[th.device, str],
            seed: int,
            # Add your custom params below (No defaults allowed!)
            gamma: float,
            batch_size: int,
    ):
        """
        CALLED: Once, when the algorithm is instantiated.
        PURPOSE: Initialize variables, hyperparameters, and the base class.
        """
        # We must call super().__init__ to set up SB3 internals (logger, env checks)
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            support_multi_env=True,
        )

        self.gamma = gamma
        self.batch_size = batch_size

        self._setup_model()

    def _setup_model(self) -> None:
        """
        CALLED: Automatically by __init__.
        PURPOSE: Construct the Policy, Optimizer, Replay Buffers, and Storage containers.
        """
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # 1. Initialize Policy (Using the SB3 internal factory)
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        # 2. Define Optimizer
        # self.optimizer = th.optim.Adam(...)

        # 3. Define Buffers/Storage
        # self.rollout_buffer = ...

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = None,
            tb_log_name: str = "CustomAlgo",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> "CustomRLSchema":
        """
        CALLED: By the user to start training.
        PURPOSE: The master training loop. Iterates through the environment and calls train().
        """
        # SB3 boilerplate to setup callbacks and timers
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            # 1. Interaction (Collect Rollouts)
            # action, _ = self.predict(last_obs)
            # new_obs, rewards, dones, infos = self.env.step(action)

            # 2. Store data
            # self.buffer.add(...)

            self.num_timesteps += 1

            # 3. Trigger Callback (Must be called every step for logging/saving to work)
            callback.on_step()

            # 4. Train (Update weights)
            self.train()

            # 5. Logging
            if log_interval is not None and self.num_timesteps % log_interval == 0:
                self._dump_logs()

        callback.on_training_end()
        return self

    def train(self) -> None:
        """
        CALLED: Inside the learn() loop.
        PURPOSE: Calculate losses, perform backpropagation, and update policy weights.
        """
        # 1. Sample from buffer
        # data = self.buffer.sample(self.batch_size)

        # 2. Forward pass & Loss calculation
        # loss = ...

        # 3. Optimization step
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # 4. Log metrics
        # self.logger.record("train/loss", loss.item())
        pass

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        CALLED: During inference/evaluation (and often during data collection).
        PURPOSE: Determine the action to take given an observation.
        """
        # Standard SB3 policy prediction
        return self.policy.predict(observation, state, episode_start, deterministic)

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        """
        CALLED: When model.save() is executed.
        PURPOSE: Tell SB3 which attributes are PyTorch tensors (state_dicts) vs standard variables.
        RETURNS: (state_dicts, torch_variables)
        """
        state_dicts = ["policy"]  # e.g. ["policy", "policy.optimizer"]
        torch_variables = []  # e.g. ["log_ent_coef"]
        return state_dicts, torch_variables

    def _excluded_save_params(self) -> List[str]:
        """
        CALLED: When model.save() is executed.
        PURPOSE: List attributes that should NOT be saved to the file (e.g. buffers, temporary vars).
        """
        return ["policy", "device", "env", "rollout_buffer"]

    def set_logger(self, logger) -> None:
        """
        CALLED: Before training starts or manually by user.
        PURPOSE: Define how/where to log (Terminal, Tensorboard, CSV).
        """
        self.logger = logger