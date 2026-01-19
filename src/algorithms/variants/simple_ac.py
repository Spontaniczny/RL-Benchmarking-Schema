from typing import Any, Dict, Optional, Type, Union

import torch as th
import torch.nn.functional as F

from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy, ActorCriticCnnPolicy
from src.algorithms.base import BaseRLSchema


class SimpleActorCritic(BaseRLSchema):
    """
    A simple example implementation of an Actor-Critic algorithm.
    """
    POLICY_MAPPING = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy
    }

    def __init__(
            self,
            policy: Union[str, Type[BasePolicy]],
            env: Union[GymEnv, str],
            learning_rate: float,
            policy_kwargs: Dict[str, Any],
            tensorboard_log: Optional[str],
            verbose: int,
            device: Union[th.device, str],
            seed: int,
            # Custom Hyperparameters below
            gamma: float,
            ent_coef: float,
            **kwargs
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            **kwargs
        )

        self.gamma = gamma
        self.ent_coef = ent_coef

        # Initialize internal structures
        if kwargs.get("_init_setup_model", True):
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # Initialize the Policy using SB3's factory
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)
        self.optimizer = self.policy.optimizer

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 10,
            tb_log_name: str = "SimpleAC",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> "SimpleActorCritic":

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        # Reset Env
        obs = self.env.reset()

        last_log_time = 0

        while self.num_timesteps < total_timesteps:
            # 1. Select Action
            with th.no_grad():
                # Convert obs to tensor
                obs_tensor = th.as_tensor(obs).to(self.device)
                actions, values, log_probs = self.policy(obs_tensor)

            actions = actions.cpu().numpy()

            # 2. Step Env
            new_obs, rewards, dones, infos = self.env.step(actions)

            for info in infos:
                if "episode" in info:
                    self.ep_info_buffer.extend([info["episode"]])

            # 3. Store transition (In a real algo, use a Buffer)
            # For this simple example, we train immediately on the single step (Online)
            self.current_transition = {
                "obs": obs_tensor,
                "action": th.as_tensor(actions).to(self.device),
                "reward": th.as_tensor(rewards).to(self.device),
                "next_obs": th.as_tensor(new_obs).to(self.device),
                "done": th.as_tensor(dones).to(self.device)
            }

            self.num_timesteps += self.env.num_envs

            # 4. Train
            self.train()

            # 5. Callbacks & Logging
            callback.on_step()

            if log_interval is not None and (self.num_timesteps - last_log_time >= log_interval):
                self.dump_logs()
                last_log_time = self.num_timesteps

            obs = new_obs

        callback.on_training_end()
        return self

    def train(self) -> None:
        """
        One step of Gradient Descent.
        """
        self.policy.set_training_mode(True)

        # Unpack data
        data = self.current_transition
        obs = data["obs"]
        next_obs = data["next_obs"]
        reward = data["reward"]
        done = data["done"]

        # Forward Pass
        # We need values and log_probs for the specific actions taken
        # (Simplified for demonstration)
        values, log_prob, entropy = self.policy.evaluate_actions(obs, data["action"])

        values = values.flatten()

        # Bootstrapping (Next Value)
        with th.no_grad():
            _, next_values, _ = self.policy(next_obs)
            next_values = next_values.flatten()

            target_value = reward + (1 - done.float()) * self.gamma * next_values

        # Calculate Losses
        advantage = target_value - values

        y_true = target_value.flatten()
        if y_true.numel() > 1:
            y_pred = values
            var_y = th.var(y_true)
            if var_y < 1e-8:
                var_y = 1.0
            explained_var = 1 - th.var(y_true - y_pred) / var_y
        else:
            explained_var = th.tensor(0.0)

        actor_loss = -(log_prob * advantage.detach()).mean()
        critic_loss = F.mse_loss(values, target_value)
        entropy_loss = -th.mean(entropy)

        total_loss = actor_loss + 0.5 * critic_loss + self.ent_coef * entropy_loss

        # Optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Log
        self.logger.record("train/loss", total_loss.item())
        self.logger.record("train/actor_loss", actor_loss.item())
        self.logger.record("train/critic_loss", critic_loss.item())

        self.logger.record("train/entropy", entropy.mean().item())
        self.logger.record("train/value_estimate", values.mean().item())
        self.logger.record("train/explained_variance", explained_var.item())