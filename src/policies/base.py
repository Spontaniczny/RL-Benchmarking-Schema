from abc import ABC, abstractmethod
from typing import Tuple

import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


class AdapterMlpExtractor(nn.Module):
    """
    A wrapper that turns an nn.ModuleDict into a callable network
    that SB3 can execute.
    """

    def __init__(self, modules_dict: nn.ModuleDict):
        super().__init__()
        self.net_dict = modules_dict

        self.policy_net = modules_dict["policy_net"]
        self.value_net = modules_dict["value_net"]

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Run the features through both networks in parallel.
        Returns: (latent_policy, latent_value)
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class BaseActorCriticPolicy(ActorCriticPolicy, ABC):
    """
    Project-Specific Base Class.
    Handles the 'Contract' with SB3 so you don't have to.
    """

    def _build_mlp_extractor(self) -> None:
        """
        Overrides SB3's default builder.
        Delegates creation to 'build_custom_arch' and fixes SB3 compatibility.
        """
        raw_arch = self.build_custom_arch()

        if not isinstance(raw_arch, nn.ModuleDict):
            raise TypeError("build_custom_arch() must return an nn.ModuleDict")
        if "policy_net" not in raw_arch or "value_net" not in raw_arch:
            raise ValueError("Returned ModuleDict must contain 'policy_net' and 'value_net' keys.")


        if not hasattr(raw_arch, "latent_dim_pi"):
            raise AttributeError("Your custom network must have a 'latent_dim_pi' attribute set.")
        if not hasattr(raw_arch, "latent_dim_vf"):
            raise AttributeError("Your custom network must have a 'latent_dim_vf' attribute set.")

        self.mlp_extractor = AdapterMlpExtractor(raw_arch)

        self.mlp_extractor.latent_dim_pi = raw_arch.latent_dim_pi
        self.mlp_extractor.latent_dim_vf = raw_arch.latent_dim_vf

    @abstractmethod
    def build_custom_arch(self) -> nn.ModuleDict:
        """
        Define your custom PyTorch architecture here.

        Returns:
            nn.ModuleDict: Must contain:
                - "policy_net": Network trunk for the Actor
                - "value_net": Network trunk for the Critic

        MUST set .latent_dim_pi and .latent_dim_vf attributes on the dict!
        """
        pass