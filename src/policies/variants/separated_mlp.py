import torch.nn as nn
from src.policies.base import BaseActorCriticPolicy


class SeparatedMlpPolicy(BaseActorCriticPolicy):
    """
    Variant: Independent networks for Actor and Critic.
    Configurable via 'policy_kwargs'.
    """

    def __init__(self, *args, **kwargs):
        self.hidden_dim = kwargs.pop("hidden_dim")

        super().__init__(*args, **kwargs)

    def build_custom_arch(self) -> nn.ModuleDict:
        h = self.hidden_dim

        # ACTOR
        policy_net = nn.Sequential(
            nn.Linear(self.features_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU()
        )

        # CRITIC
        value_net = nn.Sequential(
            nn.Linear(self.features_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU()
        )

        net = nn.ModuleDict({
            "policy_net": policy_net,
            "value_net": value_net
        })

        net.latent_dim_pi = h
        net.latent_dim_vf = h

        return net
