import torch.nn as nn
from src.policies.base import BaseActorCriticPolicy


class SeparatedMlpPolicy(BaseActorCriticPolicy):
    """
    Variant: Independent networks for Actor and Critic.
    Configurable via 'policy_kwargs'.
    """

    def __init__(self, *args, **kwargs):
        # 1. Extract your custom arguments from kwargs
        # defaults to 64 if not in JSON
        self.hidden_dim = kwargs.pop("hidden_dim", 64)

        # 2. Call Parent Init (Crucial!)
        super().__init__(*args, **kwargs)

    def build_custom_arch(self) -> nn.ModuleDict:
        # Use the variable we captured in __init__
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

        # 3. Create ModuleDict
        net = nn.ModuleDict({
            "policy_net": policy_net,
            "value_net": value_net
        })

        # 4. SET OUTPUT DIMS (The Fix)
        # We know the last layer size is 'h'
        net.latent_dim_pi = h
        net.latent_dim_vf = h

        return net
