from abc import ABC, abstractmethod

from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


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
        # 1. Build the custom network
        self.mlp_extractor = self.build_custom_arch()

        # 2. VALIDATION
        if not isinstance(self.mlp_extractor, nn.ModuleDict):
            raise TypeError("build_custom_arch() must return an nn.ModuleDict")
        if "policy_net" not in self.mlp_extractor or "value_net" not in self.mlp_extractor:
            raise ValueError("Returned ModuleDict must contain 'policy_net' and 'value_net' keys.")

        # 3. CRITICAL FIX for SB3 Compatibility
        # SB3 expects the extractor to tell it the output size of the last layer
        # so it can build the final Action/Value heads.

        # We assume the user defined these attributes in their build function.
        # If not, we try to guess or raise an error.
        if not hasattr(self.mlp_extractor, "latent_dim_pi"):
            raise AttributeError("Your custom network must have a 'latent_dim_pi' attribute set.")
        if not hasattr(self.mlp_extractor, "latent_dim_vf"):
            raise AttributeError("Your custom network must have a 'latent_dim_vf' attribute set.")

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