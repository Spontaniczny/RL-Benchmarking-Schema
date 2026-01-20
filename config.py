import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class FixedConstants:
    """
    System-wide constants loaded from JSON.
    """
    seed: int
    device: str
    verbose: int
    logs_root_dir: str
    use_wandb: bool
    wandb_project: str
    wandb_entity: Optional[str]

    @classmethod
    def from_json(cls, path: str) -> "FixedConstants":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found at: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        # 1. Check for missing keys (Strictness)
        required_keys = cls.__annotations__.keys()
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            raise ValueError(f"config.json is missing keys: {missing_keys}")

        # 2. Instantiate (Strict Type Checking)
        # We manually check types for critical fields to avoid silent bugs
        if not isinstance(data["seed"], int):
            raise TypeError(f"Seed must be int, got {type(data['seed'])}")

        return cls(**data)


@dataclass
class TrainConfig:
    """
    Combined configuration for a single training run.
    Merges FixedConstants with the JSON grid search parameters.
    """
    # From FixedConstants
    seed: int
    device: str
    verbose: int
    logs_root_dir: str
    use_wandb: bool
    wandb_project: str
    wandb_entity: Optional[str]

    # From Grid Search JSON
    group_name: str
    n_envs: int
    env_id: str
    algo_class: str
    total_timesteps: int
    policy_type: str
    policy_kwargs: Dict[str, Any]

    # Dynamic Algo Params (caught via **kwargs)
    algo_params: Dict[str, Any]

    def run_name(self) -> str:
        """
        Generates a clean, abbreviated run name.
        Format: Algo_envN_Params..._sSeed
        """
        # Dictionary of short codes.
        abbreviations = {
            "learning_rate": "lr",
            "batch_size": "bs",
            "total_timesteps": "T",
            "ent_coef": "ent",
            "gamma": "gam",
            "policy_kwargs": "pol",
            "buffer_size": "buf",
            "target_update_interval": "tau",
            "train_freq": "t_freq",
        }

        # 1. Start with Algo
        # We removed self.env_id since it is in the parent directory
        short_pol = self.policy_type.replace("Policy", "")
        parts = [
            self.algo_class,
            short_pol,  # <--- ADDED THIS
            f"env{self.n_envs}"
        ]

        # 2. Add n_envs (Critical for batch size context)

        # 3. Add Dynamic Params
        for k in sorted(self.algo_params.keys()):
            v = self.algo_params[k]
            short_key = abbreviations.get(k, k)

            # Format numbers nicely
            if isinstance(v, float):
                # Use scientific notation for small numbers like 1e-4
                val_str = f"{v:.0e}" if v < 0.001 else f"{v}"
            else:
                val_str = str(v)

            parts.append(f"{short_key}_{val_str}")

        # 4. Add Seed
        parts.append(f"s_{self.seed}")

        return "_".join(parts)