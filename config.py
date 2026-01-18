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
    n_envs: int
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
    n_envs: int
    logs_root_dir: str
    use_wandb: bool
    wandb_project: str
    wandb_entity: Optional[str]

    # From Grid Search JSON
    group_name: str
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
        Works for ANY parameter name safely.
        """
        # Dictionary of short codes.
        # If a param isn't here, we just use its full name.
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

        # 1. Start with Algo and Env
        parts = [self.algo_class, self.env_id]

        # 2. Add Dynamic Params
        # We sort keys so the name is always consistent (lr_0.01_gam_0.99 vs gam_0.99_lr_0.01)
        for k in sorted(self.algo_params.keys()):
            v = self.algo_params[k]
            short_key = abbreviations.get(k, k)  # Fallback to 'k' if unknown

            # Format numbers nicely
            if isinstance(v, float):
                # If very small, use scientific notation (1e-4), else standard
                val_str = f"{v:.0e}" if v < 0.001 else f"{v}"
            else:
                val_str = str(v)

            parts.append(f"{short_key}_{val_str}")

        parts.append(f"s_{self.seed}")
        return "_".join(parts)