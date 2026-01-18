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
    tensorboard_log_dir: str
    save_dir: str
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
    tensorboard_log_dir: str
    save_dir: str
    use_wandb: bool
    wandb_project: str
    wandb_entity: Optional[str]

    # From Grid Search JSON
    env_id: str
    algo_class: str
    total_timesteps: int
    policy_type: str
    policy_kwargs: Dict[str, Any]

    # Dynamic Algo Params (caught via **kwargs)
    algo_params: Dict[str, Any]

    def run_name(self) -> str:
        """Generates a unique name based on key parameters."""
        # Clean the name to be file-system friendly
        param_str = "_".join([f"{k}_{v}" for k, v in self.algo_params.items()])
        return f"{self.algo_class}_{self.env_id}_{param_str}_seed_{self.seed}"