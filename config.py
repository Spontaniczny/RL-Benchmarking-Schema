from dataclasses import dataclass, field
from typing import Any, Dict, List, Type, Union, Optional


@dataclass(frozen=True)
class FixedConstants:
    """
    These parameters are CONSTANT across all grid search runs.
    """
    # System Settings
    seed: int
    device: str  # "cpu", "cuda", "auto"
    verbose: int

    # Paths
    tensorboard_log_dir: str
    save_dir: str

    # Environment Defaults (can be overridden, but usually fixed per project)
    n_envs: int


@dataclass
class TrainConfig:
    """
    The Schema for a single training run.
    Constructed by merging FixedConstants + GridSearch Variables.
    """
    # From FixedConstants
    seed: int
    device: str
    tensorboard_log_dir: str
    n_envs: int
    verbose: int

    # From GridSearch (Variables)
    env_id: str
    total_timesteps: int
    learning_rate: float
    batch_size: int
    gamma: float
    policy_type: str
    policy_kwargs: Dict[str, Any]

    # Computed / Helper to get a unique run name
    def run_name(self) -> str:
        return f"{self.env_id}__lr_{self.learning_rate}__bs_{self.batch_size}__seed_{self.seed}"