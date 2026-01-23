import json
import itertools
import os
from typing import List, Dict, Any, Tuple

# Local Imports
from config import FixedConstants, TrainConfig


def load_configurations(experiment_file: str) -> Tuple[FixedConstants, Dict[str, Any]]:
    """
    Loads config files.
    Args:
        experiment_file: Path to the specific experiment JSON (e.g., 'experiments/policy_swap.json')
    """
    try:
        constants = FixedConstants.from_json("config.json")

        if not os.path.exists(experiment_file):
            raise FileNotFoundError(f"Experiment file not found: {experiment_file}")

        with open(experiment_file, "r") as f:
            grid_data = json.load(f)

        return constants, grid_data

    except Exception as e:
        print(f"\n[CRITICAL STARTUP ERROR]")
        raise e


def generate_run_configs(experiment_data: Dict[str, Any], constants: FixedConstants) -> List[TrainConfig]:
    """
    Generates the list of runs.
    Exits immediately if required keys (n_envs, total_timesteps) are missing.
    """
    configs = []
    group_name = experiment_data.get("experiment_name", "Default_Group")

    try:
        for exp_def in experiment_data["experiments"]:
            env_id = exp_def["env_id"]
            algo_class = exp_def["algo_class"]

            common = exp_def.get("common_params", {})
            specific = exp_def.get("algo_specific_params", {})
            grid = {**common, **specific}

            keys, values = zip(*grid.items())
            combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

            for params in combinations:
                total_timesteps = params.pop("total_timesteps")
                policy_type = params.pop("policy_type")
                policy_kwargs = params.pop("policy_kwargs")
                n_envs = params.pop("n_envs")

                cfg = TrainConfig(
                    seed=constants.seed,
                    device=constants.device,
                    verbose=constants.verbose,
                    logs_root_dir=constants.logs_root_dir,
                    use_wandb=constants.use_wandb,
                    wandb_project=constants.wandb_project,
                    wandb_entity=constants.wandb_entity,
                    group_name=group_name,
                    n_envs=n_envs,
                    env_id=env_id,
                    algo_class=algo_class,
                    total_timesteps=total_timesteps,
                    policy_type=policy_type,
                    policy_kwargs=policy_kwargs,
                    algo_params=params
                )
                configs.append(cfg)
        return configs

    except KeyError as e:
        print(f"\n[CRITICAL CONFIG ERROR]")
        raise f"Missing required parameter in grid_search.json: {e}"