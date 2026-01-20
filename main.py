import json
import sys
import os
import itertools
from typing import List, Dict, Any, Tuple

# Local Imports
from config import FixedConstants, TrainConfig
from src.runner import run_single_experiment


def load_configurations() -> Tuple[FixedConstants, Dict[str, Any]]:
    """
    Loads config files.
    """
    try:
        constants = FixedConstants.from_json("config.json")

        with open("experiments/policy_swap.json", "r") as f:
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
                # STRICT EXTRACTION
                # If these keys are missing in JSON, we want to know NOW.
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


def main():
    # 1. SETUP PHASE (Fail Fast)
    constants, grid_data = load_configurations()
    workload = generate_run_configs(grid_data, constants)

    print(f"=== Loaded {len(workload)} Experiments for Batch: {grid_data.get('experiment_name')} ===")

    # 2. EXECUTION PHASE (Fail Safe)
    failed_runs = []

    for i, cfg in enumerate(workload):
        print(f"\n--- Experiment {i + 1}/{len(workload)} ---")

        # We catch the boolean result here
        success = run_single_experiment(cfg)

        if not success:
            print(f"--> WARNING: Run {i + 1} Failed!")
            failed_runs.append(cfg.run_name())

    # 3. FINAL REPORTING
    print("\n" + "=" * 40)
    print(f"BATCH FINISHED.")
    print(f"Total Runs:  {len(workload)}")
    print(f"Successful:  {len(workload) - len(failed_runs)}")
    print(f"Failed:      {len(failed_runs)}")

    if failed_runs:
        print("\n[FAILED RUNS LIST]:")
        for name in failed_runs:
            print(f"- {name}")
    print("=" * 40)


if __name__ == "__main__":
    main()