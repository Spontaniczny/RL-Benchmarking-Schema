import json
import itertools
import os
import sys
from typing import Dict, Any

import gymnasium as gym
import wandb
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed

# Local Imports
from config import FixedConstants, TrainConfig
from src.algorithms import ALGO_REGISTRY
from src.utils.callbacks import StrictWandbCallback


def run_single_experiment(cfg: TrainConfig):
    """
    Executes one training run based on the provided configuration.
    """
    # 1. Set Global Seed
    set_random_seed(cfg.seed)

    # 2. Initialize W&B (if enabled)
    run = None
    if cfg.use_wandb:
        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=cfg.__dict__,
            name=cfg.run_name(),
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )

    print(f"\n---> STARTING RUN: {cfg.run_name()}")

    # 3. Setup Environment
    env = make_vec_env(
        cfg.env_id,
        n_envs=cfg.n_envs,
        seed=cfg.seed
    )

    # 4. Get Algorithm Class
    if cfg.algo_class not in ALGO_REGISTRY:
        raise ValueError(f"Algorithm '{cfg.algo_class}' not found in registry.")
    AlgoClass = ALGO_REGISTRY[cfg.algo_class]

    # 5. Initialize Model
    # We unpack algo_params dynamically. If JSON is missing a param required
    # by the class __init__, Python will raise a TypeError here.
    try:
        model = AlgoClass(
            policy=cfg.policy_type,
            env=env,
            policy_kwargs=cfg.policy_kwargs,
            tensorboard_log=cfg.tensorboard_log_dir,
            verbose=cfg.verbose,
            device=cfg.device,
            seed=cfg.seed,
            **cfg.algo_params
        )
    except TypeError as e:
        print(f"CRITICAL ERROR: Parameter mismatch for {cfg.algo_class}.")
        print(f"Details: {e}")
        if run: run.finish()
        sys.exit(1)

    # 6. Prepare Callbacks
    callbacks = []

    save_path = os.path.join(cfg.save_dir, cfg.run_name())
    callbacks.append(
        CheckpointCallback(
            save_freq=max(cfg.total_timesteps // 5, 1),  # Save 5 checkpoints per run
            save_path=save_path,
            name_prefix="model",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
    )

    if cfg.use_wandb:
        # Standard W&B sync + Model Checkpointing
        callbacks.append(
            StrictWandbCallback(
                run_id=run.id,
                save_path=os.path.join(cfg.save_dir, cfg.run_name())
            )
        )

    # --- SAVE METADATA (New Block) ---
    # We save the full configuration so visualize.py knows what to load later
    os.makedirs(save_path, exist_ok=True)
    params_path = os.path.join(save_path, "params.json")
    with open(params_path, "w") as f:
        # We convert the dataclass to a dict for JSON serialization
        # We exclude 'device' and 'verbose' as they don't affect model architecture
        json.dump(cfg.__dict__, f, indent=4, default=str)
    print(f"Saved run metadata to: {params_path}")

    # 7. Train
    try:
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=CallbackList(callbacks),
            tb_log_name=cfg.run_name()
        )
        final_model_path = os.path.join(save_path, "final_model")
        model.save(final_model_path)
        print(f"Saved final model to: {final_model_path}.zip")

        if cfg.use_wandb and run:
            wandb.save(final_model_path + ".zip", base_path=cfg.save_dir)
            print("Uploaded final model to W&B.")
    except Exception as e:
        print(f"Training crashed: {e}")
        raise e
    finally:
        env.close()
        if run:
            run.finish()

    print(f"---> FINISHED RUN: {cfg.run_name()}\n")


def main():
    # 1. Load Constants from JSON
    try:
        constants = FixedConstants.from_json("config.json")
    except Exception as e:
        print(f"CRITICAL: Failed to load config.json. {e}")
        return

    # 2. Load JSON
    json_path = "experiments/exp1.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"Batch Name: {data.get('experiment_name', 'Unnamed')}")

    # 3. Iterate Experiments
    for experiment in data["experiments"]:
        env_id = experiment["env_id"]
        algo_class = experiment["algo_class"]

        # Merge common and specific params for grid search
        common_params = experiment.get("common_params", {})
        specific_params = experiment.get("algo_specific_params", {})

        # Combine all parameter grids
        all_grids = {**common_params, **specific_params}

        keys = all_grids.keys()
        values = all_grids.values()

        # Cartesian Product (Grid Search)
        for combination in itertools.product(*values):
            run_params = dict(zip(keys, combination))

            # Extract non-algo specific args
            total_timesteps = run_params.pop("total_timesteps")
            policy_type = run_params.pop("policy_type")
            policy_kwargs = run_params.pop("policy_kwargs")

            # The rest are algorithm specific (gamma, learning_rate, etc.)
            algo_params = run_params

            # Create Config Object
            cfg = TrainConfig(
                # Constants
                seed=constants.seed,
                device=constants.device,
                verbose=constants.verbose,
                n_envs=constants.n_envs,
                tensorboard_log_dir=constants.tensorboard_log_dir,
                save_dir=constants.save_dir,
                use_wandb=constants.use_wandb,
                wandb_project=constants.wandb_project,
                wandb_entity=constants.wandb_entity,

                # Experiment Variables
                env_id=env_id,
                algo_class=algo_class,
                total_timesteps=total_timesteps,
                policy_type=policy_type,
                policy_kwargs=policy_kwargs,
                algo_params=algo_params  # Passed as **kwargs to model
            )

            run_single_experiment(cfg)


if __name__ == "__main__":
    main()