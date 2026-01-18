import json
import itertools
import os
import sys

import wandb
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

# Local Imports
from config import FixedConstants, TrainConfig
from src.algorithms import ALGO_REGISTRY


def run_single_experiment(cfg: TrainConfig):
    set_random_seed(cfg.seed)

    # 1. SETUP PATHS
    # Root for this specific batch of experiments
    # e.g. ./logs/Initial_Benchmarking/
    experiment_root = os.path.join(cfg.logs_root_dir, cfg.group_name)

    # Split: Models go to 'models', Logs go to 'tb'
    model_dir = os.path.join(experiment_root, "models", cfg.run_name())
    tb_dir = os.path.join(experiment_root, "tb")  # SB3 adds run_name automatically to this

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    # 2. W&B INIT
    run = None
    if cfg.use_wandb:
        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=cfg.__dict__,
            name=cfg.run_name(),
            group=cfg.group_name,
            job_type=cfg.algo_class,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )

    print(f"\n---> STARTING RUN: {cfg.run_name()}")
    print(f"     Model Dir: {model_dir}")
    print(f"     Tensorboard Dir: {tb_dir}")

    env = make_vec_env(cfg.env_id, n_envs=cfg.n_envs, seed=cfg.seed)

    if cfg.algo_class not in ALGO_REGISTRY:
        raise ValueError(f"Algorithm '{cfg.algo_class}' not found in registry.")
    AlgoClass = ALGO_REGISTRY[cfg.algo_class]

    try:
        model = AlgoClass(
            policy=cfg.policy_type,
            env=env,
            policy_kwargs=cfg.policy_kwargs,
            # SB3 will create a subfolder inside 'tb_dir' named after the run
            tensorboard_log=tb_dir,
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

    # 3. CALLBACKS & CHECKPOINTS
    callbacks = []
    callbacks.append(
        CheckpointCallback(
            save_freq=max(cfg.total_timesteps // 5, 1),
            save_path=model_dir,
            name_prefix="checkpoint",
            save_vecnormalize=True,
        )
    )

    # 4. SAVE METADATA (params.json)
    # Important: Save this in the model directory so visualize.py can find it
    params_path = os.path.join(model_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(cfg.__dict__, f, indent=4, default=str)

    # 5. TRAIN
    try:
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=CallbackList(callbacks),
            # This name is appended to tb_dir
            tb_log_name=cfg.run_name(),
            log_interval=1000
        )

        # 6. SAVE FINAL MODEL
        final_model_path = os.path.join(model_dir, "final_model")
        model.save(final_model_path)
        print(f"Saved final model to: {final_model_path}.zip")

        if cfg.use_wandb and run:
            wandb.save(final_model_path + ".zip", base_path=model_dir)
            wandb.save(params_path, base_path=model_dir)

    except Exception as e:
        print(f"Training crashed: {e}")
        raise e
    finally:
        env.close()
        if run:
            run.finish()

    print(f"---> FINISHED RUN: {cfg.run_name()}\n")


def main():
    try:
        constants = FixedConstants.from_json("config.json")
    except Exception as e:
        print(f"CRITICAL: Failed to load config.json. {e}")
        return

    json_path = "experiments/exp1.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    group_name = data.get("experiment_name", "Default_Experiment")
    print(f"Experiment Group: {group_name}")

    for experiment in data["experiments"]:
        env_id = experiment["env_id"]
        algo_class = experiment["algo_class"]

        common_params = experiment.get("common_params", {})
        specific_params = experiment.get("algo_specific_params", {})
        all_grids = {**common_params, **specific_params}

        keys = all_grids.keys()
        values = all_grids.values()

        for combination in itertools.product(*values):
            run_params = dict(zip(keys, combination))

            total_timesteps = run_params.pop("total_timesteps")
            policy_type = run_params.pop("policy_type")
            policy_kwargs = run_params.pop("policy_kwargs")

            algo_params = run_params

            cfg = TrainConfig(
                seed=constants.seed,
                device=constants.device,
                verbose=constants.verbose,
                n_envs=constants.n_envs,
                logs_root_dir=constants.logs_root_dir,
                use_wandb=constants.use_wandb,
                wandb_project=constants.wandb_project,
                wandb_entity=constants.wandb_entity,

                group_name=group_name,
                env_id=env_id,
                algo_class=algo_class,
                total_timesteps=total_timesteps,
                policy_type=policy_type,
                policy_kwargs=policy_kwargs,
                algo_params=algo_params
            )

            run_single_experiment(cfg)


if __name__ == "__main__":
    main()