import os
import json
import wandb
import traceback
import stable_baselines3
from typing import Optional, Dict, Tuple

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

# Local Imports
from config import TrainConfig
from src.algorithms import ALGO_REGISTRY
from src.policies import POLICY_REGISTRY


def run_single_experiment(cfg: TrainConfig) -> bool:
    """
    Returns True if successful, False if failed.
    """
    set_random_seed(cfg.seed)

    paths = _setup_paths(cfg)
    wandb_run = _init_wandb(cfg, paths)
    env, model = _build_env_and_model(cfg, paths)

    if env is None or model is None:
        _cleanup(env, wandb_run)
        return False

    success = _execute_training(model, env, cfg, paths, wandb_run)

    _cleanup(env, wandb_run)
    return success



def _setup_paths(cfg: TrainConfig) -> Dict[str, str]:
    experiment_root = os.path.join(cfg.logs_root_dir, cfg.group_name)
    run_name = cfg.run_name()

    paths = {
        "model_dir": os.path.join(experiment_root, "models", run_name),
        "tb_dir": os.path.join(experiment_root, "tb"),
        "params_file": os.path.join(experiment_root, "models", run_name, "params.json")
    }

    os.makedirs(paths["model_dir"], exist_ok=True)
    os.makedirs(paths["tb_dir"], exist_ok=True)

    print(f"\n---> STARTING RUN: {run_name}")
    return paths


def _init_wandb(cfg: TrainConfig, paths: Dict[str, str]):
    if not cfg.use_wandb:
        return None
    try:
        return wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=cfg.__dict__,
            name=cfg.run_name(),
            group=cfg.group_name,
            job_type=cfg.algo_class,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            dir=paths["model_dir"]
        )
    except Exception as e:
        print(f"WARNING: WandB failed to init. Continuing without it. Error: {e}")
        return None


def _build_env_and_model(cfg: TrainConfig, paths: Dict[str, str]) -> Tuple[Optional[VecEnv], Optional[BaseAlgorithm]]:
    try:
        env = make_vec_env(cfg.env_id, n_envs=cfg.n_envs, seed=cfg.seed)

        if cfg.algo_class in ALGO_REGISTRY:
            AlgoClass = ALGO_REGISTRY[cfg.algo_class]
        elif hasattr(stable_baselines3, cfg.algo_class):
            AlgoClass = getattr(stable_baselines3, cfg.algo_class)
        else:
            print(f"CRITICAL: Algorithm '{cfg.algo_class}' not found.")
            return None, None

        policy_id = cfg.policy_type
        if policy_id in POLICY_REGISTRY:
            policy_class = POLICY_REGISTRY[policy_id]
        else:
            policy_class = policy_id

        model = AlgoClass(
            policy=policy_class,
            env=env,
            policy_kwargs=cfg.policy_kwargs,
            tensorboard_log=paths["tb_dir"],
            verbose=cfg.verbose,
            device=cfg.device,
            seed=cfg.seed,
            **cfg.algo_params
        )
        return env, model

    except Exception as e:
        print(f"SETUP FAILED for {cfg.run_name()}")
        print(f"Error: {e}")
        traceback.print_exc()
        return None, None


def _execute_training(model: BaseAlgorithm, env: VecEnv, cfg: TrainConfig, paths: Dict[str, str], wandb_run) -> bool:
    with open(paths["params_file"], "w") as f:
        json.dump(cfg.__dict__, f, indent=4, default=str)

    callbacks = []
    callbacks.append(
        CheckpointCallback(
            save_freq=max(cfg.total_timesteps // 5, 1),
            save_path=paths["model_dir"],
            name_prefix="checkpoint",
            save_vecnormalize=True,
            verbose=cfg.verbose
        )
    )

    try:
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=CallbackList(callbacks),
            tb_log_name=cfg.run_name(),
            log_interval=1000
        )

        final_path = os.path.join(paths["model_dir"], "final_model")
        model.save(final_path)
        print(f"     SUCCESS: Training Finished.")

        if wandb_run:
            print("     --> Uploading models to WandB...")
            # This syncs the entire folder: params.json, checkpoints, final_model.zip
            wandb.save(os.path.join(paths["model_dir"], "*"), base_path=paths["model_dir"])

        return True

    except Exception:
        print(f"\n{'!' * 40}")
        print(f"!!! TRAINING CRASHED: {cfg.run_name()} !!!")
        print(f"{'!' * 40}")
        traceback.print_exc()

        if wandb_run:
            print("     --> Attempting to upload crash artifacts...")
            wandb.save(os.path.join(paths["model_dir"], "*"), base_path=paths["model_dir"])

        return False


def _cleanup(env: Optional[VecEnv], wandb_run) -> None:
    if env:
        env.close()
    if wandb_run:
        wandb_run.finish()