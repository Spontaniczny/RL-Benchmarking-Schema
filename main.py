import json
import itertools
import os

from stable_baselines3.common.env_util import make_vec_env

# Local imports
from config import FixedConstants, TrainConfig
from src.algorithms.base_schema import CustomRLSchema


def load_constants() -> FixedConstants:
    """Define your project-wide constants here."""
    return FixedConstants(
        seed=42,
        device="cpu",
        verbose=0,
        tensorboard_log_dir="./logs/tb/",
        save_dir="./logs/models/",
        n_envs=1
    )


def run_single_experiment(cfg: TrainConfig):
    print(f"--> RUNNING: {cfg.run_name()}")

    env = make_vec_env(cfg.env_id, n_envs=cfg.n_envs, seed=cfg.seed)

    # 2. Setup Algorithm
    model = CustomRLSchema(
        policy=cfg.policy_type,
        env=env,
        learning_rate=cfg.learning_rate,
        policy_kwargs=cfg.policy_kwargs,
        tensorboard_log=cfg.tensorboard_log_dir,
        verbose=cfg.verbose,
        device=cfg.device,
        seed=cfg.seed,
        gamma=cfg.gamma,
        batch_size=cfg.batch_size
    )

    # 3. Learn
    model.learn(total_timesteps=cfg.total_timesteps, tb_log_name=cfg.run_name())

    # 4. Save
    save_path = os.path.join("logs/models", cfg.run_name())
    model.save(save_path)
    print(f"--> SAVED: {save_path}\n")


def main():
    # 1. Load Fixed Constants
    constants = load_constants()

    # 2. Load Grid Search JSON
    with open("experiments/exp1.json", "r") as f:
        data = json.load(f)

    print(f"Starting Experiment Batch: {data['experiment_name']}")

    # 3. Iterate over Environments
    for env_id, params in data["experiments"].items():

        # Create a list of all parameter names and their possible values
        # e.g. keys=['learning_rate', 'batch_size'], values=[[0.001, 0.0003], [64, 128]]
        keys = params.keys()
        values = params.values()

        # itertools.product generates every combination (Grid Search)
        for combination in itertools.product(*values):
            # Create a dict for this specific run: {learning_rate: 0.001, batch_size: 64, ...}
            run_params = dict(zip(keys, combination))

            # 4. Merge Constants + Variables into Strict Config
            # If any parameter is missing, TrainConfig(...) will crash here.
            try:
                experiment_config = TrainConfig(
                    # Constants
                    seed=constants.seed,
                    device=constants.device,
                    tensorboard_log_dir=constants.tensorboard_log_dir,
                    n_envs=constants.n_envs,

                    # Variables
                    env_id=env_id,
                    **run_params
                )

                # 5. Execute
                run_single_experiment(experiment_config)

            except TypeError as e:
                print(f"CRITICAL CONFIG ERROR: Missing parameter in JSON for {env_id}.")
                print(f"Details: {e}")
                exit(1)


if __name__ == "__main__":
    main()