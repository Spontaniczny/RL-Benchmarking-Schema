import argparse
import json
import os
import time
import sys
import gymnasium as gym
from typing import Dict, Any

# Stable Baselines Internals
from stable_baselines3.common.save_util import load_from_zip_file

# Local Imports
from src.algorithms import ALGO_REGISTRY


def load_params(model_dir: str) -> Dict[str, Any]:
    params_path = os.path.join(model_dir, "params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Could not find params.json in {model_dir}.")
    with open(params_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="Path to model folder")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    # 1. Load Metadata
    print(f"Loading metadata from {args.model_dir}...")
    config_data = load_params(args.model_dir)
    env_id = config_data["env_id"]
    algo_name = config_data["algo_class"]

    # 2. Prepare Strict Arguments
    # We reconstruct the exact dictionary your __init__ expects
    init_kwargs = {
        "policy": config_data["policy_type"],
        "policy_kwargs": config_data["policy_kwargs"],
        "seed": config_data["seed"],
        "verbose": 1,
        "tensorboard_log": None,
        "device": "auto",
        "learning_rate": config_data.get("learning_rate"),  # Fallback if at top level
    }

    # Flatten 'algo_params' (gamma, ent_coef, etc.) into the main dict
    if "algo_params" in config_data:
        init_kwargs.update(config_data["algo_params"])
        # Ensure learning_rate is correct if it was inside algo_params
        if "learning_rate" in config_data["algo_params"]:
            init_kwargs["learning_rate"] = config_data["algo_params"]["learning_rate"]

    # 3. Create Environment
    print(f"Creating environment: {env_id}")
    env = gym.make(env_id, render_mode="human")

    # 4. Get Class
    AlgoClass = ALGO_REGISTRY[algo_name]

    # 5. INSTANTIATE MANUALLY (The Clean Way)
    # We create the agent exactly as if we were training it.
    # This satisfies all strict checks.
    try:
        print("Initializing Agent...")
        model = AlgoClass(env=env, **init_kwargs)
    except TypeError as e:
        print(f"CRITICAL ERROR: Params.json does not match {algo_name}.__init__")
        print(f"Details: {e}")
        sys.exit(1)

    # 6. LOAD WEIGHTS
    # Now we simply overwrite the random weights with the saved ones.
    model_path = os.path.join(args.model_dir, "final_model.zip")
    if not os.path.exists(model_path):
        # Search for checkpoints
        files = [f for f in os.listdir(args.model_dir) if f.endswith(".zip")]
        files.sort()
        model_path = os.path.join(args.model_dir, files[-1])

    print(f"Loading weights from: {model_path}")

    # SB3 internal utility to load data
    data, params, pytorch_variables = load_from_zip_file(
        model_path,
        device="auto",
        print_system_info=False
    )

    # Inject parameters into our running model
    model.set_parameters(params, exact_match=True, device="auto")
    print("Weights loaded successfully.")

    # 7. Visualization Loop
    print("\nStarting Visualization...")
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                time.sleep(1.0 / args.fps)

            print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()