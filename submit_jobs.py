import os
import subprocess
import argparse
import json
import sys
from typing import Dict, Any
from src.utils.config_loader import load_configurations, generate_run_configs


def load_slurm_config(config_path: str = "config_cyfronet.json") -> Dict[str, Any]:
    """
    Loads the SLURM configuration. Fails fast if missing.
    """
    if not os.path.exists(config_path):
        print(f"CRITICAL: SLURM config file '{config_path}' not found.")
        sys.exit(1)

    with open(config_path, "r") as f:
        return json.load(f)


def create_slurm_script(job_name: str, cmd: str, log_dir: str, slurm_cfg: Dict[str, Any]) -> str:
    """
    Generates the content of a .sh file for sbatch using loaded config.
    """
    # 1. Generate Module Loads
    modules_str = "\n".join([f"module load {m}" for m in slurm_cfg.get("modules", [])])

    # 2. Generate Script
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}_%j.out
#SBATCH --error={log_dir}/{job_name}_%j.err
#SBATCH --account={slurm_cfg['account']}
#SBATCH --partition={slurm_cfg['partition']}
#SBATCH --time={slurm_cfg['time']}
#SBATCH --mem={slurm_cfg['mem']}
#SBATCH --cpus-per-task={slurm_cfg['cpus_per_task']}
#SBATCH --gres=gpu:{slurm_cfg['gpus']}

# --- Environment Setup ---
{modules_str}

# Activate Virtual Environment
source .venv/bin/activate

# WandB Mode (Offline to prevent crashes on compute nodes)
export WANDB_MODE={slurm_cfg.get('wandb_mode', 'offline')}

# Info
echo "Starting Job: {job_name}"
echo "Node: $(hostname)"
echo "WandB Mode: $WANDB_MODE"

# --- Execution ---
{cmd}
"""
    return script


def main():
    parser = argparse.ArgumentParser(description="Submit parallel SLURM jobs.")
    parser.add_argument("config_file", help="Path to experiment JSON")
    args = parser.parse_args()

    # 1. Load Configurations
    print(f"--> Reading experiments from: {args.config_file}")
    constants, grid_data = load_configurations(args.config_file)
    workload = generate_run_configs(grid_data, constants)

    # 2. Load SLURM Config
    print(f"--> Reading SLURM settings from: {args.slurm_config}")
    slurm_cfg = load_slurm_config()

    # 3. Create SLURM Logs Directory
    slurm_logs_dir = os.path.join(constants.logs_root_dir, "slurm_logs")
    os.makedirs(slurm_logs_dir, exist_ok=True)

    print(f"--> Found {len(workload)} experiments to submit. Starting batch submission...")
    print("-" * 50)

    # 4. Iterate and Submit
    for i, cfg in enumerate(workload):
        # A. Create Single-Experiment JSON
        single_exp_name = f"job_{i}_{cfg.run_name()}"
        single_exp_file = os.path.join(slurm_logs_dir, f"{single_exp_name}.json")

        single_exp_data = {
            "experiment_name": grid_data.get("experiment_name", "Batch"),
            "experiments": [
                {
                    "env_id": cfg.env_id,
                    "algo_class": cfg.algo_class,
                    "common_params": {
                        "n_envs": [cfg.n_envs],
                        "total_timesteps": [cfg.total_timesteps]
                    },
                    "algo_specific_params": {
                        "policy_type": [cfg.policy_type],
                        "policy_kwargs": [cfg.policy_kwargs],
                        # Add back algo params
                        **{k: [v] for k, v in cfg.algo_params.items()}
                    }
                }
            ]
        }

        with open(single_exp_file, "w") as f:
            json.dump(single_exp_data, f, indent=4)

        # B. Define Command
        cmd = f"python main.py {single_exp_file}"

        # C. Generate SLURM Script
        slurm_content = create_slurm_script(single_exp_name, cmd, slurm_logs_dir, slurm_cfg)
        slurm_script_path = os.path.join(slurm_logs_dir, f"{single_exp_name}.sh")

        with open(slurm_script_path, "w") as f:
            f.write(slurm_content)

        # D. Submit Immediately
        try:
            result = subprocess.run(["sbatch", slurm_script_path], capture_output=True, text=True)
            if result.returncode == 0:
                # Output looks like: "Submitted batch job 123456"
                job_id = result.stdout.strip().split()[-1]
                print(f"[{i + 1}/{len(workload)}] Submitted: {job_id} -> {cfg.run_name()}")
            else:
                print(f"[{i + 1}/{len(workload)}] ERROR submitting {single_exp_name}: {result.stderr}")
        except FileNotFoundError:
            print("CRITICAL: 'sbatch' command not found. Are you running this on the login node?")
            sys.exit(1)

    print("-" * 50)
    print(f"Done. Submitted {len(workload)} jobs.")


if __name__ == "__main__":
    main()