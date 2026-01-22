import os
import subprocess
import argparse
import json
import sys
from typing import Dict, Any
from src.utils.config_loader import load_configurations, generate_run_configs


def run_setup(slurm_cfg: Dict[str, Any]) -> None:
    """
    Create scratch venv + install deps using uv, using the module environment from config_cyfronet.json.
    """
    rl_root = slurm_cfg["venv_root_dir"]  # this is your scratch root folder
    venv_path = os.path.join(rl_root, ".RL_venv")
    cache_path = os.path.join(rl_root, ".uv_cache")

    os.makedirs(rl_root, exist_ok=True)
    os.makedirs(cache_path, exist_ok=True)

    modules = slurm_cfg.get("modules", [])

    # IMPORTANT: 'module' works only in a shell, so we run via bash -lc
    modules_str = "\n".join([f"module load {m}" for m in modules])

    cmd = f"""
module purge
{modules_str}

export RL_ROOT="{rl_root}"
export UV_PROJECT_ENVIRONMENT="{venv_path}"
export UV_CACHE_DIR="{cache_path}"
mkdir -p "{cache_path}"

if [ ! -d "{venv_path}" ]; then
  uv venv "{venv_path}"
fi

uv sync --frozen
"""

    print("==> Running setup (uv venv + uv sync --frozen) ...")
    result = subprocess.run(["bash", "-lc", cmd], text=True)

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        print("\nCRITICAL: Setup failed.")
        sys.exit(1)

    print(result.stdout)
    print(f"==> Setup OK. Venv at: {venv_path}")


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
    venv_path = os.path.join(slurm_cfg["venv_root_dir"], ".RL_venv")

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
module purge
{modules_str}

# Activate Virtual Environment
source {venv_path}/bin/activate

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
    parser.add_argument("config_file", nargs="?", help="Path to experiment JSON")
    parser.add_argument("--setup-only", action="store_true", help="Create venv + install deps, then exit")
    args = parser.parse_args()

    if not args.setup_only and args.config_file is None:
        print("CRITICAL: Missing config_file. Usage:")
        print("  python submit_jobs.py --setup-only")
        print("  python submit_jobs.py experiments/exp.json")
        sys.exit(1)

    # load SLURM Config
    print(f"--> Reading SLURM settings from: config_cyfronet.json")
    slurm_cfg = load_slurm_config()

    # check if it's a run only for setting up the venv
    if args.setup_only:
        run_setup(slurm_cfg)
        return

    # check if the venv exists
    venv_path = os.path.join(slurm_cfg["venv_root_dir"], ".RL_venv")
    if not os.path.exists(os.path.join(venv_path, "bin", "activate")):
        print(f"CRITICAL: venv not found at: {venv_path}")
        print("Run: python submit_jobs.py --setup-only")
        sys.exit(1)

    # load experiment configurations
    print(f"--> Reading experiments from: {args.config_file}")
    constants, grid_data = load_configurations(args.config_file)
    workload = generate_run_configs(grid_data, constants)

    # create SLURM outputs directory for out, err, jsons and scripts
    slurm_root = slurm_cfg["output_dir"]
    slurm_logs_dir = os.path.join(slurm_root, "slurm_logs")
    single_json_dir = os.path.join(slurm_root, "single_run_json")
    slurm_scripts_dir = os.path.join(slurm_root, "sbatch_scripts")

    os.makedirs(slurm_logs_dir, exist_ok=True)
    os.makedirs(single_json_dir, exist_ok=True)
    os.makedirs(slurm_scripts_dir, exist_ok=True)

    print(f"--> Found {len(workload)} experiments to submit. Starting batch submission...")
    print("-" * 50)

    # iterate and submit
    for i, cfg in enumerate(workload):
        # create single-experiment JSON
        single_exp_name = f"job_{i}_{cfg.run_name()}"
        single_exp_file = os.path.join(single_json_dir, f"{single_exp_name}.json")

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

        # define Command
        cmd = f'python main.py "{single_exp_file}"'

        # generate SLURM Script
        slurm_content = create_slurm_script(single_exp_name, cmd, slurm_logs_dir, slurm_cfg)
        slurm_script_path = os.path.join(slurm_scripts_dir, f"{single_exp_name}.sh")

        with open(slurm_script_path, "w") as f:
            f.write(slurm_content)

        # submit to SLURM immediately using SBATCH
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