import argparse
from src.runner import run_single_experiment
from src.utils.config_loader import load_configurations, generate_run_configs


def parse_args():
    parser = argparse.ArgumentParser(description="Run RL Benchmarking Experiments")

    # Positional argument (required or optional, your choice).
    # nargs='?' means it's optional, defaulting to "grid_search.json" if not provided.
    parser.add_argument(
        "config_file",
        help="Path to the experiment JSON file"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config_file

    # 1. SETUP PHASE (Fail Fast)
    constants, grid_data = load_configurations(config_path)
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