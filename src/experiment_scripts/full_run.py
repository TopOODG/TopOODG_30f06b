# start run with potentially multiple model modes, seeds and datasets
# by calling single_experiment.py multiple times with different arguments
import argparse
import json
import multiprocessing as mp
import signal
import sys
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path().resolve()))
sys.path.insert(0, str(Path().resolve().parent))

from src.experiment_scripts.single_experiment import single_experiment


def force_shutdown_process_pool(pool):
    if pool is None:
        return

    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass

    processes = list(getattr(pool, "_processes", {}).values())
    for process in processes:
        if process is not None and process.is_alive():
            process.terminate()

    deadline = time.time() + 5
    for process in processes:
        if process is None:
            continue
        remaining = max(0, deadline - time.time())
        process.join(timeout=remaining)

    for process in processes:
        if process is not None and process.is_alive():
            process.kill()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full experiment with multiple seeds, datasets, and model modes."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file specifying experiment settings. "
        "Example file found in src/experiment_scripts/config_example.json.",
    )
    parser.add_argument(
        "--skip_aggregation",
        action="store_true",
        help="Whether to skip aggregation of results from all experiments after they finish.",
    )
    parser.add_argument(
        "--skip_training",
        "--aggregate_only",
        action="store_true",
        help="Whether to skip training and only aggregate results from existing experiment runs. "
        "This can be used to aggregate results after training has already been done, "
        "or to re-aggregate results after changing the aggregation code without having to re-run all experiments. "
        "Note that if this flag is set, the config file is still required "
        "and must contain the same experiment configurations as were used for the original runs, "
        "so that the code knows where to find the existing results to aggregate.",
    )
    parser.add_argument(
        "--run_folder",
        type=str,
        default=None,
        help="Name for the folder to save (optional) or load (required) results for this full run. "
        "If not specified, a folder with a timestamp will be created automatically.",
    )
    return parser.parse_args()


# Example json file:
"""
{
    "parallel_workers": 4,
    "gpus": [0, 1],
    "configs": [
        {
            "dataset": "lorenz63",
            "model": "shPLRNN",
            "num_features": 1,
            "feature_splitting": true,
            "num_seeds": 10
        },
        {
            "dataset": "lorenz63",
            "model": "shPLRNN",
            "num_features": 1,
            "feature_splitting": false,
            "num_seeds": 10
        },
        {
            "dataset": "selkov",
            "model": "node",
            "num_features": 2,
            "feature_splitting": false,
            "num_seeds": 10
        }
    ]
}
"""


def main():
    args = parse_args()

    # Load experiment configuration from JSON file
    with open(args.config, "r") as f:
        config = json.load(f)

    # Check if config contains required keys
    required_keys = ["configs", "parallel_workers"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config file must contain key '{key}'.")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_folder:
        folder_name = args.run_folder
    else:
        folder_name = (
            f"./results/{timestamp.split('_')[0]}/full_run_{timestamp.split('_')[1]}/"
        )
    print(f"Starting full experiment run at {timestamp} with config: {config}")

    futures = {}
    gpus = config.get("gpus")
    uses_mps = any(
        str(exp_config.get("device", "cpu")).lower().startswith("mps")
        for exp_config in config["configs"]
    )
    uses_cuda = bool(gpus) or any(
        str(exp_config.get("device", "cpu")).lower().startswith("cuda")
        for exp_config in config["configs"]
    )
    parallel_workers = int(config["parallel_workers"])

    if uses_cuda and gpus:
        max_cuda_workers = len(gpus)
        if parallel_workers > max_cuda_workers:
            print(
                "Reducing parallel_workers from "
                f"{parallel_workers} to {max_cuda_workers} so CUDA runs do not "
                "oversubscribe the same GPU across multiple worker processes."
            )
            parallel_workers = max_cuda_workers

    cpu_threads_per_worker = config.get("torch_num_threads")
    if cpu_threads_per_worker is None:
        available_cpu_threads = os.cpu_count() or 1
        cpu_threads_per_worker = max(
            1, available_cpu_threads // max(1, parallel_workers)
        )

    print(
        f"Using up to {parallel_workers} parallel workers with "
        f"torch_num_threads={cpu_threads_per_worker} per worker."
    )

    if uses_mps and config["parallel_workers"] > 1:
        print(
            "MPS experiments are executed sequentially because running them in "
            "ProcessPoolExecutor workers can stall on macOS."
        )

    def iter_experiment_args():
        for exp_config in config["configs"]:
            dataset_name = exp_config.get("dataset", config.get("dataset", "lorenz63"))
            model_name = exp_config.get("model", config.get("model", "shPLRNN"))
            num_features = exp_config.get("num_features", config.get("num_features", 1))
            feature_splitting = exp_config.get(
                "feature_splitting", config.get("feature_splitting", False)
            )
            num_seeds = exp_config.get("num_seeds", config.get("num_seeds", 1))
            epochs = exp_config.get("epochs", config.get("epochs", 100))
            steps_ahead = exp_config.get(
                "steps_ahead", config.get("steps_ahead", "lambda e: 4 + (e // 10)")
            )
            batch_size = exp_config.get("batch_size", config.get("batch_size", 1024))
            num_batches_per_epoch = exp_config.get(
                "num_batches_per_epoch", config.get("num_batches_per_epoch", 4)
            )
            learning_rate = exp_config.get(
                "learning_rate", config.get("learning_rate", 2e-3)
            )
            tf_alpha = exp_config.get("tf_alpha", config.get("tf_alpha", 0.25))
            tf_alpha_decay = exp_config.get(
                "tf_alpha_decay", config.get("tf_alpha_decay", 0.97)
            )
            tf_alpha_final = exp_config.get(
                "tf_alpha_final", config.get("tf_alpha_final", 0.05)
            )
            force_every = exp_config.get("force_every", config.get("force_every", 1))
            force_every_increase = exp_config.get(
                "force_every_increase", config.get("force_every_increase", 0)
            )
            feature_noise = exp_config.get(
                "feature_noise", config.get("feature_noise", "lambda e: 0.05 * (0.995 ** e)")
            )
            latent_dim = exp_config.get("latent_dim", config.get("latent_dim", 16))
            hidden_dim = exp_config.get("hidden_dim", config.get("hidden_dim", 128))
            validate_every = exp_config.get(
                "validate_every", config.get("validate_every", 10)
            )
            save_every = exp_config.get("save_every", config.get("save_every", 50))
            plot_every = exp_config.get("plot_every", config.get("plot_every", 10))
            features_l1 = exp_config.get("features_l1", config.get("features_l1", 0.0))
            feature_coupling_l1 = exp_config.get(
                "feature_coupling_l1", config.get("feature_coupling_l1", 0.0)
            )
            memory_manifold_reg = exp_config.get(
                "memory_manifold_reg", config.get("memory_manifold_reg", 1e-3)
            )
            memory_manifold_dims = exp_config.get(
                "memory_manifold_dims", config.get("memory_manifold_dims", 2)
            )

            for seed in range(num_seeds):
                yield {
                    "dataset": dataset_name,
                    "model": model_name,
                    "epochs": epochs,
                    "steps_ahead": steps_ahead,
                    "batch_size": batch_size,
                    "num_batches_per_epoch": num_batches_per_epoch,
                    "learning_rate": learning_rate,
                    "no_lr_scheduler": exp_config.get("no_lr_scheduler", config.get("no_lr_scheduler", False)),
                    "tf_alpha": tf_alpha,
                    "tf_alpha_decay": tf_alpha_decay,
                    "tf_alpha_final": tf_alpha_final,
                    "force_every": force_every,
                    "force_every_increase": force_every_increase,
                    "feature_noise": feature_noise,
                    "latent_dim": latent_dim,
                    "hidden_dim": hidden_dim,
                    "num_features": num_features,
                    "feature_splitting": feature_splitting,
                    "seed": seed,
                    "device": (
                        f"cuda:{gpus[seed % len(gpus)]}"
                        if gpus
                        else exp_config.get("device", "cpu")
                    ),
                    "torch_num_threads": exp_config.get(
                        "torch_num_threads",
                        config.get("torch_num_threads", cpu_threads_per_worker),
                    ),
                    "validate_every": validate_every,
                    "save_path": f"{folder_name}/{dataset_name}_{model_name}/{model_name}_features_{num_features}_splitting_{feature_splitting}/seed_{seed:02d}",
                    "save_every": save_every,
                    "verbose": False,
                    "save_predicted_trajectories": False,
                    "plot_every": plot_every,
                    "features_l1": features_l1,
                    "feature_coupling_l1": feature_coupling_l1,
                    "memory_manifold_reg": memory_manifold_reg,
                    "memory_manifold_dims": memory_manifold_dims,
                }

    if not args.skip_training:
        exp_arg_list = list(iter_experiment_args())

        if uses_mps:
            for exp_args in exp_arg_list:
                print(
                    f"Running experiment with dataset={exp_args['dataset']}, model={exp_args['model']}, num_features={exp_args['num_features']}, feature_splitting={exp_args['feature_splitting']}, seed={exp_args['seed']}"
                )
                single_experiment(exp_args)
        else:
            pool = None
            completed_successfully = False
            previous_sigint_handler = signal.getsignal(signal.SIGINT)
            previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

            def raise_keyboard_interrupt(signum, frame):
                raise KeyboardInterrupt(f"Received signal {signum}")

            signal.signal(signal.SIGINT, raise_keyboard_interrupt)
            signal.signal(signal.SIGTERM, raise_keyboard_interrupt)

            try:
                if uses_cuda:
                    # CUDA cannot be used safely from forked workers on Linux.
                    pool = ProcessPoolExecutor(
                        max_workers=parallel_workers,
                        mp_context=mp.get_context("spawn"),
                    )
                else:
                    pool = ProcessPoolExecutor(max_workers=parallel_workers)
                # Loop over all configurations specified in the config file
                # and run experiments for each configuration and seed
                # with up to parallel_workers parallel processes (if gpus are specified, use those for parallelization)
                for exp_args in exp_arg_list:
                    print(
                        f"Running experiment with dataset={exp_args['dataset']}, model={exp_args['model']}, num_features={exp_args['num_features']}, feature_splitting={exp_args['feature_splitting']}, seed={exp_args['seed']}"
                    )
                    future = pool.submit(single_experiment, exp_args)
                    futures[future] = exp_args
                    # Optionally, add a small delay between starting experiments to avoid overwhelming the system
                    time.sleep(1)

                for future in as_completed(futures):
                    exp_args = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        raise RuntimeError(
                            "Experiment failed for "
                            f"dataset={exp_args['dataset']}, model={exp_args['model']}, "
                            f"num_features={exp_args['num_features']}, "
                            f"feature_splitting={exp_args['feature_splitting']}, seed={exp_args['seed']}"
                        ) from exc

                completed_successfully = True
            except BaseException:
                print("Aborting full run. Cancelling queued jobs and stopping workers...")
                force_shutdown_process_pool(pool)
                raise
            finally:
                signal.signal(signal.SIGINT, previous_sigint_handler)
                signal.signal(signal.SIGTERM, previous_sigterm_handler)

                if pool is not None:
                    if completed_successfully:
                        pool.shutdown(wait=True)
                    else:
                        force_shutdown_process_pool(pool)

    else:
        print(
            "Skipping training and only aggregating results from existing experiment runs. "
        )
    if args.skip_aggregation and not args.skip_training:
        print(
            "Experiments completed. To aggregate results and generate plots, run this script again without the --skip_aggregation flag (and the same config file)."
        )
        return
    # Aggregate results from all experiments

    # collect wasserstein_distances.pth files from all experiment folders
    # and plot median + IQR W1-distance across seeds for each config vs control parameter
    w1_all = {}
    for exp_config in config["configs"]:
        dataset_name = exp_config.get("dataset", config.get("dataset", "lorenz63"))
        model_name = exp_config.get("model", config.get("model", "shPLRNN"))
        num_features = exp_config.get("num_features", config.get("num_features", 1))
        feature_splitting = exp_config.get(
            "feature_splitting", config.get("feature_splitting", False)
        )
        print(
            f"Aggregating results for config dataset={dataset_name}, model={model_name}, num_features={num_features}, feature_splitting={feature_splitting}"
        )

        w1_distances_id = []
        w1_distances_ood = []
        for seed in range(exp_config.get("num_seeds", config.get("num_seeds", 10))):
            w1_path = Path(
                f"{folder_name}/{dataset_name}_{model_name}/{model_name}_features_{num_features}_splitting_{feature_splitting}/seed_{seed:02d}/wasserstein_distances.pth"
            )
            if w1_path.exists():
                w1_data = torch.load(w1_path)
                w1_distances_id.append(w1_data["w1_distances_id"])
                w1_distances_ood.append(w1_data["w1_distances_ood"])
            else:
                print(
                    f"Warning: W1 distance file not found for seed {seed} at {w1_path}"
                )

        if w1_distances_id and w1_distances_ood:
            print(
                f"Successfully loaded W1 distance data for {len(w1_distances_id)} seeds. Aggregating results..."
            )
            # Plot median + IQR W1 distance vs control parameter for this config
            w1_distances_id = torch.tensor(w1_distances_id)
            w1_distances_ood = torch.tensor(w1_distances_ood)
            w1_id_median = torch.median(w1_distances_id, dim=0).values
            w1_id_iqr_lower = torch.quantile(w1_distances_id, 0.25, dim=0)
            w1_id_iqr_upper = torch.quantile(w1_distances_id, 0.75, dim=0)
            w1_ood_median = torch.median(w1_distances_ood, dim=0).values
            w1_ood_iqr_lower = torch.quantile(w1_distances_ood, 0.25, dim=0)
            w1_ood_iqr_upper = torch.quantile(w1_distances_ood, 0.75, dim=0)

            # Save aggregated results for this config
            aggregated_results = {
                "w1_id_median": w1_id_median,
                "w1_id_iqr_lower": w1_id_iqr_lower,
                "w1_id_iqr_upper": w1_id_iqr_upper,
                "w1_ood_median": w1_ood_median,
                "w1_ood_iqr_lower": w1_ood_iqr_lower,
                "w1_ood_iqr_upper": w1_ood_iqr_upper,
            }
            torch.save(
                aggregated_results,
                f"{folder_name}/{dataset_name}_{model_name}/{model_name}_features_{num_features}_splitting_{feature_splitting}/aggregated_wasserstein_distances.pth",
            )

            # plot using matplotlib
            plt.figure(figsize=(10, 6))
            # try to load dataset to get control parameter values for x-axis
            if os.path.exists(
                f"./data/{dataset_name}/{dataset_name}_dataset_standardized.npz"
            ):
                dataset = np.load(
                    f"./data/{dataset_name}/{dataset_name}_dataset_standardized.npz"
                )
            elif os.path.exists(dataset_name):
                dataset = np.load(dataset_name)
            else:
                print(
                    f"Warning: Dataset file not found for {dataset_name}, skipping plot."
                )
                continue
            cps_id = dataset["cps_id"]
            cps_ood = dataset["cps_ood"]
            all_cps = np.concatenate([cps_id, cps_ood])
            w1_median = torch.cat([w1_id_median, w1_ood_median])
            w1_iqr_lower = torch.cat([w1_id_iqr_lower, w1_ood_iqr_lower])
            w1_iqr_upper = torch.cat([w1_id_iqr_upper, w1_ood_iqr_upper])
            # sort by control parameter values
            sorted_indices = np.argsort(all_cps)
            all_cps = all_cps[sorted_indices]
            w1_median = w1_median[sorted_indices]
            w1_iqr_lower = w1_iqr_lower[sorted_indices]
            w1_iqr_upper = w1_iqr_upper[sorted_indices]

            if not dataset_name in w1_all:
                w1_all[dataset_name] = {}
            if not "cps_id" in w1_all[dataset_name]:
                w1_all[dataset_name]["cps_id"] = cps_id
            if not "cps_ood" in w1_all[dataset_name]:
                w1_all[dataset_name]["cps_ood"] = cps_ood
            w1_all[dataset_name][(model_name, num_features, feature_splitting)] = {
                "cps": all_cps,
                "w1_median": w1_median,
                "w1_iqr_lower": w1_iqr_lower,
                "w1_iqr_upper": w1_iqr_upper,
            }

            plt.plot(all_cps, w1_median, label="Median W1 Distance", color="blue")
            plt.fill_between(
                all_cps,
                w1_iqr_lower,  # type: ignore
                w1_iqr_upper,  # type: ignore
                color="blue",
                alpha=0.3,
            )
            # shade ID region in light grey
            plt.axvspan(
                cps_id.min(),
                cps_id.max(),
                color="grey",
                alpha=0.2,
                label="Training domain",
            )
            plt.xlabel("Control Parameter")
            plt.ylabel("Wasserstein-1 Distance")
            plt.title(
                f"W1 Distance vs Control Parameter for {model_name} on {dataset_name} (features={num_features}, splitting={feature_splitting})"
            )
            plt.legend()
            plt.grid()
            plt.savefig(
                f"{folder_name}/{dataset_name}_{model_name}/{model_name}_features_{num_features}_splitting_{feature_splitting}/w1_distance_plot.png"
            )
            plt.close()
        else:
            print(
                f"Warning: No W1 distance data found for config dataset={dataset_name}, model={model_name}, num_features={num_features}, feature_splitting={feature_splitting}. Skipping aggregation and plot."
            )

    combined_plot_path = f"{folder_name}/figures/"
    for dataset_name in w1_all:
        os.makedirs(Path(combined_plot_path, dataset_name), exist_ok=True)
        plt.figure(figsize=(10, 6))
        for i, key in enumerate(w1_all[dataset_name]):
            if key in ["cps_id", "cps_ood"]:
                continue
            print(
                f"Plotting W1 distance data for config {key} on dataset {dataset_name}"
            )
            cps = w1_all[dataset_name][key]["cps"]
            w1_median = w1_all[dataset_name][key]["w1_median"]
            w1_iqr_lower = w1_all[dataset_name][key]["w1_iqr_lower"]
            w1_iqr_upper = w1_all[dataset_name][key]["w1_iqr_upper"]
            label = f"{key[0]} features={key[1]} splitting={key[2]}"
            plt.plot(cps, w1_median, label=label, color="C" + str(i))
            plt.fill_between(
                cps, w1_iqr_lower, w1_iqr_upper, alpha=0.3, color="C" + str(i)
            )
        # shade ID region in light grey
        plt.axvspan(
            w1_all[dataset_name]["cps_id"].min(),
            w1_all[dataset_name]["cps_id"].max(),
            color="grey",
            alpha=0.2,
            label="Training domain",
        )
        plt.xlabel("Control Parameter")
        plt.ylabel("Wasserstein-1 Distance")
        plt.title(
            f"Comparison of W1 Distance vs Control Parameter for different configs on {dataset_name}"
        )
        plt.legend()
        plt.grid()
        plt.xlim(
            min(
                w1_all[dataset_name]["cps_id"].min(),
                w1_all[dataset_name]["cps_ood"].min(),
            ),
            max(
                w1_all[dataset_name]["cps_id"].max(),
                w1_all[dataset_name]["cps_ood"].max(),
            ),
        )
        plt.yscale("log")
        plt.savefig(f"{combined_plot_path}/{dataset_name}/w1_distance_comparison.png")
        plt.close()


if __name__ == "__main__":
    main()
