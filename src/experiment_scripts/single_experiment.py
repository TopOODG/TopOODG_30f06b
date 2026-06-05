# run single model on single dataset with single hyperparameter configuration
import argparse
import os
from datetime import datetime
import json
import numpy as np
import re

import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from typing import cast

# Add parent directory to path for imports
sys.path.insert(0, str(Path().resolve()))
sys.path.insert(0, str(Path().resolve().parent))

from src.models.shPLRNN import shPLRNN
from src.models.node import NODE
from src.models.encoder_decoder import ConcatEncoder, ReadoutDecoder
from src.training.teacher_forcing import SparseGeneralizedTeacherForcing as SGTF
from src.training.teacher_forcing import ManifoldGeneralizedTeacherForcing as MGTF
from src.training.training import train_model
from src.training.regularization import (
    FeaturesL1Regularizer,
    FeatureCouplingL1Regularizer,
    MemoryManifoldRegularizer,
)
from src.extrapolation import FeatureExtrapolation
from src.metrics import wasserstein_distance_1d

from data.dataset import get_dataloader, load_dataset, get_datasets, get_control_params


class ArgDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]


def _combine_true_bifurcation_points(bifurcation_data, domain):
    extrema = bifurcation_data[f"{domain}_extrema"]
    extrema_counts = bifurcation_data[f"{domain}_extrema_counts"]
    fixed_points = bifurcation_data[f"{domain}_fixed_points"]
    fixed_point_counts = bifurcation_data[f"{domain}_fixed_points_counts"]

    return [
        np.concatenate(
            [
                extrema[i, : extrema_counts[i]],
                fixed_points[i, : fixed_point_counts[i]],
            ]
        )
        for i in range(extrema.shape[0])
    ]


def _combine_predicted_bifurcation_points(bifurcation_points):
    return [
        np.concatenate(
            [
                points["extrema"].detach().cpu().numpy(),
                points["fixed_points"].detach().cpu().numpy(),
            ]
        )
        for _, points in sorted(bifurcation_points.items())
    ]


def _compute_wasserstein_distances(true_points, predicted_points):
    return [
        wasserstein_distance_1d(
            torch.as_tensor(predicted_cp_points),
            torch.as_tensor(true_cp_points),
        )
        for true_cp_points, predicted_cp_points in zip(true_points, predicted_points)
    ]


def _prepare_trajectories_for_save(trajectories, stride=1, dtype="float32"):
    if stride > 1:
        trajectories = trajectories[:, ::stride]

    trajectories = trajectories.detach().cpu()

    if dtype == "float16":
        trajectories = trajectories.to(torch.float16)
    elif dtype == "float32":
        trajectories = trajectories.to(torch.float32)
    else:
        raise ValueError(
            f"Unsupported trajectory save dtype '{dtype}'. Use 'float16' or 'float32'."
        )

    return trajectories


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a single experiment with specified configuration."
    )

    experiment_group = parser.add_argument_group("Experiment")
    experiment_group.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset or file to use."
    )
    experiment_group.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model architecture to use (e.g., shPLRNN or node).",
    )

    training_group = parser.add_argument_group("Training")
    training_group.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    training_group.add_argument(
        "--steps_ahead",
        type=str,
        default="lambda e: 4 + (e // 10)",
        help="Number of steps ahead to predict during training."
        + " Can be an integer or a lambda function of epoch"
        + " (e.g., 'lambda e: 4 + (e // 10)').",
    )
    training_group.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size for training."
    )
    training_group.add_argument(
        "--num_batches_per_epoch",
        type=int,
        default=4,
        help="Limit the number of batches per epoch (for faster training). If None, use all batches.",
    )
    training_group.add_argument(
        "--learning_rate",
        "--lr",
        dest="learning_rate",
        type=float,
        default=2e-3,
        help="Learning rate for the optimizer.",
    )
    training_group.add_argument(
        "--no-lr-scheduler",
        action="store_true",
        help="Disable learning rate scheduler.",
    )
    training_group.add_argument(
        "--tf_alpha",
        type=float,
        default=0.25,
        help="Teacher forcing alpha (if None, no teacher forcing).",
    )
    training_group.add_argument(
        "--tf_alpha_decay",
        type=float,
        default=0.97,
        help="Exponential decay rate for teacher forcing alpha per epoch (if tf_alpha is set).",
    )
    training_group.add_argument(
        "--tf_alpha_final",
        type=float,
        default=0.05,
        help="Minimum teacher forcing alpha to decay to (if tf_alpha is set).",
    )
    training_group.add_argument(
        "--force_every",
        type=int,
        default=4,
        help="Force every N steps (if None, no forcing).",
    )
    training_group.add_argument(
        "--force_every_increase",
        type=int,
        default=0,
        help="Increase force_every by 1 every N epochs (if force_every is set).",
    )
    training_group.add_argument(
        "--feature_noise",
        type=str,
        default="lambda e: 0.05 * (0.995 ** e)",
        help="Standard deviation of Gaussian noise added to features during training."
        " Can be a float or a lambda function of epoch (e.g., 'lambda e: 0.05 * (0.995 ** e)').",
    )

    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--latent_dim", type=int, default=16, help="Dimensionality of the latent space."
    )
    model_group.add_argument(
        "--hidden_dim", type=int, default=128, help="Dimensionality of hidden layer."
    )
    model_group.add_argument(
        "--num_features",
        type=int,
        default=1,
        help="Number of features in the dataset.",
    )
    model_group.add_argument(
        "--feature_splitting",
        type=bool,
        default=True,
        help="Whether to use feature splitting into positional"
        + " and dynamical features.",
    )

    runtime_group = parser.add_argument_group("Runtime")
    runtime_group.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the experiment on (e.g., 'cpu', 'mps', 'cuda', 'cuda:<device_id>').",
    )
    runtime_group.add_argument(
        "--torch_num_threads",
        type=int,
        default=None,
        help="Optional cap for the number of CPU threads used by this process.",
    )
    runtime_group.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    runtime_group.add_argument(
        "--verbose", action="store_true", help="Whether to print training progress."
    )

    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--save_path",
        type=str,
        default="auto",
        help="Path to save model checkpoints. If 'auto', saves to './results/{date}/single_exp_{timestamp}_{dataset}_{model}'.",
    )
    output_group.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Frequency (in epochs) of saving model checkpoints.",
    )
    output_group.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Optional path to a checkpoint or final model file to load before evaluation or resumed training.",
    )
    output_group.add_argument(
        "--new_save_dir",
        action="store_true",
        help="Create a new output directory even when loading from an existing checkpoint or final model.",
    )
    output_group.add_argument(
        "--save_predicted_trajectories",
        action="store_true",
        help="Save full predicted rollout trajectories alongside bifurcation points.",
    )
    output_group.add_argument(
        "--trajectory_save_stride",
        type=int,
        default=1,
        help="Keep every Nth predicted rollout step when saving trajectories.",
    )
    output_group.add_argument(
        "--trajectory_save_dtype",
        type=str,
        default="float32",
        choices=["float16", "float32"],
        help="Data type used when saving predicted trajectories.",
    )
    output_group.add_argument(
        "--plot_every",
        type=int,
        default=10,
        help="Frequency (in epochs) of saving training plots.",
    )
    output_group.add_argument(
        "--validate_every",
        type=int,
        default=10,
        help="Frequency (in epochs) of running validation during training.",
    )

    regularization_group = parser.add_argument_group("Regularization")
    regularization_group.add_argument(
        "--features_l1",
        type=float,
        default=0.0,
        help="L1 regularization weight for features.",
    )
    regularization_group.add_argument(
        "--feature_coupling_l1",
        type=float,
        default=0.0,
        help="L1 regularization weight for feature coupling.",
    )
    regularization_group.add_argument(
        "--memory_manifold_reg",
        type=float,
        default=1e-3,
        help="Weight for memory manifold regularization.",
    )
    regularization_group.add_argument(
        "--memory_manifold_dims",
        type=int,
        default=2,
        help="Number of dimensions to regularize in the memory manifold.",
    )

    return parser.parse_args()


def single_experiment(args=None):
    if args is None:
        args = parse_args()
    else:
        args = ArgDict(args)

    if args.torch_num_threads is not None:
        torch.set_num_threads(args.torch_num_threads)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

    load_path = getattr(args, "load_path", None)
    new_save_dir = getattr(args, "new_save_dir", False)

    if args.save_path != "auto":
        save_path = args.save_path
    elif load_path is not None and not new_save_dir:
        load_parent = os.path.dirname(load_path)
        save_path = (
            os.path.dirname(load_parent)
            if os.path.basename(load_parent) == "model_checkpoints"
            else load_parent
        )
    else:
        save_path = f"./results/{datetime.now().strftime('%Y-%m-%d')}/single_exp_{datetime.now().strftime('%H-%M-%S')}_{args.dataset}_{args.model}"

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "plots", "training"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "plots", "eval"), exist_ok=True)

    # save setup and hyperparameters to json file with timestamp
    config_to_save = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
    with open(os.path.join(save_path, "experiment_config.json"), "w") as f:
        json.dump(config_to_save, f, indent=4)

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # print experiment configuration
    print("Starting experiment on dataset:", args.dataset)
    print("Using model:", args.model)
    print("On device:", args.device)

    # Load dataset
    # check if dataset is a file path or a known dataset name
    if os.path.isfile(args.dataset):
        dataset = load_dataset(args.dataset)
    else:
        if os.path.isfile(
            os.path.join(
                "data", args.dataset, f"{args.dataset}_dataset_standardized.npz"
            )
        ):
            dataset = load_dataset(
                os.path.join(
                    "data", args.dataset, f"{args.dataset}_dataset_standardized.npz"
                )
            )
        else:
            raise ValueError(
                f"Dataset '{args.dataset}' not found as file or in data directory (data/{args.dataset}/{args.dataset}_dataset_standardized.npz). You are in {os.getcwd()}."
            )
    X_train, _, _ = get_datasets(dataset)
    train_dataloader, test_id_dataloader, test_ood_dataloader = get_dataloader(
        dataset, batch_size=args.batch_size
    )
    cps_id = get_control_params(dataset, domain="id")
    cps_ood = get_control_params(dataset, domain="ood")

    # Initialize model (example with shPLRNN, adjust as needed)
    encoder = ConcatEncoder(obs_dim=X_train.shape[-1], latent_dim=args.latent_dim)
    decoder = ReadoutDecoder(latent_dim=args.latent_dim, obs_dim=X_train.shape[-1])

    if args.model.lower() == "shplrnn":
        model = shPLRNN(
            obs_dim=X_train.shape[-1],
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            encoder=encoder,
            decoder=decoder,
            num_train_subjects=X_train.shape[0],
            num_features=args.num_features,
            feature_splitting=args.feature_splitting,
            seed=args.seed,
        )
    elif args.model.lower() == "node":
        model = NODE(
            obs_dim=X_train.shape[-1],
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            encoder=encoder,
            decoder=decoder,
            num_train_subjects=X_train.shape[0],
            num_features=args.num_features,
            feature_splitting=args.feature_splitting,
            seed=args.seed,
            use_odeint=False,  # set to True to use odeint instead of RK4 for integration in NODE model
            node_dt=0.15,  # time step to use for integration in NODE model (only relevant if use_odeint=False)
        )
    else:
        raise ValueError(f"Model {args.model} not recognized.")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5
    )
    lr_scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.75, patience=40, min_lr=1e-6
        )
        if not args.no_lr_scheduler
        else None
    )

    start_epoch = 0
    skip_training = False

    regularizers = []
    if args.features_l1 > 0.0:
        regularizers.append(FeaturesL1Regularizer(weight=args.features_l1))
    if args.feature_coupling_l1 > 0.0:
        regularizers.append(
            FeatureCouplingL1Regularizer(weight=args.feature_coupling_l1)
        )
    if args.memory_manifold_reg > 0.0:
        regularizers.append(
            MemoryManifoldRegularizer(
                weight=args.memory_manifold_reg,
                memory_dims=args.memory_manifold_dims,
            )
        )

    if load_path is not None:
        checkpoint = torch.load(load_path, map_location=args.device, weights_only=True)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "epoch" in checkpoint:
                start_epoch = int(checkpoint["epoch"])
            else:
                match = re.search(
                    r"model_epoch_(\d+)\.pt$", os.path.basename(load_path)
                )
                if match is not None:
                    start_epoch = int(match.group(1))
        else:
            model.load_state_dict(checkpoint)
            if os.path.basename(load_path) == "final_model.pth":
                skip_training = True

        print(f"Loaded model state from {load_path}.")
        if skip_training:
            print("Detected final model file. Skipping training.")
        elif start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch + 1} of {args.epochs}.")

    if args.tf_alpha is not None and args.tf_alpha > 0.0:
        print(
            f"Using teacher forcing with initial alpha={args.tf_alpha} and decay of {args.tf_alpha_decay} per epoch."
        )

        def tf_alpha(e):
            return (
                args.tf_alpha - args.tf_alpha_final
            ) * args.tf_alpha_decay**e + args.tf_alpha_final

    else:
        print("Not using teacher forcing.")
        tf_alpha = None  # type: ignore

    if args.force_every is not None and args.force_every > 0:
        print(f"Forcing every {args.force_every} steps.")

        def force_every(e):
            if args.force_every_increase > 0:
                return args.force_every + (e // args.force_every_increase)
            else:
                return args.force_every

    else:
        print("Not using forcing.")
        force_every = None  # type: ignore

    # tf = SGTF(forcing_dim=X_train.shape[-1], encoder=encoder, decoder=decoder)
    tf = MGTF(forcing_dim=args.latent_dim, encoder=encoder, decoder=decoder)

    # Train model
    if not skip_training:
        train_model(
            model=model,
            train_dataloader=train_dataloader,
            test_id_dataloader=test_id_dataloader,
            epochs=args.epochs,
            start_epoch=start_epoch,
            num_batches_per_epoch=args.num_batches_per_epoch,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            steps_ahead=(
                eval(args.steps_ahead)
                if isinstance(args.steps_ahead, str)
                else args.steps_ahead
            ),
            tf=tf,
            tf_alpha=tf_alpha,
            force_every=force_every,
            feature_noise=(
                eval(args.feature_noise)
                if isinstance(args.feature_noise, str)
                else args.feature_noise
            ),
            regularizers=regularizers,
            device=torch.device(args.device),
            validate_every=args.validate_every,
            save_path=save_path,
            save_every=args.save_every,
            verbose=args.verbose,
            plot_every=args.plot_every,
            plot_path=os.path.join(save_path, "plots", "training"),
            seed=args.seed,
        )

    # save final model
    torch.save(model.state_dict(), os.path.join(save_path, "final_model.pth"))

    # plot features
    if model.feature_splitting:
        plt.scatter(
            range(model.num_train_subjects),
            model.features_pos[:, 0].cpu().detach().numpy(),
            c="blue",
            label="Positional Feature 1",
        )
        plt.twinx()
        plt.scatter(
            range(model.num_train_subjects),
            model.features_dyn[:, 0].cpu().detach().numpy(),
            c="orange",
            label="Dynamical Feature 1",
        )
    else:
        plt.scatter(
            range(model.num_train_subjects),
            model.features[:, 0].cpu().detach().numpy(),
            c="blue",
            label="Feature 1",
        )
        if model.num_features > 1:
            plt.twinx()
            plt.scatter(
                range(model.num_train_subjects),
                model.features[:, 1].cpu().detach().numpy(),
                c="orange",
                label="Feature 2",
            )
    plt.title("Learned Features")
    plt.xlabel("Subject Index")
    plt.legend()
    plt.savefig(os.path.join(save_path, "plots", "eval", "learned_features_id.png"))
    plt.close()

    # plot bifurcation diagram
    feature_values = (
        model.features.detach()
        if not model.feature_splitting
        else model.features_dyn.detach()
    )
    feature_values_pos = (
        model.features_pos.detach() if model.feature_splitting else None
    )
    bifurcation_result = model.bifurcation_diagram_points(
        feature_values=feature_values,
        feature_values_pos=feature_values_pos,
        num_ics=13,
        relevant_dim=0,
        rollout_steps=25_000,
        transient_steps=5_000,
        tf_warmup=tf,
        return_trajs=args.save_predicted_trajectories,
    )
    if args.save_predicted_trajectories:
        bif_points, pred_obs = bifurcation_result
    else:
        bif_points = cast(dict, bifurcation_result)
        pred_obs = None
    for key in bif_points:
        plt.scatter(
            [cps_id[key]] * len(bif_points[key]["extrema"]),
            bif_points[key]["extrema"],
            color="blue",
            s=10,
            label="extrema" if key == 0 else "",
        )
        plt.scatter(
            [cps_id[key]] * len(bif_points[key]["fixed_points"]),
            bif_points[key]["fixed_points"],
            color="red",
            s=10,
            label="Fixed points" if key == 0 else "",
        )
    plt.xlabel("Control Parameter")
    plt.ylabel("Observed extrema / fixed points")
    plt.title("Bifurcation diagram")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(save_path, "plots", "eval", "bifurcation_diagram_id.png"))
    plt.close()

    # extrapolate features to OOD control parameters
    feature_extrapolator = FeatureExtrapolation(model)
    feature_extrapolator.fit(cps_id, extrapolation_control_parameters=cps_ood)

    if model.feature_splitting:
        feature_values_dyn_extrap, feature_values_pos_extrap = (
            feature_extrapolator.predict(cps_ood)
        )
        for i in range(model.num_features):
            plt.scatter(
                cps_id,
                model.features_pos[:, i].cpu().detach().numpy(),
                c=f"C{i}",
                label=f"Positional Feature {i + 1}",
            )
            plt.scatter(
                cps_id,
                model.features_dyn[:, i].cpu().detach().numpy(),
                c=f"C{i}",
                label=f"Dynamical Feature {i + 1}",
            )
            plt.scatter(
                cps_ood,
                feature_values_pos_extrap[:, i].cpu().detach().numpy(),
                c=f"C{i}",
                marker="x",  # type: ignore
                label=f"Extrapolated Positional Feature {i + 1}",
            )
            plt.scatter(
                cps_ood,
                feature_values_dyn_extrap[:, i].cpu().detach().numpy(),
                c=f"C{i}",
                marker="x",  # type: ignore
                label=f"Extrapolated Dynamical Feature {i + 1}",
            )
        plt.title("Learned Features and Extrapolation")
        plt.xlabel("Control Parameter")
        plt.ylabel("Feature Value")
        plt.legend()
        plt.savefig(
            os.path.join(save_path, "plots", "eval", "feature_extrapolation.png")
        )
        plt.close()
    else:
        feature_values_extrap = feature_extrapolator.predict(cps_ood)
        for i in range(model.num_features):
            plt.scatter(
                cps_id,
                model.features[:, i].cpu().detach().numpy(),
                c=f"C{i}",
                label=f"Feature {i + 1}",
            )
            plt.scatter(
                cps_ood,
                feature_values_extrap[:, i].cpu().detach().numpy(),  # type: ignore
                c=f"C{i}",
                marker="x",  # type: ignore
                label=f"Extrapolated Feature {i + 1}",
            )
        plt.title("Learned Feature and Extrapolation")
        plt.xlabel("Control Parameter")
        plt.ylabel("Feature Value")
        plt.legend()
        plt.savefig(
            os.path.join(save_path, "plots", "eval", "feature_extrapolation.png")
        )
        plt.close()

    # bifurcation diagram for extrapolated features
    bifurcation_result_extrap = model.bifurcation_diagram_points(
        feature_values=(
            feature_values_dyn_extrap
            if model.feature_splitting
            else feature_values_extrap
        ),  # type: ignore
        feature_values_pos=(
            feature_values_pos_extrap if model.feature_splitting else None
        ),
        num_ics=13,
        relevant_dim=0,
        rollout_steps=25_000,
        transient_steps=5_000,
        return_trajs=args.save_predicted_trajectories,
        tf_warmup=tf,
    )
    if args.save_predicted_trajectories:
        bif_points_extrap, pred_obs_extrap = bifurcation_result_extrap
    else:
        bif_points_extrap = cast(dict, bifurcation_result_extrap)
        pred_obs_extrap = None

    # plot bifurcation diagram for extrapolated features
    for key in bif_points_extrap:
        plt.scatter(
            [cps_ood[key]] * len(bif_points_extrap[key]["extrema"]),
            bif_points_extrap[key]["extrema"],
            color="blue",
            s=10,
            label="extrema" if key == 0 else "",
        )
        plt.scatter(
            [cps_ood[key]] * len(bif_points_extrap[key]["fixed_points"]),
            bif_points_extrap[key]["fixed_points"],
            color="red",
            s=10,
            label="Fixed points" if key == 0 else "",
        )
    plt.xlabel("Control Parameter (extrapolated)")
    plt.ylabel("Observed extrema / fixed points")
    plt.title("Bifurcation diagram (extrapolated)")
    plt.legend(loc="upper right")
    plt.savefig(
        os.path.join(save_path, "plots", "eval", "bifurcation_diagram_extrapolated.png")
    )
    plt.close()

    # plot bifurcation diagram with both ID and OOD control parameters
    for key in bif_points:
        plt.scatter(
            [cps_id[key]] * len(bif_points[key]["extrema"]),
            bif_points[key]["extrema"],
            color="blue",
            s=10,
            label="extrema (ID)" if key == 0 else "",
        )
        plt.scatter(
            [cps_id[key]] * len(bif_points[key]["fixed_points"]),
            bif_points[key]["fixed_points"],
            color="red",
            s=10,
            label="Fixed points (ID)" if key == 0 else "",
        )
    for key in bif_points_extrap:
        plt.scatter(
            [cps_ood[key]] * len(bif_points_extrap[key]["extrema"]),
            bif_points_extrap[key]["extrema"],
            color="cyan",
            s=10,
            label="extrema (OOD)" if key == 0 else "",
        )
        plt.scatter(
            [cps_ood[key]] * len(bif_points_extrap[key]["fixed_points"]),
            bif_points_extrap[key]["fixed_points"],
            color="magenta",
            s=10,
            label="Fixed points (OOD)" if key == 0 else "",
        )
    plt.xlabel("Control Parameter")
    plt.ylabel("Observed extrema / fixed points")
    plt.title("Bifurcation diagram (ID and OOD)")
    plt.legend(loc="upper right")
    plt.savefig(
        os.path.join(save_path, "plots", "eval", "bifurcation_diagram_id_ood.png")
    )
    plt.close()

    # compute Wasserstein-1 distance between predicted and true distributions of extrema for ID and OOD control parameters
    # true extrema and fixed points can be obtained from dataset folder (bifurcation_data.npz)
    true_bifurcation_data = np.load(
        os.path.join("data", args.dataset, "bifurcation_data.npz")
    )
    w1_distances_id = _compute_wasserstein_distances(
        _combine_true_bifurcation_points(true_bifurcation_data, domain="id"),
        _combine_predicted_bifurcation_points(bif_points),
    )
    w1_distances_ood = _compute_wasserstein_distances(
        _combine_true_bifurcation_points(true_bifurcation_data, domain="ood"),
        _combine_predicted_bifurcation_points(bif_points_extrap),
    )

    # save Wasserstein distances to a file
    torch.save(
        {
            "w1_distances_id": w1_distances_id,
            "w1_distances_ood": w1_distances_ood,
        },
        os.path.join(save_path, "wasserstein_distances.pth"),
    )

    # save bifurcation points and optionally the full predicted trajectories
    bifurcation_data_to_save = {
        "bifurcation_points": bif_points,
        "bifurcation_points_extrap": bif_points_extrap,
    }
    if args.save_predicted_trajectories:
        bifurcation_data_to_save["predicted_trajectories"] = (
            _prepare_trajectories_for_save(
                pred_obs,
                stride=args.trajectory_save_stride,
                dtype=args.trajectory_save_dtype,
            )
        )
        bifurcation_data_to_save["predicted_trajectories_extrap"] = (
            _prepare_trajectories_for_save(
                pred_obs_extrap,
                stride=args.trajectory_save_stride,
                dtype=args.trajectory_save_dtype,
            )
        )
        bifurcation_data_to_save["predicted_trajectories_save_stride"] = (
            args.trajectory_save_stride
        )
        bifurcation_data_to_save["predicted_trajectories_save_dtype"] = (
            args.trajectory_save_dtype
        )
    torch.save(
        bifurcation_data_to_save,
        os.path.join(save_path, "bifurcation_data.pth"),
    )


if __name__ == "__main__":
    single_experiment()
