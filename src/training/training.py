import torch
import torch.nn as nn
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Union, Optional, Callable

# Add parent directory to path for imports
sys.path.insert(0, str(Path().resolve().parent))

from src.models.base_models import HierarchicalDSRModel
from .regularization import Regularizer
from .teacher_forcing import TeacherForcing

from data.dataset import DSRDataLoader


def train_model(
    model: HierarchicalDSRModel,
    train_dataloader: DSRDataLoader,
    test_id_dataloader: torch.utils.data.DataLoader,
    epochs: int = 100,
    start_epoch: int = 0,
    num_batches_per_epoch: Optional[int] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    steps_ahead: Union[int, Callable[[int], int]] = 10,
    tf: Optional[TeacherForcing] = None,
    tf_alpha: Optional[Union[float, Callable[[int], float]]] = None,
    force_every: Optional[Union[int, Callable[[int], int]]] = None,
    feature_noise: Optional[Union[float, Callable[[int], float]]] = None,
    device: torch.device = torch.device("cpu"),
    regularizers: Optional[list[Regularizer]] = None,
    validate_every: int = 5,
    validation_batches: int = 8,
    validation_steps: int = 200,
    plot_every: int = 10,
    plot_path: Optional[str] = None,
    save_every: int = 10,
    save_path: str = "./model_checkpoints",
    verbose: bool = True,
    seed: Optional[int] = None,
    return_best_model: bool = True,
    best_model_in_last_epochs: Union[int, float] = 0.2,
) -> None:
    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    # Create directory for saving model checkpoints if it doesn't exist
    Path(f"{save_path}/model_checkpoints").mkdir(parents=True, exist_ok=True)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)

    last_val_loss = torch.tensor(float("inf"), device=device)
    # check if best_model_in_last_epochs is a fraction and convert to number of epochs if so
    if (
        isinstance(best_model_in_last_epochs, float)
        and 0 < best_model_in_last_epochs < 1
    ):
        best_model_in_last_epochs = int(epochs * best_model_in_last_epochs)
    if not 0 < best_model_in_last_epochs < epochs:
        best_model_in_last_epochs = int(
            epochs * 0.2
        )  # default to last 20% of epochs if invalid value provided
    best_val_loss = torch.tensor(float("inf"), device=device)
    best_model_state_dict = None

    for e in range(start_epoch, epochs):
        epoch_loss = 0.0
        tf_alpha_value = tf_alpha(e) if callable(tf_alpha) else tf_alpha
        steps_ahead_value = steps_ahead(e) if callable(steps_ahead) else steps_ahead
        train_dataloader.set_seq_length(steps_ahead_value)
        force_every_value = force_every(e) if callable(force_every) else force_every
        for batch_idx, (true_seq, subject_idx) in enumerate(train_dataloader):
            if num_batches_per_epoch is not None and batch_idx >= num_batches_per_epoch:
                break
            optimizer.zero_grad()
            feature_values = (
                model.features[subject_idx.to(device)]
                if not model.feature_splitting
                else model.features_dyn[subject_idx.to(device)]
            )
            if feature_noise is not None:
                feature_values += torch.randn_like(feature_values) * (
                    feature_noise(e) if callable(feature_noise) else feature_noise  # type: ignore
                )
            feature_values_pos = (
                model.features_pos[subject_idx.to(device)]
                if model.feature_splitting
                else None
            )
            if feature_values_pos is not None and feature_noise is not None:
                feature_values_pos += torch.randn_like(feature_values_pos) * (
                    feature_noise(e) if callable(feature_noise) else feature_noise  # type: ignore
                )
            pred_seq = model(
                true_seq.to(device),
                feature_values=feature_values,
                feature_values_pos=feature_values_pos,
                tf=tf,
                tf_alpha=tf_alpha_value,
                force_every=force_every_value,
            )
            recon_loss = nn.MSELoss()(pred_seq, true_seq.to(device))
            reg_loss = torch.tensor(0.0, device=device)
            if regularizers is not None:
                for regularizer in regularizers:
                    reg_loss += regularizer(model)
            loss = recon_loss + reg_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        log_str = (
            f"Epoch {e + 1}/{epochs}, "
            f"Loss: {epoch_loss / ((batch_idx + 1) if num_batches_per_epoch is None else num_batches_per_epoch):.2e}"
            f" (Recon: {recon_loss.item():.2e}, Reg: {reg_loss.item():.2e}),"
            f" Steps ahead: {steps_ahead_value}, TF alpha: {tf_alpha_value:.4f}, Force every: {force_every_value},"
            f" LR: {optimizer.param_groups[0]['lr']:.2e},"
            f" Feature noise: {feature_noise(e) if callable(feature_noise) else feature_noise:.4f},"
            if feature_noise is not None
            else ""
        )
        with open(f"{save_path}/training_log.txt", "a") as log_file:
            log_file.write(log_str + "\n")
        if verbose:
            print(log_str)

        # Validation
        if (e + 1) % validate_every == 0:
            val_loss = torch.tensor(0.0, device=device)
            for batch_idx, (test_seq, test_subject_idx) in enumerate(
                test_id_dataloader
            ):
                if batch_idx >= validation_batches:
                    break
                test_seq = test_seq.to(device)
                test_subject_idx = test_subject_idx.to(device)
                with torch.no_grad():
                    feature_values = (
                        model.features[test_subject_idx]
                        if not model.feature_splitting
                        else model.features_dyn[test_subject_idx]
                    )
                    feature_values_pos = (
                        model.features_pos[test_subject_idx]
                        if model.feature_splitting
                        else None
                    )
                    pred_test_seq = model.forward(
                        test_seq[:, :validation_steps, :],
                        feature_values=feature_values,
                        feature_values_pos=feature_values_pos,
                        tf=tf,
                        tf_alpha=0.0,
                        force_every=999_999,  # effectively never force during validation
                    )
                    test_loss = nn.MSELoss()(
                        pred_test_seq, test_seq[:, :validation_steps, :]
                    )
                val_loss += test_loss.item()
            val_loss /= validation_batches
            last_val_loss = val_loss
            if (
                best_model_in_last_epochs > 0
                and e >= epochs - best_model_in_last_epochs
            ):
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state_dict = model.state_dict()
            log_str = f"Validation Loss after Epoch {e + 1}: {val_loss:.2e}"
            with open(f"{save_path}/training_log.txt", "a") as log_file:
                log_file.write(log_str + "\n")
            if verbose:
                print(log_str)
            if (e + 1) % plot_every == 0:
                plt.figure(figsize=(12, 6))
                plt.plot(test_seq[0, :validation_steps, 0].cpu(), label="True")
                plt.plot(
                    pred_test_seq[0, :validation_steps, 0].detach().cpu(),
                    label="Predicted",
                )
                data_min = test_seq[0, :validation_steps, 0].cpu().min()
                data_max = test_seq[0, :validation_steps, 0].cpu().max()
                plt.ylim(
                    data_min - 0.1 * (data_max - data_min),
                    data_max + 0.1 * (data_max - data_min),
                )
                plt.title(f"Validation Trajectory after Epoch {e + 1}")
                plt.legend()
                if plot_path is not None:
                    plt.savefig(f"{plot_path}/validation_trajectory_epoch_{e + 1}.png")
                else:
                    plt.show()
                plt.close()

        # Step the learning rate scheduler if provided using the last val loss for ReduceLROnPlateau, otherwise step normally
        if lr_scheduler is not None and isinstance(
            lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            lr_scheduler.step(last_val_loss)
        elif lr_scheduler is not None:
            lr_scheduler.step()

        # save model and optimizer state dicts
        if (e + 1) % save_every == 0:
            torch.save(
                {
                    "epoch": e + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                f"{save_path}/model_checkpoints/model_epoch_{e + 1}.pt",
            )

    # save best model if applicable
    if best_model_state_dict is not None:
        torch.save(
            {
                "epoch": epochs,
                "model": best_model_state_dict,
                "optimizer": optimizer.state_dict(),
            },
            f"{save_path}/model_checkpoints/best_model.pt",
        )
        if return_best_model:
            model.load_state_dict(best_model_state_dict)
