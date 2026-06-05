import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union
from .encoder_decoder import Encoder, Decoder
from ..training.teacher_forcing import TeacherForcing


class DSRModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        encoder: Encoder,
        decoder: Decoder,
    ):
        super(DSRModel, self).__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x_obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_obs)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def latent_step(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Latent step is not implemented yet.")

    def forward(
        self,
        true_traj: torch.Tensor,
        tf: Optional[TeacherForcing] = None,
        tf_alpha: Optional[float] = None,
        force_every: Optional[int] = None,
        return_latent: bool = False,
    ) -> torch.Tensor:
        """
        true_traj: (batch_size, num_steps, obs_dim)
        tf_alpha: float
        force_every: int
        """
        true_latent = self.encode(true_traj[..., : self.obs_dim])
        pred_latent = torch.zeros_like(true_latent)
        z = true_latent[:, 0, :]
        pred_latent[:, 0, :] = z
        for t in range(1, true_traj.shape[1]):
            z = self.latent_step(z)
            if tf is not None:
                z = tf(true_traj[..., t, : self.obs_dim], z, tf_alpha, force_every, t)
            pred_latent[:, t, :] = z
        return self.decode(pred_latent) if not return_latent else pred_latent

    def free_rollout(
        self,
        true_warmup: torch.Tensor,
        steps_ahead: int,
        tf_warmup: Optional[TeacherForcing] = None,
        return_latent: bool = False,
    ) -> torch.Tensor:
        """
        true_warmup: (batch_size, warmup_steps, obs_dim)
        steps_ahead: int
        """
        true_latent_warmup = self.encode(true_warmup[..., : self.obs_dim])
        z = true_latent_warmup[:, 0, :]
        for t in range(true_warmup.shape[1]):
            z = self.latent_step(z)
            if tf_warmup is not None:
                z = tf_warmup(true_warmup[..., t, : self.obs_dim], z, 1.0, 1, t)
        pred_latent = torch.zeros(true_warmup.shape[0], steps_ahead, self.obs_dim)
        for t in range(steps_ahead):
            z = self.latent_step(z)
            pred_latent[..., t, :] = z
        if return_latent:
            return pred_latent
        else:
            return self.decode(pred_latent)


class HierarchicalDSRModel(DSRModel):
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        encoder: Encoder,
        decoder: Decoder,
        rank: Optional[int] = None,
        num_features: int = 1,
        num_train_subjects: int = 1,
        feature_splitting: bool = False,
    ):
        super(HierarchicalDSRModel, self).__init__(
            obs_dim, latent_dim, encoder, decoder
        )
        self.rank = rank
        self.num_features = num_features
        self.num_train_subjects = num_train_subjects
        self.feature_splitting = feature_splitting
        if not feature_splitting:
            # repeat random features for each training subject
            self.features = nn.Parameter(
                torch.randn(num_features).repeat(num_train_subjects, 1) * 1e-2
            )
        else:
            self.features_dyn = nn.Parameter(
                torch.randn(num_features).repeat(num_train_subjects, 1) * 1e-2
            )
            self.features_pos = nn.Parameter(
                torch.randn(num_features).repeat(num_train_subjects, 1) * 1e-2
            )

    def construct_parameters(
        self,
        feature_values: torch.Tensor,
        feature_values_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        feature_values: (batch_size, num_features)
            treated as dynamic features if feature_splitting is True
            and feature_values_pos is not None
        feature_values_pos: (batch_size, num_features)
            treated as positional features if feature_splitting is True
        """
        raise NotImplementedError("Construct parameters is not implemented yet.")

    def get_centered_parameters(self) -> Tuple[torch.Tensor, ...]:
        """
        Get the centered parameters (theta_c) that are shared across features and subjects.
        """
        raise NotImplementedError("Get centered parameters is not implemented yet.")

    def get_feature_coupling_parameters(self) -> Tuple[torch.Tensor, ...]:
        """
        Get the feature-coupling parameters (theta_v) that are modulated by the features, but shared across subjects.
        """
        raise NotImplementedError(
            "Get feature-coupling parameters is not implemented yet."
        )

    def get_subject_specific_parameters(self) -> Tuple[torch.Tensor, ...]:
        """
        Get the subject-specific parameters (features) that are specific to each subject.
        """
        raise NotImplementedError(
            "Get subject-specific parameters is not implemented yet."
        )

    def latent_step(
        self, z: torch.Tensor, weights: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """
        z: (batch_size, latent_dim)
        weights: tuple of parameters constructed from feature values, to be used in the latent step
        """
        raise NotImplementedError("Latent step is not implemented yet.")

    def forward(
        self,
        true_traj: torch.Tensor,
        feature_values: torch.Tensor,
        feature_values_pos: Optional[torch.Tensor] = None,
        tf: Optional[TeacherForcing] = None,
        tf_alpha: Optional[float] = None,
        force_every: Optional[int] = None,
        return_latent: bool = False,
    ) -> torch.Tensor:
        """
        true_traj: (batch_size, num_steps, obs_dim)
        feature_values: (batch_size, num_features)
            treated as dynamic features if feature_splitting is True
            and feature_values_pos is not None
        feature_values_pos: (batch_size, num_features)
            treated as positional features if feature_splitting is True
        tf_alpha: Optional[float]
        force_every: Optional[int]
        return_latent: bool
        """
        weights = self.construct_parameters(feature_values, feature_values_pos)
        true_latent = self.encode(true_traj[..., : self.obs_dim])
        pred_latent = torch.zeros_like(true_latent)
        z = true_latent[:, 0, :]
        pred_latent[:, 0, :] = z
        for t in range(1, true_traj.shape[1]):
            z = self.latent_step(z, weights)
            if tf is not None:
                z = tf(true_traj[..., t, : self.obs_dim], z, tf_alpha, force_every, t)
            pred_latent[:, t, :] = z
        if return_latent:
            return pred_latent
        else:
            return self.decode(pred_latent)

    @torch.inference_mode()
    def free_rollout(
        self,
        warmup: torch.Tensor,
        steps_ahead: int,
        feature_values: torch.Tensor,
        feature_values_pos: Optional[torch.Tensor] = None,
        tf_warmup: Optional[TeacherForcing] = None,
        return_latent: bool = False,
    ) -> torch.Tensor:
        """
        warmup: (batch_size, warmup_steps, obs_dim)
        steps_ahead: int
        feature_values: (batch_size, num_features)
            treated as dynamic features if feature_splitting is True
            and feature_values_pos is not None
        feature_values_pos: (batch_size, num_features)
            treated as positional features if feature_splitting is True
        return_latent: bool
        """
        warmup_latent = self.encode(warmup[..., : self.obs_dim])
        weights = self.construct_parameters(feature_values, feature_values_pos)
        z = warmup_latent[:, 0, :]
        for t in range(1, warmup.shape[1]):
            z = self.latent_step(z, weights)
            if tf_warmup is not None:
                z = tf_warmup(warmup[..., t, : self.obs_dim], z, 1.0, 1, t)
        pred_latent = torch.zeros(warmup.shape[0], steps_ahead, self.latent_dim)
        for t in range(steps_ahead):
            z = self.latent_step(z, weights)
            pred_latent[..., t, :] = z
        if return_latent:
            return pred_latent
        else:
            return self.decode(pred_latent)

    @torch.inference_mode()
    def bifurcation_diagram_points(
        self,
        feature_values: torch.Tensor,
        feature_values_pos: Optional[torch.Tensor] = None,
        num_ics: int = 32,
        relevant_dim: int = 0,
        rollout_steps: int = 12_000,
        transient_steps: int = 2_000,
        tf_warmup: Optional[TeacherForcing] = None,
        return_trajs: bool = False,
    ) -> Union[dict, Tuple[dict, torch.Tensor]]:
        """
        Produce list of e.g. extrema+fixed points in specified dimension
        for a given feature value.

        feature_values: (batch_size, num_features)
            treated as dynamic features if feature_splitting is True
            and feature_values_pos is not None
        feature_values_pos: (batch_size, num_features)
            treated as positional features if feature_splitting is True
        num_ics: int
        rollout_steps: int
        transient_steps: int
        """
        # generate long free rollouts for given feature values using random initial conditions
        rand_ics = (
            torch.randn(
                num_ics * feature_values.shape[0],
                self.obs_dim,
                device=next(self.parameters()).device,
            )
            * 2.0
        )
        pred_obs = self.free_rollout(
            warmup=rand_ics.unsqueeze(1),
            steps_ahead=rollout_steps,
            feature_values=feature_values.repeat(num_ics, 1),
            feature_values_pos=(
                feature_values_pos.repeat(num_ics, 1)
                if feature_values_pos is not None
                else None
            ),
            tf_warmup=tf_warmup,
            return_latent=False,
        )

        # extract points of interest from the rollouts (e.g. extrema, fixed points)
        def find_extrema(x):
            return ((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])) | (
                (x[1:-1] < x[:-2]) & (x[1:-1] < x[2:])
            )

        bif_diag_points = {}
        for i in range(feature_values.shape[0]):
            peaks = np.array([])
            fps = []
            for j in range(num_ics):
                peaks_j = find_extrema(
                    pred_obs[
                        i + j * feature_values.shape[0], transient_steps:, relevant_dim
                    ]
                )
                # concatenate and add to list of peaks
                peaks = np.concatenate(
                    (
                        peaks,
                        pred_obs[
                            i + j * feature_values.shape[0],
                            transient_steps + 1 : -1,
                            relevant_dim,
                        ][peaks_j]
                        .cpu()
                        .numpy(),
                    )
                )
                if peaks_j.sum() == 0:
                    # if no peaks found, check for fixed points by looking at the tail
                    tail = pred_obs[
                        i + j * feature_values.shape[0], -100:, relevant_dim
                    ]
                    if torch.all(torch.isclose(tail, tail[0], atol=1e-2)):
                        fps.append(tail[0])
            peaks = torch.tensor(np.array(peaks)).view(-1)
            fps = torch.tensor(np.array(fps)).view(-1)
            bif_diag_points[i] = {
                "extrema": peaks,
                "fixed_points": fps,
            }
        if not return_trajs:
            return bif_diag_points
        else:
            return bif_diag_points, pred_obs
