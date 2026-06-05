import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from .base_models import HierarchicalDSRModel
from .encoder_decoder import Encoder, Decoder
from ..training.teacher_forcing import TeacherForcing


class shPLRNN(HierarchicalDSRModel):
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int,
        encoder: Encoder,
        decoder: Decoder,
        rank: Optional[int] = None,
        num_features: int = 1,
        num_train_subjects: int = 1,
        feature_splitting: bool = False,
        seed: Optional[int] = None,
    ):
        super(shPLRNN, self).__init__(
            obs_dim,
            latent_dim,
            encoder,
            decoder,
            rank,
            num_features,
            num_train_subjects,
            feature_splitting,
        )
        if seed is not None:
            torch.manual_seed(seed)

        # centered weights (theta_c)
        self.A = nn.Parameter(torch.ones(latent_dim)*0.95)
        self.W1 = nn.Parameter(torch.randn(latent_dim, hidden_dim) * 1e-1)
        self.W2 = nn.Parameter(torch.randn(hidden_dim, latent_dim) * 1e-1)
        self.h1 = nn.Parameter(torch.randn(latent_dim) * 1e-1)
        self.h2 = nn.Parameter(torch.randn(hidden_dim) * 1e-1)

        # feature-coupling weights (theta_v)
        if rank is None:
            self.A_v = nn.Parameter(torch.randn(num_features, latent_dim) * 1e-2)
            self.W1_v = nn.Parameter(torch.randn(num_features, latent_dim, hidden_dim) * 1e-2)
            self.W2_v = nn.Parameter(torch.randn(num_features, hidden_dim, latent_dim) * 1e-2)
            self.h1_v = nn.Parameter(torch.randn(num_features, latent_dim) * 1e-2)
            self.h2_v = nn.Parameter(torch.randn(num_features, hidden_dim) * 1e-2)
        else:
            self.W1_v_U = nn.Parameter(torch.randn(num_features, latent_dim, rank) * 1e-1)
            self.W1_v_V = nn.Parameter(torch.randn(num_features, rank, hidden_dim) * 1e-1)
            self.W2_v_U = nn.Parameter(torch.randn(num_features, hidden_dim, rank) * 1e-1)
            self.W2_v_V = nn.Parameter(torch.randn(num_features, rank, latent_dim) * 1e-1)
            self.h1_v = nn.Parameter(torch.randn(num_features, latent_dim) * 1e-2)
            self.h2_v = nn.Parameter(torch.randn(num_features, hidden_dim) * 1e-2)

            @property
            def W1_v(self):
                return torch.einsum("fij,fjk->fik", self.W1_v_U, self.W1_v_V)

            @property
            def W2_v(self):
                return torch.einsum("fij,fjk->fik", self.W2_v_U, self.W2_v_V)

        self.activation = nn.ReLU()

    def construct_parameters(
        self,
        feature_values: torch.Tensor,
        feature_values_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        feature_values: (batch_size, num_features)
            treated as dynamic features if feature_splitting is True
            and feature_values_pos is not None
        feature_values_pos: (batch_size, num_features)
            treated as positional features if feature_splitting is True
        """
        if self.feature_splitting:
            if feature_values_pos is None:
                raise ValueError(
                    "feature_values_pos must be provided when feature_splitting is True."
                )
            feature_values_dyn = feature_values
            A_v_diag = torch.eye(self.latent_dim).unsqueeze(0).to(device=self.A.device) * self.A_v.unsqueeze(1)
            A_v_diag = A_v_diag.to(device=self.A.device)
            A = torch.diag(self.A) + torch.einsum("bf,fij->bij", feature_values_dyn, A_v_diag)
            W1 = self.W1 + torch.einsum("bf,fij->bij", feature_values_dyn, self.W1_v)
            W2 = self.W2 + torch.einsum("bf,fij->bij", feature_values_dyn, self.W2_v)
            h2 = self.h2 + torch.einsum("bf,fj->bj", feature_values_dyn, self.h2_v)
            h1 = self.h1 + torch.einsum("bf,fj->bj", feature_values_pos, self.h1_v)
        else:
            if feature_values_pos is not None:
                # send warning that feature_values_pos is ignored
                print(
                    "Warning: feature_values_pos is ignored when feature_splitting is False."
                )
            A_v_diag = torch.eye(self.latent_dim).unsqueeze(0).to(device=self.A.device) * self.A_v.unsqueeze(1)
            A = torch.diag(self.A) + torch.einsum("bf,fij->bij", feature_values, A_v_diag)
            W1 = self.W1 + torch.einsum("bf,fij->bij", feature_values, self.W1_v)
            W2 = self.W2 + torch.einsum("bf,fij->bij", feature_values, self.W2_v)
            h1 = self.h1 + torch.einsum("bf,fj->bj", feature_values, self.h1_v)
            h2 = self.h2 + torch.einsum("bf,fj->bj", feature_values, self.h2_v)
        return A, W1, W2, h1, h2

    def get_centered_parameters(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        A = torch.diag(self.A)
        W1 = self.W1
        W2 = self.W2
        h1 = self.h1
        h2 = self.h2
        return A, W1, W2, h1, h2
    
    def get_feature_coupling_parameters(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.A_v, self.W1_v, self.W2_v, self.h1_v, self.h2_v
    
    def get_subject_specific_parameters(self) -> Tuple[torch.Tensor, ...]:
        return (self.features,) if not self.feature_splitting else (self.features_dyn, self.features_pos)

    def latent_step(
        self,
        z: torch.Tensor,
        weights: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        """
        z: (batch_size, latent_dim)
        weights: Tuple of (A, W1, W2, h1, h2) where:
            A: (batch_size, latent_dim, latent_dim)
            W1: (batch_size, latent_dim, hidden_dim)
            W2: (batch_size, hidden_dim, latent_dim)
            h1: (batch_size, latent_dim)
            h2: (batch_size, hidden_dim)

        z_next = Az + W1 * activation(W2 * z + h2) + h1
        """
        A, W1, W2, h1, h2 = weights
        Az = torch.einsum("bij,bj->bi", A, z)
        W2z = torch.einsum("bij,bj->bi", W2, z)
        W1_act = torch.einsum("bij,bj->bi", W1, self.activation(W2z + h2))
        z_next = Az + W1_act + h1
        return z_next