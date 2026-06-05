import torch
import torch.nn as nn
from torchdiffeq import odeint  # type: ignore
from typing import List, Optional, Tuple, Union
from .base_models import HierarchicalDSRModel
from .encoder_decoder import Encoder, Decoder
from ..training.teacher_forcing import TeacherForcing


class NODE(HierarchicalDSRModel):
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
        use_odeint: bool = False,
        node_dt: float = 0.15,
        seed: Optional[int] = None,
    ):
        """
        Neural Ordinary Differential Equation (NODE) model for learning continuous-time dynamics.
        """
        super(NODE, self).__init__(
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
        self.W1 = nn.Parameter(torch.randn(latent_dim, hidden_dim) * 1e-1)
        self.W2 = nn.Parameter(torch.randn(hidden_dim, latent_dim) * 1e-1)
        self.h1 = nn.Parameter(torch.randn(latent_dim) * 1e-1)
        self.h2 = nn.Parameter(torch.randn(hidden_dim) * 1e-1)

        # feature-coupling weights (theta_v)
        if rank is None:
            self.W1_v = nn.Parameter(
                torch.randn(num_features, latent_dim, hidden_dim) * 1e-2
            )
            self.W2_v = nn.Parameter(
                torch.randn(num_features, hidden_dim, latent_dim) * 1e-2
            )
            self.h1_v = nn.Parameter(torch.randn(num_features, latent_dim) * 1e-2)
            self.h2_v = nn.Parameter(torch.randn(num_features, hidden_dim) * 1e-2)
        else:
            self.W1_v_U = nn.Parameter(
                torch.randn(num_features, latent_dim, rank) * 1e-1
            )
            self.W1_v_V = nn.Parameter(
                torch.randn(num_features, rank, hidden_dim) * 1e-1
            )
            self.W2_v_U = nn.Parameter(
                torch.randn(num_features, hidden_dim, rank) * 1e-1
            )
            self.W2_v_V = nn.Parameter(
                torch.randn(num_features, rank, latent_dim) * 1e-1
            )
            self.h1_v = nn.Parameter(torch.randn(num_features, latent_dim) * 1e-2)
            self.h2_v = nn.Parameter(torch.randn(num_features, hidden_dim) * 1e-2)

            @property
            def W1_v(self):
                return torch.einsum("fij,fjk->fik", self.W1_v_U, self.W1_v_V)

            @property
            def W2_v(self):
                return torch.einsum("fij,fjk->fik", self.W2_v_U, self.W2_v_V)

        self.activation = nn.Tanh()

        self.use_odeint = use_odeint
        self.node_dt = node_dt

    def construct_parameters(
        self,
        feature_values: torch.Tensor,
        feature_values_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            W1 = self.W1 + torch.einsum("bf,fij->bij", feature_values, self.W1_v)
            W2 = self.W2 + torch.einsum("bf,fij->bij", feature_values, self.W2_v)
            h1 = self.h1 + torch.einsum("bf,fj->bj", feature_values, self.h1_v)
            h2 = self.h2 + torch.einsum("bf,fj->bj", feature_values, self.h2_v)
        return W1, W2, h1, h2

    def get_centered_parameters(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        W1 = self.W1
        W2 = self.W2
        h1 = self.h1
        h2 = self.h2
        return W1, W2, h1, h2

    def get_feature_coupling_parameters(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.W1_v, self.W2_v, self.h1_v, self.h2_v

    def get_subject_specific_parameters(self) -> Tuple[torch.Tensor, ...]:
        return (
            (self.features,)
            if not self.feature_splitting
            else (self.features_dyn, self.features_pos)
        )

    def z_dot(
        self,
        z: torch.Tensor,
        weights: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        z: (batch_size, latent_dim)
        weights: Tuple of (W1, W2, h1, h2) where:
            W1: (batch_size, latent_dim, hidden_dim)
            W2: (batch_size, hidden_dim, latent_dim)
            h1: (batch_size, latent_dim)
            h2: (batch_size, hidden_dim)

        z_dot = W1 @ activation(W2 @ z + h2) + h1
        """
        W1, W2, h1, h2 = weights
        W2z = torch.einsum("bij,bj->bi", W2, z)
        W1_act = torch.einsum("bij,bj->bi", W1, self.activation(W2z + h2))
        z_dot = W1_act + h1
        return z_dot

    def rk4_step(
        self,
        z: torch.Tensor,
        weights: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Perform a single Runge-Kutta 4th order (RK4) step to compute the next latent state.
        """
        k1 = self.z_dot(z, weights)
        k2 = self.z_dot(z + 0.5 * dt * k1, weights)
        k3 = self.z_dot(z + 0.5 * dt * k2, weights)
        k4 = self.z_dot(z + dt * k3, weights)
        z_next = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return z_next

    def latent_step(
        self,
        z: torch.Tensor,
        weights: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        z: (batch_size, latent_dim)
        weights: Tuple of (W1, W2, h1, h2) where:
            W1: (batch_size, latent_dim, hidden_dim)
            W2: (batch_size, hidden_dim, latent_dim)
            h1: (batch_size, latent_dim)
            h2: (batch_size, hidden_dim)

        z_next = integrate(z_dot, t) where f is defined as:
        f(z) = W1 @ activation(W2 @ z + h2) + h1
        """
        W1, W2, h1, h2 = weights
        dz_dt = lambda t, z_: self.z_dot(z_, (W1, W2, h1, h2))
        if self.use_odeint:
            z_next = odeint(
                dz_dt,
                z,
                torch.tensor([0, self.node_dt], device=z.device, dtype=torch.float32),
                rtol=1e-2,
                atol=1e-4,
            )[-1]
        else:
            z_next = self.rk4_step(z, (W1, W2, h1, h2), dt=self.node_dt)
        return z_next  # type: ignore
