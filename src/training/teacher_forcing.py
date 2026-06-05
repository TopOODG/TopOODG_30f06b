import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


class TeacherForcing(nn.Module):
    def __init__(self, forcing_dim: int):
        super(TeacherForcing, self).__init__()
        self.forcing_dim = forcing_dim

    def forward(
        self,
        true: torch.Tensor,
        pred: torch.Tensor,
        tf_alpha: Optional[float] = None,
        force_every: Optional[int] = None,
        current_step: int = 0,
    ) -> torch.Tensor:
        raise NotImplementedError("Teacher forcing is not implemented yet.")


class GeneralizedTeacherForcing(TeacherForcing):
    """
    Generalized Teacher Forcing.
    x_next = tf_alpha * x_true + (1 - tf_alpha) * x_pred
    """

    def __init__(self, forcing_dim: int):
        super(GeneralizedTeacherForcing, self).__init__(forcing_dim)

    def forward(
        self,
        true: torch.Tensor,
        pred: torch.Tensor,
        tf_alpha: Optional[float] = None,
        force_every: Optional[int] = None,
        current_step: int = 0,
    ) -> torch.Tensor:
        next = pred.clone()
        if tf_alpha is not None:
            next[..., : self.forcing_dim] = (
                tf_alpha * true[..., : self.forcing_dim]
                + (1 - tf_alpha) * pred[..., : self.forcing_dim]
            )
        else:
            print("Warning: tf_alpha is None, defaulting to no teacher forcing.")
        return next


class SparseTeacherForcing(TeacherForcing):
    """
    Sparse Teacher Forcing.
    x_next = x_true if current_step % force_every == 0 else x_pred
    """

    def __init__(self, forcing_dim: int):
        super(SparseTeacherForcing, self).__init__(forcing_dim)

    def forward(
        self,
        true: torch.Tensor,
        pred: torch.Tensor,
        tf_alpha: Optional[float] = None,
        force_every: Optional[int] = None,
        current_step: int = 0,
    ) -> torch.Tensor:
        next = pred.clone()
        if force_every is not None:
            next[..., : self.forcing_dim] = (
                true[..., : self.forcing_dim]
                if current_step % force_every == 0
                else pred[..., : self.forcing_dim]
            )
        else:
            print("Warning: force_every is None, defaulting to no teacher forcing.")
        return next


class SparseGeneralizedTeacherForcing(TeacherForcing):
    """
    Sparse Generalized Teacher Forcing.
    x_next = tf_alpha * x_true + (1 - tf_alpha) * x_pred if current_step % force_every == 0 else x_pred
    """

    def __init__(self, forcing_dim: int):
        super(SparseGeneralizedTeacherForcing, self).__init__(forcing_dim)

    def forward(
        self,
        true: torch.Tensor,
        pred: torch.Tensor,
        tf_alpha: Optional[float] = None,
        force_every: Optional[int] = None,
        current_step: int = 0,
    ) -> torch.Tensor:
        next = pred.clone()
        if tf_alpha is not None and force_every is not None:
            next[..., : self.forcing_dim] = (
                tf_alpha * true[..., : self.forcing_dim]
                + (1 - tf_alpha) * pred[..., : self.forcing_dim]
                if current_step % force_every == 0
                else pred[..., : self.forcing_dim]
            )
        else:
            print(
                "Warning: tf_alpha or force_every is None, defaulting to no teacher forcing."
            )
        return next


class ManifoldGeneralizedTeacherForcing(TeacherForcing):
    """
    Manifold Generalized Teacher Forcing.
    z_next = z_pred - tf_alpha * (encode(decode(z_pred)) - encode(x_true))
    """

    def __init__(self, forcing_dim: int, encoder: nn.Module, decoder: nn.Module):
        super(ManifoldGeneralizedTeacherForcing, self).__init__(forcing_dim)
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        true: torch.Tensor,
        pred: torch.Tensor,
        tf_alpha: Optional[float] = None,
        force_every: Optional[int] = None,
        current_step: int = 0,
    ) -> torch.Tensor:
        if force_every is None:
            force_every = 1  # Default to always force if not specified
        next = pred.clone()
        if tf_alpha is not None and force_every is not None:
            next[..., : self.forcing_dim] = (
                pred[..., : self.forcing_dim]
                - tf_alpha
                * (self.encoder(self.decoder(pred)) - self.encoder(true))[
                    ..., : self.forcing_dim
                ]
                if current_step % force_every == 0
                else pred[..., : self.forcing_dim]
            )  # i.e. force the forcing dimensions if it's the right step, otherwise keep the prediction as is
        else:
            print("Warning: tf_alpha is None, defaulting to no teacher forcing.")
        return next
