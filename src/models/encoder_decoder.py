import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

class Encoder(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int):
        super(Encoder, self).__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Encoder forward pass is not implemented yet.")

class LinearEncoder(Encoder):
    def __init__(self, obs_dim: int, latent_dim: int):
        super(LinearEncoder, self).__init__(obs_dim, latent_dim)
        self.encoder_matrix = nn.Parameter(torch.randn(obs_dim, latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.encoder_matrix

class ConcatEncoder(Encoder):
    def __init__(self, obs_dim: int, latent_dim: int):
        super(ConcatEncoder, self).__init__(obs_dim, latent_dim)
        self.encoder_matrix = nn.Parameter(torch.randn(obs_dim, latent_dim-obs_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, torch.einsum("...i,ij->...j", x, self.encoder_matrix)], dim=-1)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, obs_dim: int):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Decoder forward pass is not implemented yet.")

class LinearDecoder(Decoder):
    def __init__(self, latent_dim: int, obs_dim: int):
        super(LinearDecoder, self).__init__(latent_dim, obs_dim)
        self.decoder_matrix = nn.Parameter(torch.randn(latent_dim, obs_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.decoder_matrix

class ReadoutDecoder(Decoder):
    def __init__(self, latent_dim: int, obs_dim: int):
        super(ReadoutDecoder, self).__init__(latent_dim, obs_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :self.obs_dim]

