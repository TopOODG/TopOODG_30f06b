import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Union, Optional, Tuple, Callable

# Add parent directory to path for imports
sys.path.insert(0, str(Path().resolve().parent))

from src.models.base_models import HierarchicalDSRModel
from src.models.shPLRNN import shPLRNN
# from src.models.node import Node

class Regularizer:
    def __init__(self, weight: float = 1e-2):
        self.weight = weight

    def __call__(self, model: HierarchicalDSRModel, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Regularizer must implement the __call__ method.")

class FeatureCouplingL1Regularizer(Regularizer):
    def __call__(self, model: HierarchicalDSRModel, *args, **kwargs) -> torch.Tensor:
        l1_norm = torch.tensor(0.0, device=next(model.parameters()).device)
        feature_coupling_params = model.get_feature_coupling_parameters()
        for param in feature_coupling_params:
            l1_norm += torch.sum(torch.abs(param)) / param.numel()  # Normalize by number of elements
        return self.weight * l1_norm

class FeaturesL1Regularizer(Regularizer):
    def __call__(self, model: HierarchicalDSRModel, *args, **kwargs) -> torch.Tensor:
        l1_norm = torch.tensor(0.0, device=next(model.parameters()).device)
        feature_tuple = model.get_subject_specific_parameters()
        for param in feature_tuple:
            l1_norm += torch.sum(torch.abs(param)) / param.numel()  # Normalize by number of elements
        return self.weight * l1_norm

class MemoryManifoldRegularizer(Regularizer):
    def __init__(self, weight: float = 1e-2, memory_dims: int = 2):
        super().__init__(weight)
        self.memory_dims = memory_dims

    def __call__(self, model: HierarchicalDSRModel, *args, **kwargs) -> torch.Tensor:
        manifold_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        if isinstance(model, shPLRNN):
            A = model.A[:self.memory_dims]  # Only regularize the first few dimensions of A
            manifold_loss += torch.sum((A - 0.99) ** 2) / A.numel()  # Normalize by number of elements
        W1, W2 = model.W1, model.W2
        W1W2 = torch.einsum("lh,hi->li", W1, W2)  # (latent_dim, latent_dim)
        W1W2[self.memory_dims:, self.memory_dims:] = 0.0  # Only regularize the first few dimensions of the manifold
        manifold_loss += torch.sum(W1W2 ** 2) / W1W2.numel()  # Normalize by number of elements
        return self.weight * manifold_loss