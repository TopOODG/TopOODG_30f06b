import torch
import torch.nn as nn
import numpy as np

def wasserstein_distance_1d(predicted: torch.Tensor, true: torch.Tensor) -> float:
    """
    Compute the Wasserstein-1 distance between two 1D distributions represented as tensors, potentially of different lengths.
    """
    predicted = torch.as_tensor(predicted, dtype=torch.float32).flatten()
    true = torch.as_tensor(true, dtype=torch.float32).flatten()

    if predicted.numel() == 0 and true.numel() == 0:
        return 0.0
    if predicted.numel() == 0:
        predicted = torch.zeros(1, dtype=true.dtype, device=true.device)
    if true.numel() == 0:
        true = torch.zeros(1, dtype=predicted.dtype, device=predicted.device)

    # Interpolate both distributions to a common set of points (e.g., using linear interpolation)
    # use max length of the two distributions as the number of points for interpolation
    num_points = max(predicted.shape[0], true.shape[0])
    predicted_sorted, _ = torch.sort(predicted)
    true_sorted, _ = torch.sort(true)
    predicted_interp = nn.functional.interpolate(predicted_sorted.view(1, 1, -1), size=num_points, mode='linear', align_corners=False).view(-1)
    true_interp = nn.functional.interpolate(true_sorted.view(1, 1, -1), size=num_points, mode='linear', align_corners=False).view(-1)

    # Compute the Wasserstein-1 distance as the average absolute difference between the two interpolated distributions
    distance = torch.mean(torch.abs(predicted_interp - true_interp)).item()
    return distance