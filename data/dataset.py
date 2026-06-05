"""
Dataset module.

This module contains the dataset classes and functions for loading the data from
the .npz files and preparing it for training and evaluation, e.g. providing PyTorch Dataset and DataLoader classes.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple, Union

# files for data are stored in, e.g., data/lorenz63/lorenz63_dataset_standardized.npz

# loading data
def load_dataset(file_path: str) -> dict:
    """
    Load dataset from .npz file.
    keys are:
    - X_train_standardized: (num_subjects, num_trajs_per_subject, num_timesteps, obs_dim)
    - X_test_id_standardized: (num_subjects, num_trajs_per_subject, num_timesteps, obs_dim)
    - X_test_ood_standardized: (num_subjects, num_trajs_per_subject, num_timesteps, obs_dim)
    - cps_id: (num_subjects,)
    - cps_ood: (num_subjects,)
    - standardization: (2 x obs_dim, ) (mean and std for each dimension)
    """
    data = np.load(file_path)
    return {key: data[key] for key in data.keys()}

def get_datasets(dataset: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert the loaded dataset into PyTorch tensors.
    """
    X_train = torch.from_numpy(dataset['X_train_standardized']).float()
    X_test_id = torch.from_numpy(dataset['X_test_id_standardized']).float()
    X_test_ood = torch.from_numpy(dataset['X_test_ood_standardized']).float()
    return X_train, X_test_id, X_test_ood

def get_control_params(dataset: dict, domain: str = "id") -> torch.Tensor:
    """
    Get control parameters for in-domain or out-of-domain data.
    """
    if domain == "id":
        return torch.from_numpy(dataset['cps_id']).float()
    elif domain == "ood":
        return torch.from_numpy(dataset['cps_ood']).float()
    else:
        raise ValueError("Domain must be 'id' or 'ood'.")

# DSRTrainDataset class should be able to provide sequences of data from all subjects in each batch
# (ideally, otherwise go round robin across subjects across batches), length can be adjusted in
# every epoch, e.g. start with short sequences and increase length across epochs
class DSRTrainDataset(Dataset):
    def __init__(self, X_train: torch.Tensor, seq_length: int):
        """
        X_train: (num_subjects, num_trajs_per_subject, num_timesteps, obs_dim)
        seq_length: int
        """
        self.X_train = X_train
        self.seq_length = seq_length

    def set_seq_length(self, seq_length: int) -> None:
        """Update sequence length on the fly."""
        self.seq_length = seq_length

    def __len__(self) -> int:
        return self.X_train.shape[0] * self.X_train.shape[1] * (self.X_train.shape[2] - self.seq_length + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        num_subjects, num_trajs_per_subject, num_timesteps, obs_dim = self.X_train.shape
        subject_idx = idx // (num_trajs_per_subject * (num_timesteps - self.seq_length + 1))
        traj_idx = (idx // (num_timesteps - self.seq_length + 1)) % num_trajs_per_subject
        time_idx = idx % (num_timesteps - self.seq_length + 1)
        return self.X_train[subject_idx, traj_idx, time_idx : time_idx + self.seq_length, :], subject_idx
    
# DSRTestDataset can just be a wrapper around the test data tensors, 
# providing the full trajectories for evaluation
class DSRTestDataset(Dataset):
    def __init__(self, X_test: torch.Tensor):
        """
        X_test: (num_subjects, num_trajs_per_subject, num_timesteps, obs_dim)
        """
        self.X_test = X_test

    def __len__(self) -> int:
        return self.X_test.shape[0] * self.X_test.shape[1]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        num_subjects, num_trajs_per_subject, num_timesteps, obs_dim = self.X_test.shape
        subject_idx = idx // num_trajs_per_subject
        traj_idx = idx % num_trajs_per_subject
        return self.X_test[subject_idx, traj_idx, :, :], subject_idx
    
class DSRDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True):
        super(DSRDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle)

    def set_seq_length(self, seq_length: int) -> None:
        """Update sequence length on the fly (only for training dataloader)."""
        if isinstance(self.dataset, DSRTrainDataset):
            self.dataset.set_seq_length(seq_length)
    
def get_dataloader(dataset: dict, batch_size: int, initial_seq_length: int = 10) -> Tuple[DSRDataLoader, DataLoader, DataLoader]:
    """
    Get PyTorch DataLoaders for training and testing.
    """
    X_train, X_test_id, X_test_ood = get_datasets(dataset)
    train_dataset = DSRTrainDataset(X_train, seq_length=initial_seq_length)  # example sequence length
    train_dataloader = DSRDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_id_dataset = DSRTestDataset(X_test_id)
    test_ood_dataset = DSRTestDataset(X_test_ood)
    test_id_dataloader = DataLoader(test_id_dataset, batch_size=batch_size, shuffle=False)
    test_ood_dataloader = DataLoader(test_ood_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_id_dataloader, test_ood_dataloader
