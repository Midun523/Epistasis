"""PyTorch Dataset, splits, and DataLoaders for Epistasis data."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class EpistasisDataset(Dataset):
    """PyTorch Dataset for SNP genotypes and phenotypic labels."""

    def __init__(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        encoding: str = "categorical",  # 'categorical' (int64 for embeddings) or 'additive' (float32)
    ):
        """
        Args:
            X: Genotypes array of shape (N, S) with values in {0, 1, 2}.
            y: Labels array of shape (N,) (binary or continuous).
            encoding: 'categorical' (integers 0, 1, 2 for Embedding) or 'additive' (float32).
        """
        if isinstance(X, np.ndarray):
            if encoding == "categorical":
                self.X = torch.tensor(X, dtype=torch.long)
            else:
                self.X = torch.tensor(X, dtype=torch.float32)
        else:
            self.X = X.long() if encoding == "categorical" else X.float()

        if isinstance(y, np.ndarray):
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = y.float()

        self.encoding = encoding

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    splits: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    random_state: Optional[int] = None,
    stratify: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Splits data into train, validation, and test subsets.

    Args:
        X: Genotype matrix (N, S).
        y: Labels (N,).
        splits: (train_ratio, val_ratio, test_ratio) summing to 1.0.
        random_state: Seed.
        stratify: Whether to preserve class balance (for classification).
    """
    assert abs(sum(splits) - 1.0) < 1e-5, f"Splits must sum to 1.0, got {splits}"
    n_samples = len(y)
    rng = np.random.RandomState(random_state)

    is_binary = np.all(np.isin(y, [0, 1])) and len(np.unique(y)) <= 2

    if stratify and is_binary:
        cases_idx = np.where(y == 1)[0]
        ctrls_idx = np.where(y == 0)[0]
        rng.shuffle(cases_idx)
        rng.shuffle(ctrls_idx)

        n_train_case = int(len(cases_idx) * splits[0])
        n_val_case = int(len(cases_idx) * splits[1])

        n_train_ctrl = int(len(ctrls_idx) * splits[0])
        n_val_ctrl = int(len(ctrls_idx) * splits[1])

        train_idx = np.concatenate([cases_idx[:n_train_case], ctrls_idx[:n_train_ctrl]])
        val_idx = np.concatenate([cases_idx[n_train_case : n_train_case + n_val_case], ctrls_idx[n_train_ctrl : n_train_ctrl + n_val_ctrl]])
        test_idx = np.concatenate([cases_idx[n_train_case + n_val_case :], ctrls_idx[n_train_ctrl + n_val_ctrl :]])

        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)
    else:
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        n_train = int(n_samples * splits[0])
        n_val = int(n_samples * splits[1])

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

    return {
        "train": (X[train_idx], y[train_idx]),
        "val": (X[val_idx], y[val_idx]),
        "test": (X[test_idx], y[test_idx]),
    }


def create_dataloaders(
    data_dict: Dict[str, Union[np.ndarray, List[int], Dict]],
    batch_size: int = 64,
    splits: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    encoding: str = "categorical",
    random_state: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Creates PyTorch DataLoaders for train, validation, and test partitions.
    """
    X = data_dict["X"]
    y = data_dict["y"]

    partitioned_data = split_data(X, y, splits=splits, random_state=random_state)

    train_ds = EpistasisDataset(*partitioned_data["train"], encoding=encoding)
    val_ds = EpistasisDataset(*partitioned_data["val"], encoding=encoding)
    test_ds = EpistasisDataset(*partitioned_data["test"], encoding=encoding)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, partitioned_data
