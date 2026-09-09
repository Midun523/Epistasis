"""DeepCOMBI: Explainable Artificial Intelligence / MLP Baseline for GWAS Interpretation."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class DeepCOMBIMLP(nn.Module):
    """Multi-Layer Perceptron for DeepCOMBI with gradient and LRP-style relevance scoring."""

    def __init__(self, num_snps: int = 1000, hidden_dims: List[int] = [128, 64], task: str = "classification"):
        super().__init__()
        self.num_snps = num_snps
        self.task = task

        layers = []
        in_dim = num_snps
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DeepCOMBIBaseline:
    """Wrapper for training DeepCOMBI MLP and extracting SNP relevance scores."""

    def __init__(self, num_snps: int = 1000, epochs: int = 30, lr: float = 1e-3, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.num_snps = num_snps
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.model = DeepCOMBIMLP(num_snps=num_snps).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray, batch_size: int = 64) -> "DeepCOMBIBaseline":
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch_x).squeeze(-1)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def score_features(self, X: np.ndarray) -> np.ndarray:
        """
        Computes input-gradient attribution (Saliency / Input * Gradient) across samples.
        """
        self.model.eval()
        tensor_x = torch.tensor(X, dtype=torch.float32, device=self.device, requires_grad=True)
        out = self.model(tensor_x).squeeze(-1)
        # Sum outputs to compute total gradient with respect to inputs
        grad_out = torch.ones_like(out)
        out.backward(gradient=grad_out)

        # Relevance = |Input * Gradient|
        grads = tensor_x.grad.detach().cpu().numpy()
        relevance = np.mean(np.abs(X * grads), axis=0)
        return relevance.astype(np.float32)
