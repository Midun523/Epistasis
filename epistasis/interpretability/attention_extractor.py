"""Attention map extractor for Transformer models in Epistasis detection."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from epistasis.models.partitioned_transformer import EpistasisTransformer


def extract_attention_scores(
    model: EpistasisTransformer,
    X: Union[np.ndarray, torch.Tensor],
    batch_size: int = 64,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Computes global attention importance scores across all SNPs over the dataset.

    Args:
        model: Trained EpistasisTransformer.
        X: Genotype matrix of shape (N_samples, N_snps).
        batch_size: Batch size for forward passes.
        device: Device to run evaluation on.

    Returns:
        1D numpy array of shape (N_snps,) with mean attention weights per SNP.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    if isinstance(X, np.ndarray):
        tensor_X = torch.tensor(X, dtype=torch.long)
    else:
        tensor_X = X.long()

    all_attentions = []
    n_samples = len(tensor_X)

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = tensor_X[i : i + batch_size].to(device)
            _, attn_map = model(batch, return_attention=True)  # (B, N_snps)
            all_attentions.append(attn_map.cpu().numpy())

    concat_attn = np.concatenate(all_attentions, axis=0)  # (N_samples, N_snps)
    mean_attn = np.mean(concat_attn, axis=0)  # (N_snps,)
    return mean_attn.astype(np.float32)
