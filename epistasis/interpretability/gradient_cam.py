"""Attentive Class Activation Tokens: Combined Attention and Gradient Interpretation."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from epistasis.models.partitioned_transformer import EpistasisTransformer


def min_max_normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalizes an array to [0, 1]."""
    min_v = np.min(arr)
    max_v = np.max(arr)
    if max_v - min_v < eps:
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)


def compute_attentive_class_activation_tokens(
    model: EpistasisTransformer,
    X: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.5,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Attentive Class Activation Tokens (ACAT) combining attention scores and token gradients.

    Args:
        model: Trained EpistasisTransformer.
        X: Genotype matrix (N_samples, N_snps).
        alpha: Weight balancing attention vs gradient scores (0.5 = equal contribution).
        batch_size: Evaluation batch size.
        device: Torch device.

    Returns:
        (combined_scores, attention_scores, gradient_scores) each of shape (N_snps,).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    if isinstance(X, np.ndarray):
        tensor_X = torch.tensor(X, dtype=torch.long)
    else:
        tensor_X = X.long()

    n_samples, n_snps = tensor_X.shape
    all_attentions = []
    all_gradients = []

    for i in range(0, n_samples, batch_size):
        batch = tensor_X[i : i + batch_size].to(device)
        B = batch.size(0)

        # 1. Forward pass with attention extraction
        # To compute token gradients, embed the tokens and enable gradient tracking on embeddings
        embedded_tokens = model.snp_embedding(batch)  # (B, N, D)
        embedded_tokens.retain_grad()

        # Add locus embedding
        locus_emb = model.locus_embedding[:, :n_snps, :]
        tokens_with_pos = embedded_tokens + locus_emb

        # Expand query token
        q = model.query_token.expand(B, -1, -1)

        last_attention = None
        for layer_idx, (attn_layer, norm_a, ff_layer, norm_f) in enumerate(
            zip(model.layers, model.norm_layers, model.ff_layers, model.ff_norm_layers)
        ):
            is_last = (layer_idx == len(model.layers) - 1)
            attn_out, attn_map = attn_layer(q, tokens_with_pos, return_attention=is_last)
            q = norm_a(q + attn_out)
            q = norm_f(q + ff_layer(q))
            if is_last:
                last_attention = attn_map

        q = model.final_norm(q)
        logits = model.head(q.squeeze(1))  # (B, 1)

        # 2. Backward pass for gradients
        model.zero_grad()
        # Backpropagate predicted score
        logits.backward(gradient=torch.ones_like(logits))

        grad_embeddings = embedded_tokens.grad  # (B, N, D)
        if grad_embeddings is not None:
            # L2 norm across embedding dimension D: (B, N)
            grad_norm = torch.norm(grad_embeddings, p=2, dim=-1).detach().cpu().numpy()
        else:
            grad_norm = np.zeros((B, n_snps), dtype=np.float32)

        all_attentions.append(last_attention.detach().cpu().numpy())
        all_gradients.append(grad_norm)

    # Average across all samples
    concat_attn = np.concatenate(all_attentions, axis=0)  # (N_samples, N_snps)
    concat_grad = np.concatenate(all_gradients, axis=0)   # (N_samples, N_snps)

    mean_attn = np.mean(concat_attn, axis=0)
    mean_grad = np.mean(concat_grad, axis=0)

    # Min-max scale both signals to [0, 1]
    norm_attn = min_max_normalize(mean_attn)
    norm_grad = min_max_normalize(mean_grad)

    # Element-wise combination
    combined_scores = alpha * norm_attn + (1.0 - alpha) * norm_grad

    return combined_scores.astype(np.float32), norm_attn.astype(np.float32), norm_grad.astype(np.float32)
