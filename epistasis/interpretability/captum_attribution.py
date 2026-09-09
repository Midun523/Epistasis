"""Captum-based model interpretability (Integrated Gradients and Layer Conductance)."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, LayerConductance

from epistasis.models.partitioned_transformer import EpistasisTransformer


class EmbeddingForwardWrapper(nn.Module):
    """Wraps model so that input to forward is continuous embedding tensor, facilitating Captum attribution."""

    def __init__(self, model: EpistasisTransformer):
        super().__init__()
        self.model = model

    def forward(self, embedded_tokens: torch.Tensor) -> torch.Tensor:
        # embedded_tokens: (B, N, D)
        B, N, D = embedded_tokens.shape
        locus_emb = self.model.locus_embedding[:, :N, :]
        tokens = embedded_tokens + locus_emb
        q = self.model.query_token.expand(B, -1, -1)

        for attn_layer, norm_a, ff_layer, norm_f in zip(
            self.model.layers, self.model.norm_layers, self.model.ff_layers, self.model.ff_norm_layers
        ):
            attn_out, _ = attn_layer(q, tokens, return_attention=False)
            q = norm_a(q + attn_out)
            q = norm_f(q + ff_layer(q))

        q = self.model.final_norm(q)
        return self.model.head(q.squeeze(1))  # (B, 1)


def compute_integrated_gradients(
    model: EpistasisTransformer,
    X: Union[np.ndarray, torch.Tensor],
    n_steps: int = 25,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Computes Integrated Gradients attribution using Captum over input SNP embeddings.

    Args:
        model: Trained EpistasisTransformer.
        X: Genotype matrix (N_samples, N_snps).
        n_steps: Riemann approximation steps for integral.
        device: Target torch device.

    Returns:
        1D array of shape (N_snps,) with mean attribution magnitude per SNP.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    if isinstance(X, np.ndarray):
        tensor_X = torch.tensor(X, dtype=torch.long, device=device)
    else:
        tensor_X = X.long().to(device)

    wrapper = EmbeddingForwardWrapper(model).to(device)
    ig = IntegratedGradients(wrapper)

    # Convert discrete input into embeddings
    embedded_inputs = model.snp_embedding(tensor_X)  # (B, N, D)
    baseline_inputs = torch.zeros_like(embedded_inputs)

    # Compute attributions in batches
    attributions = []
    batch_size = 32
    for i in range(0, len(tensor_X), batch_size):
        sub_inp = embedded_inputs[i : i + batch_size]
        sub_base = baseline_inputs[i : i + batch_size]
        attr = ig.attribute(sub_inp, baselines=sub_base, n_steps=n_steps)
        # L2 norm across embedding dimension D: (B, N)
        attr_norm = torch.norm(attr, p=2, dim=-1).detach().cpu().numpy()
        attributions.append(attr_norm)

    all_attr = np.concatenate(attributions, axis=0)  # (N_samples, N_snps)
    mean_attr = np.mean(all_attr, axis=0)  # (N_snps,)
    return mean_attr.astype(np.float32)
