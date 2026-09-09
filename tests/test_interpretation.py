"""Unit tests for interpretability and attribution methods."""

import numpy as np
import torch
from epistasis.models.partitioned_transformer import EpistasisTransformer
from epistasis.interpretability.attention_extractor import extract_attention_scores
from epistasis.interpretability.gradient_cam import compute_attentive_class_activation_tokens
from epistasis.interpretability.captum_attribution import compute_integrated_gradients


def test_interpretability_methods():
    N_samples, N_snps = 32, 50
    model = EpistasisTransformer(
        num_snps=N_snps,
        embed_dim=32,
        num_heads=2,
        num_partitions=2,
        combination_size=2,
        num_layers=1,
    )

    X = np.random.randint(0, 3, size=(N_samples, N_snps))

    # 1. Pure Attention
    attn = extract_attention_scores(model, X, batch_size=16)
    assert attn.shape == (N_snps,)
    assert not np.isnan(attn).any()

    # 2. ACAT (Attention + Gradients)
    combined, a_norm, g_norm = compute_attentive_class_activation_tokens(model, X, alpha=0.5, batch_size=16)
    assert combined.shape == (N_snps,)
    assert a_norm.shape == (N_snps,)
    assert g_norm.shape == (N_snps,)
    assert np.all(combined >= 0.0) and np.all(combined <= 1.0)

    # 3. Captum Integrated Gradients
    ig_attr = compute_integrated_gradients(model, X, n_steps=5)
    assert ig_attr.shape == (N_snps,)
    assert not np.isnan(ig_attr).any()
