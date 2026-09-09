"""Unit tests for the Partitioned Sparse Transformer model and components."""

import torch
import pytest
from epistasis.models.partitioned_transformer import PartitionedAttention, EpistasisTransformer


def test_partitioned_attention():
    B, N, D = 4, 100, 32
    num_heads = 4
    attn = PartitionedAttention(
        embed_dim=D,
        num_heads=num_heads,
        num_partitions=4,
        combination_size=2,
        sparsity_ratio=0.8,
    )

    q = torch.randn(B, 1, D)
    kv = torch.randn(B, N, D)

    out, attn_map = attn(q, kv, return_attention=True)

    assert out.shape == (B, 1, D)
    assert attn_map.shape == (B, N)
    assert not torch.isnan(out).any()
    assert not torch.isnan(attn_map).any()


def test_epistasis_transformer_classification():
    B, N = 8, 200
    model = EpistasisTransformer(
        num_snps=N,
        embed_dim=32,
        num_heads=4,
        num_partitions=4,
        combination_size=2,
        num_layers=2,
        task="classification",
    )

    x = torch.randint(0, 3, (B, N))
    logits, attn_scores = model(x, return_attention=True)

    assert logits.shape == (B, 1)
    assert attn_scores.shape == (B, N)

    # Backward gradient flow
    loss = logits.sum()
    loss.backward()

    assert model.query_token.grad is not None
    assert model.snp_embedding.weight.grad is not None


def test_epistasis_transformer_regression():
    B, N = 8, 150
    model = EpistasisTransformer(
        num_snps=N,
        embed_dim=32,
        num_heads=2,
        num_partitions=3,
        combination_size=2,
        num_layers=1,
        task="regression",
    )

    x = torch.randint(0, 3, (B, N))
    preds = model(x, return_attention=False)
    assert preds.shape == (B, 1)
