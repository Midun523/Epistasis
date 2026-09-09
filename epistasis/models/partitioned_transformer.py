"""PyTorch implementation of the Single-Query Partitioned Sparse Transformer for Epistasis Detection."""

from __future__ import annotations
import itertools
import math
from typing import Dict, List, Literal, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

TaskType = Literal["classification", "regression"]


class PartitionedAttention(nn.Module):
    """
    Computes single-query multi-head attention partitioned across subsets of SNPs
    with Top-KAST sparsification to isolate epistatic signals.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_partitions: int = 6,
        combination_size: int = 2,
        sparsity_ratio: float = 0.90,  # Top-KAST constraint: keep top (1 - sparsity_ratio)
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.num_partitions = num_partitions
        self.combination_size = combination_size
        self.sparsity_ratio = sparsity_ratio

        # Precompute partition combinations
        self.combinations = list(itertools.combinations(range(num_partitions), combination_size))

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,  # (B, 1, D)
        keys_values: torch.Tensor,  # (B, N, D)
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass computing partitioned attention.

        Args:
            query: Query tensor of shape (B, 1, embed_dim).
            keys_values: SNP token embeddings of shape (B, N_snps, embed_dim).
            return_attention: If True, returns global aggregated attention scores over SNPs (B, N_snps).

        Returns:
            (context_vector of shape (B, 1, embed_dim), optional attention_scores of shape (B, N_snps))
        """
        B, N, D = keys_values.shape

        # Linear projections & split into heads: (B, num_heads, L, head_dim)
        Q = self.q_proj(query).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(keys_values).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(keys_values).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Determine partition boundaries
        partition_size = math.ceil(N / self.num_partitions)
        partition_slices = []
        for p in range(self.num_partitions):
            start = p * partition_size
            end = min((p + 1) * partition_size, N)
            partition_slices.append((start, end))

        combination_contexts = []
        global_attention_map = (
            torch.zeros(B, N, device=keys_values.device, dtype=torch.float32)
            if return_attention
            else None
        )

        scale = 1.0 / math.sqrt(self.head_dim)

        # Iterate over partition combinations (e.g. C(6, 2) = 15 combinations)
        for comb in self.combinations:
            # Collect indices for this partition combination
            selected_indices: List[int] = []
            for p_idx in comb:
                st, ed = partition_slices[p_idx]
                if st < ed:
                    selected_indices.extend(range(st, ed))

            if not selected_indices:
                continue

            idx_tensor = torch.tensor(selected_indices, device=keys_values.device, dtype=torch.long)
            K_sub = torch.index_select(K, dim=2, index=idx_tensor)  # (B, H, M, d)
            V_sub = torch.index_select(V, dim=2, index=idx_tensor)  # (B, H, M, d)
            M = K_sub.size(2)

            # Dot-product attention: Q (B, H, 1, d) @ K_sub^T (B, H, d, M) -> (B, H, 1, M)
            attn_scores = torch.matmul(Q, K_sub.transpose(-2, -1)) * scale

            # Top-KAST Sparsification: retain top (1 - sparsity_ratio)
            if 0.0 < self.sparsity_ratio < 1.0 and M > 1:
                k_keep = max(1, int(M * (1.0 - self.sparsity_ratio)))
                topk_vals, _ = torch.topk(attn_scores, k=k_keep, dim=-1)
                threshold = topk_vals[:, :, :, -1:]
                # Mask out elements below top-k threshold
                mask = attn_scores < threshold
                attn_scores = attn_scores.masked_fill(mask, -1e9)

            attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, 1, M)
            attn_weights = self.dropout(attn_weights)

            # Context for this combination: (B, H, 1, M) @ (B, H, M, d) -> (B, H, 1, d)
            context = torch.matmul(attn_weights, V_sub)
            combination_contexts.append(context)

            if return_attention and global_attention_map is not None:
                # Average across heads: (B, 1, M) -> (B, M)
                comb_weights_avg = attn_weights.squeeze(2).mean(dim=1)  # (B, M)
                global_attention_map.index_add_(
                    dim=1, index=idx_tensor, source=comb_weights_avg
                )

        if not combination_contexts:
            # Fallback if no combinations were valid
            combined_context = torch.zeros(B, self.num_heads, 1, self.head_dim, device=keys_values.device)
        else:
            # Average context representations across all partition combinations
            combined_context = torch.stack(combination_contexts, dim=0).mean(dim=0)  # (B, H, 1, d)

        # Concatenate heads: (B, 1, H * d) = (B, 1, D)
        combined_context = (
            combined_context.transpose(1, 2).contiguous().view(B, 1, self.embed_dim)
        )
        output = self.out_proj(combined_context)

        if return_attention and global_attention_map is not None:
            # Normalize global attention map
            norm_factor = max(len(self.combinations), 1)
            global_attention_map = global_attention_map / norm_factor

        return output, global_attention_map


class EpistasisTransformer(nn.Module):
    """
    Complete Deep Learning Transformer for Higher-Order Epistasis Detection.
    Features:
    - Discrete SNP Allele Embedding {0, 1, 2} + Learnable Locus Position Embeddings
    - Single-Query Phenotype / Class Token
    - Multi-Partition Sparse Attention (Graça et al. architecture)
    - Dual Classification & Regression Heads for Case/Control and Continuous Pharmacogenomics
    """

    def __init__(
        self,
        num_snps: int = 1000,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_partitions: int = 6,
        combination_size: int = 2,
        sparsity_ratio: float = 0.90,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
        task: TaskType = "classification",
    ):
        super().__init__()
        self.num_snps = num_snps
        self.embed_dim = embed_dim
        self.task = task

        # SNP Allele Embedding (0: AA, 1: Aa, 2: aa) -> (embed_dim)
        self.snp_embedding = nn.Embedding(3, embed_dim)
        # Learnable Locus Positional Embedding for each SNP marker
        self.locus_embedding = nn.Parameter(torch.randn(1, num_snps, embed_dim) * 0.02)

        # Learnable Class / Query Token
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Partitioned Attention Layers
        self.layers = nn.ModuleList([
            PartitionedAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_partitions=num_partitions,
                combination_size=combination_size,
                sparsity_ratio=sparsity_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, embed_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        self.ff_norm_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

        self.final_norm = nn.LayerNorm(embed_dim)

        # Output Prediction Head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, N_snps) categorical {0, 1, 2} or continuous float
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Tensor of shape (B, N_snps).
            return_attention: If True, returns (logits, attention_scores).

        Returns:
            Output logits/predictions (B, 1) or tuple with attention scores (B, N_snps).
        """
        B, N = x.shape
        if x.dtype in (torch.int32, torch.int64):
            tokens = self.snp_embedding(x)  # (B, N, D)
        else:
            # Continuous/additive input: linear projection
            tokens = x.unsqueeze(-1) * self.locus_embedding[:, :N, :]

        tokens = tokens + self.locus_embedding[:, :N, :]

        # Expand query token for batch: (B, 1, D)
        q = self.query_token.expand(B, -1, -1)

        last_attention = None
        for i, (attn_layer, norm_a, ff_layer, norm_f) in enumerate(
            zip(self.layers, self.norm_layers, self.ff_layers, self.ff_norm_layers)
        ):
            # Compute partitioned attention
            is_last = (i == len(self.layers) - 1)
            attn_out, attn_map = attn_layer(q, tokens, return_attention=(return_attention and is_last))
            q = norm_a(q + attn_out)
            q = norm_f(q + ff_layer(q))
            if is_last and return_attention:
                last_attention = attn_map

        q = self.final_norm(q)
        output = self.head(q.squeeze(1))  # (B, 1)

        if return_attention:
            return output, last_attention
        return output
