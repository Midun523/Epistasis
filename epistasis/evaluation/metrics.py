"""Evaluation metrics for epistasis detection benchmarks."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


def evaluate_epistasis_detection(
    scores: np.ndarray,
    causal_indices: List[int],
    top_percent: float = 0.05,
) -> Dict[str, Union[float, bool, List[int]]]:
    """
    Computes detection power, precision, recall, and ranking statistics for a single simulation run.

    Args:
        scores: 1D array of SNP importance scores of shape (N_snps,).
        causal_indices: List of ground-truth causal SNP indices.
        top_percent: Fraction of top SNPs to evaluate (default: 0.05 = top 5%).

    Returns:
        Dictionary of detection metrics.
    """
    n_snps = len(scores)
    causal_set = set(causal_indices)
    k_causal = len(causal_indices)

    # Rank SNPs in descending order of score
    ranked_indices = np.argsort(scores)[::-1]

    # Top 5% threshold count
    top_k_count = max(1, int(n_snps * top_percent))
    top_k_snps = set(ranked_indices[:top_k_count])

    # Detection Power criterion: ARE ALL CAUSAL SNPS IN TOP 5%?
    all_detected_top_pct = causal_set.issubset(top_k_snps)

    # Intersection at top 5%
    tp_top_pct = len(causal_set.intersection(top_k_snps))
    precision_top_pct = tp_top_pct / top_k_count
    recall_top_pct = tp_top_pct / k_causal
    f1_top_pct = (
        (2.0 * precision_top_pct * recall_top_pct) / (precision_top_pct + recall_top_pct)
        if (precision_top_pct + recall_top_pct) > 0
        else 0.0
    )

    # Exact Top-K (where K = number of causal SNPs)
    top_exact_snps = set(ranked_indices[:k_causal])
    tp_exact = len(causal_set.intersection(top_exact_snps))
    precision_exact = tp_exact / k_causal
    recall_exact = tp_exact / k_causal
    f1_exact = precision_exact

    # Ranks of each causal SNP (1-indexed)
    causal_ranks = []
    for c_idx in causal_indices:
        # Find position of c_idx in ranked_indices
        rank_0 = int(np.where(ranked_indices == c_idx)[0][0])
        causal_ranks.append(rank_0 + 1)

    mean_causal_rank = float(np.mean(causal_ranks))
    max_causal_rank = int(np.max(causal_ranks))

    return {
        "detected_all_top_5pct": bool(all_detected_top_pct),
        "precision_top_5pct": float(precision_top_pct),
        "recall_top_5pct": float(recall_top_pct),
        "f1_top_5pct": float(f1_top_pct),
        "precision_exact": float(precision_exact),
        "recall_exact": float(recall_exact),
        "f1_exact": float(f1_exact),
        "causal_ranks": causal_ranks,
        "mean_causal_rank": mean_causal_rank,
        "max_causal_rank": max_causal_rank,
        "top_k_count": top_k_count,
        "n_snps": n_snps,
    }


def compute_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "classification",
) -> Dict[str, float]:
    """Computes phenotypic prediction performance metrics on test splits."""
    if task == "classification":
        # y_pred are probabilities or logits
        if np.any(y_pred < 0) or np.any(y_pred > 1):
            y_prob = 1.0 / (1.0 + np.exp(-y_pred))
        else:
            y_prob = y_pred

        y_bin = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_true, y_bin)
        try:
            auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
        except Exception:
            auc, pr_auc = 0.5, 0.5

        return {
            "accuracy": float(acc),
            "roc_auc": float(auc),
            "pr_auc": float(pr_auc),
        }
    else:  # regression
        mse = float(np.mean((y_true - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        var_y = np.var(y_true)
        r2 = float(1.0 - (mse / max(var_y, 1e-8)))
        return {
            "mse": mse,
            "mae": mae,
            "r2": r2,
        }
