"""Visualization utilities for generating publication-quality figures."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_manhattan_importance(
    scores: np.ndarray,
    causal_indices: List[int],
    title: str = "SNP Importance Scores (ACAT: Attention + Gradients)",
    output_path: Optional[Union[str, Path]] = "reports/figures/manhattan_importance.png",
) -> None:
    """Plots a Manhattan-style scatter plot of SNP importance with causal loci highlighted."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

    n_snps = len(scores)
    x = np.arange(n_snps)

    # Plot background SNPs
    non_causal = [i for i in range(n_snps) if i not in causal_indices]
    ax.scatter(non_causal, scores[non_causal], color="#94a3b8", alpha=0.6, s=16, label="Non-causal Background SNPs")

    # Highlight causal SNPs
    ax.scatter(
        causal_indices,
        scores[causal_indices],
        color="#ef4444",
        s=80,
        edgecolor="black",
        linewidth=1.2,
        zorder=5,
        label=f"Interacting Causal Loci (k={len(causal_indices)})",
    )

    # Add 95th percentile threshold line
    p95 = np.percentile(scores, 95)
    ax.axhline(p95, color="#3b82f6", linestyle="--", linewidth=1.5, label="Top 5% Threshold")

    ax.set_xlabel("SNP Locus Index", fontsize=12, fontweight="bold")
    ax.set_ylabel("Attentive Activation Score (ACAT)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out)
        plt.close()


def plot_benchmark_comparison(
    summary_df: pd.DataFrame,
    title: str = "Epistasis Detection Power Comparison (Top 5% Criterion)",
    output_path: Optional[Union[str, Path]] = "reports/figures/benchmark_comparison.png",
) -> None:
    """Plots comparative detection power bar chart across methods."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    palette = ["#3b82f6" if "Transformer" in m else "#94a3b8" for m in summary_df["method"]]
    sns.barplot(data=summary_df, x="method", y="detection_power", palette=palette, ax=ax, edgecolor="black", linewidth=1.0)

    ax.set_xlabel("Method", fontsize=12, fontweight="bold")
    ax.set_ylabel("Detection Power (%)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(0, 105)
    plt.xticks(rotation=20, ha="right")

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height:.1f}%",
            (p.get_x() + p.get_width() / 2.0, height + 2),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out)
        plt.close()
