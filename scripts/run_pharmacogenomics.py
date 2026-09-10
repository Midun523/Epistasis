"""Script for Phase 2/3: Pharmacogenomic Epistasis Detection and Biological Validation."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from epistasis.data.pharmacogenomics import PharmacogenomicsDataLoader
from epistasis.training import run_full_pipeline
from epistasis.evaluation.pathway_validation import BiologicalValidator
from epistasis.utils.logger import get_logger

logger = get_logger("pharmacogenomics_runner")


def run_pharmacogenomics_pipeline(
    n_cell_lines: int = 800,
    n_genes: int = 50,
    interaction_order: int = 3,
    heritability: float = 0.35,
    epochs: int = 40,
    num_replicates: int = 5,
    output_dir: str = "reports/pharmacogenomics",
) -> pd.DataFrame:
    """Runs pharmacogenomic continuous phenotype epistasis detection and pathway validation across multiple replicates."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        f"Starting Pharmacogenomics Multi-Replicate Pipeline: Cell Lines={n_cell_lines}, "
        f"Genes={n_genes}, Order={interaction_order}, H2={heritability}, Replicates={num_replicates}"
    )

    seeds = [42, 101, 202, 303, 404][:num_replicates]
    rep_records = []
    all_top_genes = []

    for rep_idx, seed in enumerate(seeds):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        loader = PharmacogenomicsDataLoader()
        data = loader.generate_synthetic_pharmacogenomic_cohort(
            n_cell_lines=n_cell_lines,
            n_genes=n_genes,
            interaction_order=interaction_order,
            heritability=heritability,
            random_state=seed,
        )

        gene_names = data["gene_names"]
        causal_genes = data["causal_genes"]
        causal_indices = data["causal_indices"]

        # Run transformer pipeline with regression head and EMA-smoothed R^2
        res = run_full_pipeline(
            data_dict=data,
            num_partitions=5,
            combination_size=2,
            sparsity_ratio=0.80,
            epochs=epochs,
            warmup_epochs=5,
            ema_decay=0.7,
            random_state=seed,
            device=device,
        )

        scores = res["combined_scores"]
        d_eval = res["detection_eval"]
        best_r2 = max(res["history"]["val_metric"])
        smoothed_r2 = max(res["history"]["val_metric_smoothed"])

        # Top 5 ranked genes for this replicate
        ranked_idx = np.argsort(scores)[::-1]
        top_genes_rep = [gene_names[i] for i in ranked_idx[:5]]
        all_top_genes.extend(top_genes_rep)

        logger.info(
            f"  [Rep {rep_idx+1}/{num_replicates} | Seed {seed}] "
            f"Best Val R2: {best_r2:.4f} (Smoothed: {smoothed_r2:.4f}) | "
            f"Causal Genes: {causal_genes} | "
            f"Causal Ranks: {str(d_eval['causal_ranks']):<16} | "
            f"Mean Rank: {d_eval['mean_causal_rank']:4.1f} / {n_genes} | "
            f"Detected Top 10%: {d_eval['recall_top_5pct'] >= 1.0}"
        )

        rep_records.append({
            "replicate": rep_idx + 1,
            "seed": seed,
            "best_val_r2": round(best_r2, 4),
            "smoothed_val_r2": round(smoothed_r2, 4),
            "causal_genes": str(causal_genes),
            "causal_ranks": str(d_eval["causal_ranks"]),
            "mean_causal_rank": round(d_eval["mean_causal_rank"], 1),
            "top_1_gene": gene_names[ranked_idx[0]],
            "top_2_gene": gene_names[ranked_idx[1]],
            "top_3_gene": gene_names[ranked_idx[2]],
            "detected_all_top_10pct": bool(max(d_eval["causal_ranks"]) <= max(1, int(n_genes * 0.10))),
        })

    df_reps = pd.DataFrame(rep_records)
    df_reps.to_csv(out_path / "pharmacogenomics_replicate_runs.csv", index=False)

    summary_record = [{
        "cohort_size": n_cell_lines,
        "n_genes": n_genes,
        "interaction_order": interaction_order,
        "heritability": heritability,
        "mean_val_r2": f"{df_reps['best_val_r2'].mean():.4f} +/- {df_reps['best_val_r2'].std():.4f}",
        "mean_causal_rank": f"{df_reps['mean_causal_rank'].mean():.1f} +/- {df_reps['mean_causal_rank'].std():.1f}",
        "detection_power_top_10pct": f"{df_reps['detected_all_top_10pct'].mean()*100:.1f}%",
    }]
    df_summary = pd.DataFrame(summary_record)
    df_summary.to_csv(out_path / "pharmacogenomics_summary.csv", index=False)

    logger.info(f"\n=== PHARMACOGENOMICS MULTI-REPLICATE SUMMARY ===\n{df_summary.to_string(index=False)}\n")

    # Query STRING API on unique top candidate genes
    unique_top_genes = list(dict.fromkeys(all_top_genes))[:15]
    logger.info(f"Querying STRING API on Top Candidate Genes: {unique_top_genes}")
    validator = BiologicalValidator()
    string_interactions = validator.query_string_interactions(unique_top_genes)

    logger.info(f"Discovered {len(string_interactions)} Live Protein-Protein / Functional Interactions:")
    for inter in string_interactions[:10]:
        logger.info(f"  {inter['gene_a']} <---> {inter['gene_b']} (Score: {inter.get('string_score', 0):.3f})")

    df_inter = pd.DataFrame(string_interactions)
    if not df_inter.empty:
        df_inter.to_csv(out_path / "candidate_pathway_interactions.csv", index=False)

    return df_reps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pharmacogenomics Epistasis Runner")
    parser.add_argument("--cell_lines", type=int, default=800)
    parser.add_argument("--genes", type=int, default=50)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--h2", type=float, default=0.35)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--replicates", type=int, default=5)
    args = parser.parse_args()

    run_pharmacogenomics_pipeline(
        n_cell_lines=args.cell_lines,
        n_genes=args.genes,
        interaction_order=args.order,
        heritability=args.h2,
        epochs=args.epochs,
        num_replicates=args.replicates,
    )
