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
    epochs: int = 20,
    output_dir: str = "reports/pharmacogenomics",
) -> pd.DataFrame:
    """Runs pharmacogenomic continuous phenotype epistasis detection and pathway validation."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        f"Starting Pharmacogenomics Pipeline: Cell Lines={n_cell_lines}, "
        f"Genes={n_genes}, Order={interaction_order}, H2={heritability}"
    )

    loader = PharmacogenomicsDataLoader()
    data = loader.generate_synthetic_pharmacogenomic_cohort(
        n_cell_lines=n_cell_lines,
        n_genes=n_genes,
        interaction_order=interaction_order,
        heritability=heritability,
        random_state=42,
    )

    gene_names = data["gene_names"]
    causal_genes = data["causal_genes"]
    causal_indices = data["causal_indices"]

    logger.info(f"Target Interacting Causal Genes: {causal_genes}")

    # Run transformer pipeline with regression head
    res = run_full_pipeline(
        data_dict=data,
        num_partitions=5,
        combination_size=2,
        sparsity_ratio=0.80,
        epochs=epochs,
        device=device,
    )

    scores = res["combined_scores"]
    d_eval = res["detection_eval"]

    # Rank all candidate genes
    ranked_idx = np.argsort(scores)[::-1]
    top_k = min(15, n_genes)
    top_genes = [gene_names[i] for i in ranked_idx[:top_k]]

    logger.info(f"Top {top_k} Candidate Genes Ranked by Transformer:")
    gene_records = []
    for rank, idx in enumerate(ranked_idx[:top_k], start=1):
        gene_name = gene_names[idx]
        is_causal = idx in causal_indices
        score = float(scores[idx])
        gene_records.append({
            "rank": rank,
            "gene": gene_name,
            "importance_score": score,
            "is_ground_truth_causal": is_causal,
        })
        logger.info(f"  Rank {rank:02d}: {gene_name} (Score: {score:.4f}) {'[CAUSAL TARGET]' if is_causal else ''}")

    df_genes = pd.DataFrame(gene_records)
    df_genes.to_csv(out_path / "ranked_candidate_genes.csv", index=False)

    # Biological pathway / STRING validation
    logger.info("Performing Biological Validation against STRING / Synthetic Lethality databases...")
    validator = BiologicalValidator()
    string_interactions = validator.query_string_interactions(top_genes)

    logger.info(f"Discovered {len(string_interactions)} Protein-Protein / Functional Interactions in Top Candidates:")
    for inter in string_interactions:
        logger.info(f"  {inter['gene_a']} <---> {inter['gene_b']} (Score: {inter.get('string_score', 0):.3f})")

    df_inter = pd.DataFrame(string_interactions)
    if not df_inter.empty:
        df_inter.to_csv(out_path / "candidate_pathway_interactions.csv", index=False)

    logger.info(
        f"Pipeline Complete! Causal Detection Power: {d_eval['detected_all_top_5pct']}, "
        f"Mean Causal Rank: {d_eval['mean_causal_rank']:.1f} / {n_genes}"
    )

    return df_genes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pharmacogenomics Epistasis Runner")
    parser.add_argument("--cell_lines", type=int, default=800)
    parser.add_argument("--genes", type=int, default=50)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--h2", type=float, default=0.35)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    run_pharmacogenomics_pipeline(
        n_cell_lines=args.cell_lines,
        n_genes=args.genes,
        interaction_order=args.order,
        heritability=args.h2,
        epochs=args.epochs,
    )
