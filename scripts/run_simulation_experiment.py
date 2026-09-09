"""Script to execute Phase 1a simulation parameter sweeps and evaluate epistasis detection power."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from epistasis.data.gametes_simulator import EpistasisSimulator
from epistasis.training import run_full_pipeline
from epistasis.utils.logger import get_logger

logger = get_logger("simulation_runner")


def run_experiment(
    order: int = 2,
    model_type: str = "xor",
    maf: float = 0.2,
    heritability: float = 0.2,
    n_snps: int = 1000,
    n_samples: int = 1600,
    num_replicates: int = 5,
    num_partitions: int = 6,
    combination_size: int = 2,
    sparsity_ratio: float = 0.90,
    epochs: int = 15,
    output_dir: str = "reports/simulations",
) -> pd.DataFrame:
    """Runs replicates for a single parameter configuration and records detection metrics."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        f"Starting Simulation Run: Order={order}, Model={model_type}, MAF={maf}, "
        f"H2={heritability}, Replicates={num_replicates}, Device={device}"
    )

    records = []

    for rep in tqdm(range(num_replicates), desc=f"Replicates (Order {order} {model_type})"):
        sim = EpistasisSimulator(
            n_snps=n_snps,
            n_samples=n_samples,
            order=order,
            maf=maf,
            heritability=heritability,
            model_type=model_type,
            random_state=1000 + rep,
        )
        data = sim.generate_case_control()

        res = run_full_pipeline(
            data_dict=data,
            num_partitions=num_partitions,
            combination_size=combination_size,
            sparsity_ratio=sparsity_ratio,
            epochs=epochs,
            device=device,
        )

        d_eval = res["detection_eval"]
        records.append({
            "replicate": rep + 1,
            "order": order,
            "model_type": model_type,
            "maf": maf,
            "heritability": heritability,
            "detected_all_top_5pct": int(d_eval["detected_all_top_5pct"]),
            "precision_top_5pct": d_eval["precision_top_5pct"],
            "recall_top_5pct": d_eval["recall_top_5pct"],
            "f1_top_5pct": d_eval["f1_top_5pct"],
            "precision_exact": d_eval["precision_exact"],
            "recall_exact": d_eval["recall_exact"],
            "mean_causal_rank": d_eval["mean_causal_rank"],
            "max_causal_rank": d_eval["max_causal_rank"],
        })

    df = pd.DataFrame(records)
    detection_power = df["detected_all_top_5pct"].mean()
    mean_recall = df["recall_top_5pct"].mean()
    mean_rank = df["mean_causal_rank"].mean()

    logger.info(
        f"=== RESULTS [Order {order} | {model_type} | MAF {maf} | H2 {heritability}] ===\n"
        f"  Detection Power (@ Top 5%): {detection_power * 100:.1f}%\n"
        f"  Mean Recall: {mean_recall:.3f}\n"
        f"  Mean Causal SNP Rank: {mean_rank:.1f} / {n_snps}"
    )

    filename = f"results_{model_type}_order{order}_maf{maf}_h2{heritability}.csv"
    df.to_csv(out_path / filename, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Epistasis Simulation Runner")
    parser.add_argument("--order", type=int, default=2, help="Interaction order (2-8)")
    parser.add_argument("--model", type=str, default="xor", choices=["additive", "multiplicative", "threshold", "xor"])
    parser.add_argument("--maf", type=float, default=0.2, help="Minor Allele Frequency")
    parser.add_argument("--h2", type=float, default=0.2, help="Broad-sense Heritability")
    parser.add_argument("--replicates", type=int, default=5, help="Number of replicates")
    parser.add_argument("--snps", type=int, default=1000, help="Total SNPs")
    parser.add_argument("--samples", type=int, default=1600, help="Total samples")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    args = parser.parse_args()

    run_experiment(
        order=args.order,
        model_type=args.model,
        maf=args.maf,
        heritability=args.h2,
        n_snps=args.snps,
        n_samples=args.samples,
        num_replicates=args.replicates,
        epochs=args.epochs,
    )
