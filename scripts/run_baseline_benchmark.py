"""Benchmark script comparing Transformer vs Classical/ML Baselines across epistasis models."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from epistasis.data.gametes_simulator import EpistasisSimulator
from epistasis.models.baselines import (
    MultifactorDimensionalityReduction,
    InteractionRegressionBaseline,
    TreeEnsembleBaseline,
)
from epistasis.models.deepcombi_mlp import DeepCOMBIBaseline
from epistasis.training import run_full_pipeline
from epistasis.evaluation.metrics import evaluate_epistasis_detection
from epistasis.utils.logger import get_logger

logger = get_logger("baseline_benchmark")


def run_benchmark(
    order: int = 2,
    model_type: str = "additive",
    maf: float = 0.2,
    heritability: float = 0.4,
    n_snps: int = 100,
    n_samples: int = 1600,
    num_replicates: int = 5,
    epochs: int = 40,
    output_dir: str = "reports/benchmarks",
) -> pd.DataFrame:
    """Executes multi-model benchmark comparison across replicate datasets."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        f"Starting Baseline Benchmark: Model={model_type}, Order={order}, "
        f"MAF={maf}, H2={heritability}, SNPs={n_snps}, Replicates={num_replicates}"
    )

    results = []

    for rep in range(num_replicates):
        sim = EpistasisSimulator(
            n_snps=n_snps,
            n_samples=n_samples,
            order=order,
            maf=maf,
            heritability=heritability,
            model_type=model_type,
            random_state=42 + rep,
        )
        data = sim.generate_case_control()
        X, y = data["X"], data["y"]
        causal = data["causal_indices"]

        # 1. Gated Partitioned Transformer (Proposed)
        num_partitions = 12 if n_snps >= 500 else 6
        tf_res = run_full_pipeline(
            data,
            num_partitions=num_partitions,
            combination_size=2,
            sparsity_ratio=0.85,
            aggregation="gated",
            epochs=epochs,
            device=device,
        )
        tf_eval = tf_res["detection_eval"]
        results.append({
            "replicate": rep + 1,
            "method": "Partitioned Transformer (Ours)",
            "detected_all_top_5pct": int(tf_eval["detected_all_top_5pct"]),
            "recall_top_5pct": tf_eval["recall_top_5pct"],
            "f1_top_5pct": tf_eval["f1_top_5pct"],
            "mean_causal_rank": tf_eval["mean_causal_rank"],
        })

        # 2. XGBoost Baseline
        xgb_model = TreeEnsembleBaseline(model_type="xgboost", n_estimators=100).fit(X, y)
        xgb_scores = xgb_model.score_features(n_snps)
        xgb_eval = evaluate_epistasis_detection(xgb_scores, causal)
        results.append({
            "replicate": rep + 1,
            "method": "XGBoost",
            "detected_all_top_5pct": int(xgb_eval["detected_all_top_5pct"]),
            "recall_top_5pct": xgb_eval["recall_top_5pct"],
            "f1_top_5pct": xgb_eval["f1_top_5pct"],
            "mean_causal_rank": xgb_eval["mean_causal_rank"],
        })

        # 3. Random Forest Baseline
        rf_model = TreeEnsembleBaseline(model_type="random_forest", n_estimators=100).fit(X, y)
        rf_scores = rf_model.score_features(n_snps)
        rf_eval = evaluate_epistasis_detection(rf_scores, causal)
        results.append({
            "replicate": rep + 1,
            "method": "Random Forest",
            "detected_all_top_5pct": int(rf_eval["detected_all_top_5pct"]),
            "recall_top_5pct": rf_eval["recall_top_5pct"],
            "f1_top_5pct": rf_eval["f1_top_5pct"],
            "mean_causal_rank": rf_eval["mean_causal_rank"],
        })

        # 4. Logistic Regression with Interaction Terms
        reg_model = InteractionRegressionBaseline(max_features=100).fit(X, y)
        reg_scores = reg_model.score_features(n_snps)
        reg_eval = evaluate_epistasis_detection(reg_scores, causal)
        results.append({
            "replicate": rep + 1,
            "method": "Interaction Logistic Regression",
            "detected_all_top_5pct": int(reg_eval["detected_all_top_5pct"]),
            "recall_top_5pct": reg_eval["recall_top_5pct"],
            "f1_top_5pct": reg_eval["f1_top_5pct"],
            "mean_causal_rank": reg_eval["mean_causal_rank"],
        })

        # 5. DeepCOMBI MLP Baseline
        deepcombi = DeepCOMBIBaseline(num_snps=n_snps, epochs=25, device=device).fit(X, y)
        deepcombi_scores = deepcombi.score_features(X)
        deepcombi_eval = evaluate_epistasis_detection(deepcombi_scores, causal)
        results.append({
            "replicate": rep + 1,
            "method": "DeepCOMBI (MLP + LRP)",
            "detected_all_top_5pct": int(deepcombi_eval["detected_all_top_5pct"]),
            "recall_top_5pct": deepcombi_eval["recall_top_5pct"],
            "f1_top_5pct": deepcombi_eval["f1_top_5pct"],
            "mean_causal_rank": deepcombi_eval["mean_causal_rank"],
        })

        # 6. Multifactor Dimensionality Reduction (MDR)
        mdr_candidates = list(range(min(n_snps, 50)))
        mdr_model = MultifactorDimensionalityReduction(order=order).fit(X, y, candidate_indices=mdr_candidates)
        mdr_scores = mdr_model.score_features(n_snps)
        mdr_eval = evaluate_epistasis_detection(mdr_scores, causal)
        results.append({
            "replicate": rep + 1,
            "method": "MDR (Candidate Filtered)",
            "detected_all_top_5pct": int(mdr_eval["detected_all_top_5pct"]),
            "recall_top_5pct": mdr_eval["recall_top_5pct"],
            "f1_top_5pct": mdr_eval["f1_top_5pct"],
            "mean_causal_rank": mdr_eval["mean_causal_rank"],
        })

    df = pd.DataFrame(results)
    summary = df.groupby("method").agg({
        "detected_all_top_5pct": "mean",
        "recall_top_5pct": "mean",
        "mean_causal_rank": "mean",
    }).rename(columns={"detected_all_top_5pct": "Detection Power (@ Top 5%)"})

    logger.info(f"\n=== BENCHMARK SUMMARY ({model_type.upper()} Epistasis, Order {order}) ===\n" + str(summary))
    df.to_csv(out_path / f"benchmark_{model_type}_order{order}.csv", index=False)
    summary.to_csv(out_path / f"benchmark_summary_{model_type}_order{order}.csv")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Epistasis Baseline Benchmark")
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--model", type=str, default="additive", choices=["additive", "multiplicative", "threshold", "xor"])
    parser.add_argument("--maf", type=float, default=0.2)
    parser.add_argument("--h2", type=float, default=0.4)
    parser.add_argument("--snps", type=int, default=100)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    run_benchmark(
        order=args.order,
        model_type=args.model,
        maf=args.maf,
        heritability=args.h2,
        n_snps=args.snps,
        num_replicates=args.replicates,
        epochs=args.epochs,
    )
