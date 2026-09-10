"""Comprehensive 5-Seed Fair Baseline Benchmark on 1000 SNPs (n=1600, order=2 additive)."""

import sys
import time
import numpy as np
import pandas as pd
import torch

from epistasis.data.gametes_simulator import EpistasisSimulator
from epistasis.data.dataset import create_dataloaders
from epistasis.models.baselines import (
    MultifactorDimensionalityReduction,
    InteractionRegressionBaseline,
    TreeEnsembleBaseline,
)
from epistasis.models.deepcombi_mlp import DeepCOMBIBaseline
from epistasis.training import run_full_pipeline
from epistasis.evaluation.metrics import evaluate_epistasis_detection

def run_fair_benchmark(seeds=[42, 101, 202, 303, 404], n_snps=1000, n_samples=1600):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running Fair 1000-SNP 5-Seed Benchmark on {device.upper()} ===", flush=True)
    print(f"Setup: N={n_snps} SNPs, n={n_samples} samples, order=2 additive, h2=0.4, MAF=0.2\n", flush=True)

    raw_results = []

    for rep_idx, seed in enumerate(seeds):
        print(f"\n==================================================================", flush=True)
        print(f"=== REPLICATE {rep_idx+1}/{len(seeds)} | SEED {seed} ===", flush=True)
        print(f"==================================================================", flush=True)

        # Set PRNG seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # 1. Generate Dataset
        sim = EpistasisSimulator(
            n_snps=n_snps,
            n_samples=n_samples,
            order=2,
            maf=0.2,
            heritability=0.4,
            model_type="additive",
            random_state=seed,
        )
        data = sim.generate_case_control()
        causal = data["causal_indices"]
        print(f"Dataset generated. Causal Loci: {causal}", flush=True)

        # 2. Strict Train/Val/Test Split (70% / 15% / 15%)
        train_loader, val_loader, test_loader, splits = create_dataloaders(
            data, batch_size=64, random_state=seed
        )
        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        X_test, y_test = splits["test"]

        # -------------------------------------------------------------
        # Model 1: XGBoost (Trained on Train split only)
        # -------------------------------------------------------------
        t0 = time.time()
        xgb_model = TreeEnsembleBaseline(model_type="xgboost", n_estimators=100).fit(X_train, y_train)
        xgb_scores = xgb_model.score_features(n_snps)
        xgb_eval = evaluate_epistasis_detection(xgb_scores, causal, top_percent=0.05)
        t_xgb = time.time() - t0
        print(f"  [XGBoost]                 Time: {t_xgb:5.1f}s | Causal: {str(xgb_eval['causal_ranks']):<14} | Mean Rank: {xgb_eval['mean_causal_rank']:5.1f} | Top 5%: {xgb_eval['detected_all_top_5pct']}", flush=True)
        raw_results.append({
            "seed": seed,
            "method": "XGBoost",
            "detected_top_5pct": xgb_eval["detected_all_top_5pct"],
            "mean_causal_rank": round(xgb_eval["mean_causal_rank"], 1),
            "causal_ranks": str(xgb_eval["causal_ranks"]),
            "runtime_sec": round(t_xgb, 1),
        })

        # -------------------------------------------------------------
        # Model 2: Random Forest (Trained on Train split only)
        # -------------------------------------------------------------
        t0 = time.time()
        rf_model = TreeEnsembleBaseline(model_type="random_forest", n_estimators=100).fit(X_train, y_train)
        rf_scores = rf_model.score_features(n_snps)
        rf_eval = evaluate_epistasis_detection(rf_scores, causal, top_percent=0.05)
        t_rf = time.time() - t0
        print(f"  [Random Forest]           Time: {t_rf:5.1f}s | Causal: {str(rf_eval['causal_ranks']):<14} | Mean Rank: {rf_eval['mean_causal_rank']:5.1f} | Top 5%: {rf_eval['detected_all_top_5pct']}", flush=True)
        raw_results.append({
            "seed": seed,
            "method": "Random Forest",
            "detected_top_5pct": rf_eval["detected_all_top_5pct"],
            "mean_causal_rank": round(rf_eval["mean_causal_rank"], 1),
            "causal_ranks": str(rf_eval["causal_ranks"]),
            "runtime_sec": round(t_rf, 1),
        })

        # -------------------------------------------------------------
        # Model 3: Interaction Logistic Regression (Train split only)
        # -------------------------------------------------------------
        t0 = time.time()
        reg_model = InteractionRegressionBaseline(max_features=100).fit(X_train, y_train)
        reg_scores = reg_model.score_features(n_snps)
        reg_eval = evaluate_epistasis_detection(reg_scores, causal, top_percent=0.05)
        t_reg = time.time() - t0
        print(f"  [Interaction Logistic]   Time: {t_reg:5.1f}s | Causal: {str(reg_eval['causal_ranks']):<14} | Mean Rank: {reg_eval['mean_causal_rank']:5.1f} | Top 5%: {reg_eval['detected_all_top_5pct']}", flush=True)
        raw_results.append({
            "seed": seed,
            "method": "Interaction Logistic",
            "detected_top_5pct": reg_eval["detected_all_top_5pct"],
            "mean_causal_rank": round(reg_eval["mean_causal_rank"], 1),
            "causal_ranks": str(reg_eval["causal_ranks"]),
            "runtime_sec": round(t_reg, 1),
        })

        # -------------------------------------------------------------
        # Model 4: DeepCOMBI MLP (Trained on Train, evaluated on Test)
        # -------------------------------------------------------------
        t0 = time.time()
        deepcombi = DeepCOMBIBaseline(num_snps=n_snps, epochs=30, device=device).fit(X_train, y_train)
        deepcombi_scores = deepcombi.score_features(X_test)
        deepcombi_eval = evaluate_epistasis_detection(deepcombi_scores, causal, top_percent=0.05)
        t_dc = time.time() - t0
        print(f"  [DeepCOMBI MLP]           Time: {t_dc:5.1f}s | Causal: {str(deepcombi_eval['causal_ranks']):<14} | Mean Rank: {deepcombi_eval['mean_causal_rank']:5.1f} | Top 5%: {deepcombi_eval['detected_all_top_5pct']}", flush=True)
        raw_results.append({
            "seed": seed,
            "method": "DeepCOMBI (MLP + Saliency)",
            "detected_top_5pct": deepcombi_eval["detected_all_top_5pct"],
            "mean_causal_rank": round(deepcombi_eval["mean_causal_rank"], 1),
            "causal_ranks": str(deepcombi_eval["causal_ranks"]),
            "runtime_sec": round(t_dc, 1),
        })

        # -------------------------------------------------------------
        # Model 5: Multifactor Dimensionality Reduction (MDR)
        # Scan top 100 univariate variance/association SNPs on train
        # -------------------------------------------------------------
        t0 = time.time()
        variances = np.var(X_train, axis=0)
        mdr_candidates = np.argsort(variances)[-100:].tolist()
        mdr_model = MultifactorDimensionalityReduction(order=2).fit(X_train, y_train, candidate_indices=mdr_candidates)
        mdr_scores = mdr_model.score_features(n_snps)
        mdr_eval = evaluate_epistasis_detection(mdr_scores, causal, top_percent=0.05)
        t_mdr = time.time() - t0
        print(f"  [MDR (Top 100 candidates)]Time: {t_mdr:5.1f}s | Causal: {str(mdr_eval['causal_ranks']):<14} | Mean Rank: {mdr_eval['mean_causal_rank']:5.1f} | Top 5%: {mdr_eval['detected_all_top_5pct']}", flush=True)
        raw_results.append({
            "seed": seed,
            "method": "MDR (Candidate Filtered)",
            "detected_top_5pct": mdr_eval["detected_all_top_5pct"],
            "mean_causal_rank": round(mdr_eval["mean_causal_rank"], 1),
            "causal_ranks": str(mdr_eval["causal_ranks"]),
            "runtime_sec": round(t_mdr, 1),
        })

        # -------------------------------------------------------------
        # Model 6: Gated Partitioned Transformer (P=6, EMA-smoothed, Test-only ACAT)
        # -------------------------------------------------------------
        t0 = time.time()
        tf_res = run_full_pipeline(
            data_dict=data,
            epochs=30,
            num_partitions=6,
            combination_size=2,
            sparsity_ratio=0.85,
            aggregation="gated",
            warmup_epochs=5,
            ema_decay=0.7,
            random_state=seed,
            device=device,
        )
        tf_eval = tf_res["detection_eval"]
        t_tf = time.time() - t0
        print(f"  [Sparse Transformer (P=6)]Time: {t_tf:5.1f}s | Causal: {str(tf_eval['causal_ranks']):<14} | Mean Rank: {tf_eval['mean_causal_rank']:5.1f} | Top 5%: {tf_eval['detected_all_top_5pct']}", flush=True)
        raw_results.append({
            "seed": seed,
            "method": "Partitioned Transformer (P=6)",
            "detected_top_5pct": tf_eval["detected_all_top_5pct"],
            "mean_causal_rank": round(tf_eval["mean_causal_rank"], 1),
            "causal_ranks": str(tf_eval["causal_ranks"]),
            "runtime_sec": round(t_tf, 1),
        })

    df_raw = pd.DataFrame(raw_results)
    
    # Compute Summary Statistics
    summary_rows = []
    for method, grp in df_raw.groupby("method", sort=False):
        summary_rows.append({
            "method": method,
            "detection_power": f"{grp['detected_top_5pct'].mean()*100:.1f}%",
            "mean_causal_rank": f"{grp['mean_causal_rank'].mean():.1f} +/- {grp['mean_causal_rank'].std():.1f}",
            "mean_runtime_sec": f"{grp['runtime_sec'].mean():.1f}s",
        })
    df_summary = pd.DataFrame(summary_rows)

    print("\n\n==================================================================", flush=True)
    print("=== FINAL 5-SEED FAIR BENCHMARK SUMMARY (1000 SNPs, n=1600) ===", flush=True)
    print("==================================================================", flush=True)
    print(df_summary.to_string(index=False), flush=True)

    df_raw.to_csv("reports/benchmarks/baseline_comparison_1000snp_5seed.csv", index=False)
    df_summary.to_csv("reports/benchmarks/baseline_summary_1000snp_5seed.csv", index=False)
    print("\nSaved raw per-seed results to reports/benchmarks/baseline_comparison_1000snp_5seed.csv", flush=True)
    print("Saved summary table to reports/benchmarks/baseline_summary_1000snp_5seed.csv", flush=True)

if __name__ == "__main__":
    run_fair_benchmark()
