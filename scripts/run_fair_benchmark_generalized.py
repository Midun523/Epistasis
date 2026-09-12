"""
Fair N-Seed Baseline Benchmark, generalized for interaction order and model type.

This extends run_fair_1000snp_benchmark.py (order=2 additive only) to test the
actual core hypothesis of the project: does the transformer's advantage over
classical baselines emerge at higher interaction order and/or harder epistasis
models (xor, threshold), where order-2 testing already showed classical
methods (XGBoost, RF, DeepCOMBI) winning cleanly?

Known, expected limitation kept intentionally (not a bug): InteractionRegressionBaseline
only builds pairwise (X_i * X_j) terms. For order >= 3, it structurally cannot
represent the true interaction directly. It's kept in the comparison anyway --
its failure (or partial success via lower-order marginal structure some models
leak) is itself an informative, reportable data point.
"""

import argparse
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


import hashlib
import os
import uuid
import datetime

def run_fair_benchmark(
    seeds=(42, 101, 202, 303, 404),
    n_snps=1000,
    n_samples=1600,
    order=2,
    model_type="additive",
    heritability=0.4,
    maf=0.2,
    epochs=30,
    num_partitions=6,
    mdr_candidate_size=100,
):
    script_start_time = time.time()
    run_id = uuid.uuid4().hex[:8]
    run_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Adaptive batch size: at large N, the transformer's locus_embedding (N×64) and
    # token tensor (B×N×64) must fit in GPU VRAM alongside per-combination autograd
    # intermediates. Even with gradient checkpointing, B=64 at N=100,000 creates a
    # 1.5 GB token tensor; B=16 keeps it at ~400 MB with comfortable headroom on 6 GB.
    if n_snps >= 50000:
        batch_size = 16
    elif n_snps >= 10000:
        batch_size = 32
    else:
        batch_size = 64

    print(f"=== Fair {len(seeds)}-Seed Benchmark | Run ID: {run_id} | Timestamp: {run_timestamp} ===", flush=True)
    print(f"=== order={order} model={model_type} N={n_snps} n={n_samples} h2={heritability} epochs={epochs} batch_size={batch_size} on {device.upper()} ===", flush=True)

    raw_results = []

    for rep_idx, seed in enumerate(seeds):
        print(f"\n=== REPLICATE {rep_idx+1}/{len(seeds)} | SEED {seed} ===", flush=True)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        sim = EpistasisSimulator(
            n_snps=n_snps, n_samples=n_samples, order=order, maf=maf,
            heritability=heritability, model_type=model_type, random_state=seed,
        )
        data = sim.generate_case_control()
        causal = data["causal_indices"]
        print(f"Dataset generated. Causal Loci ({order}-way): {causal}", flush=True)

        train_loader, val_loader, test_loader, splits = create_dataloaders(
            data, batch_size=batch_size, random_state=seed
        )
        X_train, y_train = splits["train"]
        X_test, y_test = splits["test"]

        def _record(method, eval_result, runtime, epochs_trained=None, best_val=None, note=""):
            ep_info = f" (ep {epochs_trained}, val {best_val:.4f})" if epochs_trained is not None and best_val is not None else ""
            print(f"  [{method:<28}] Time: {runtime:6.1f}s | Causal: {str(eval_result['causal_ranks']):<16} "
                  f"| Mean Rank: {eval_result['mean_causal_rank']:6.1f} | Top 5%: {eval_result['detected_all_top_5pct']}"
                  f"{ep_info}{'  ' + note if note else ''}", flush=True)
            raw_results.append({
                "run_id": run_id,
                "run_timestamp": run_timestamp,
                "epochs_budget": epochs,
                "epochs_trained": epochs_trained if epochs_trained is not None else 1,
                "best_val_metric": round(best_val, 4) if best_val is not None else np.nan,
                "seed": seed, "order": order, "model_type": model_type, "method": method,
                "detected_top_5pct": eval_result["detected_all_top_5pct"],
                "mean_causal_rank": round(eval_result["mean_causal_rank"], 1),
                "causal_ranks": str(eval_result["causal_ranks"]),
                "runtime_sec": round(runtime, 1),
            })

        # XGBoost
        t0 = time.time()
        xgb = TreeEnsembleBaseline(model_type="xgboost", n_estimators=100).fit(X_train, y_train)
        xgb_eval = evaluate_epistasis_detection(xgb.score_features(n_snps), causal, top_percent=0.05)
        _record("XGBoost", xgb_eval, time.time() - t0)

        # Random Forest
        t0 = time.time()
        rf = TreeEnsembleBaseline(model_type="random_forest", n_estimators=100).fit(X_train, y_train)
        rf_eval = evaluate_epistasis_detection(rf.score_features(n_snps), causal, top_percent=0.05)
        _record("Random Forest", rf_eval, time.time() - t0)

        # Interaction Logistic (pairwise-only -- structurally limited for order>=3)
        t0 = time.time()
        reg = InteractionRegressionBaseline(max_features=100).fit(X_train, y_train)
        reg_eval = evaluate_epistasis_detection(reg.score_features(n_snps), causal, top_percent=0.05)
        note = "(pairwise-only; can't represent 3-way+ directly)" if order >= 3 else ""
        _record("Interaction Logistic", reg_eval, time.time() - t0, note=note)

        # DeepCOMBI MLP
        t0 = time.time()
        deepcombi = DeepCOMBIBaseline(num_snps=n_snps, epochs=30, device=device).fit(X_train, y_train)
        dc_eval = evaluate_epistasis_detection(deepcombi.score_features(X_test), causal, top_percent=0.05)
        _record("DeepCOMBI (MLP + Saliency)", dc_eval, time.time() - t0)

        # MDR at the true interaction order (exhaustive if comb <= 2,000,000, else candidate-filtered)
        t0 = time.time()
        from math import comb
        if comb(n_snps, order) > 2_000_000:
            variances = np.var(X_train, axis=0)
            max_feasible = min(mdr_candidate_size, n_snps)
            mdr_candidates = np.argsort(variances)[-max_feasible:].tolist()
            mdr_label = f"MDR (order={order}, candidate-filtered)"
        else:
            mdr_candidates = None
            mdr_label = f"MDR (order={order}, exhaustive)"
        mdr = MultifactorDimensionalityReduction(order=order).fit(X_train, y_train, candidate_indices=mdr_candidates)
        mdr_eval = evaluate_epistasis_detection(mdr.score_features(n_snps), causal, top_percent=0.05)
        _record(mdr_label, mdr_eval, time.time() - t0)

        # Partitioned Transformer (gated, EMA-smoothed, test-only ACAT -- all this session's fixes)
        t0 = time.time()
        tf_res = run_full_pipeline(
            data_dict=data, epochs=epochs, num_partitions=num_partitions, combination_size=2,
            sparsity_ratio=0.85, aggregation="gated", warmup_epochs=5, ema_decay=0.7,
            random_state=seed, device=device, batch_size=batch_size,
        )
        tf_eval = tf_res["detection_eval"]
        tf_epochs = len(tf_res.get("history", {}).get("val_metric", []))
        tf_best_val = max(tf_res.get("history", {}).get("val_metric", [0.0]))
        _record(f"Partitioned Transformer (P={num_partitions})", tf_eval, time.time() - t0,
                epochs_trained=tf_epochs, best_val=tf_best_val)

    df_raw = pd.DataFrame(raw_results)
    summary_rows = []
    for method, grp in df_raw.groupby("method", sort=False):
        summary_rows.append({
            "run_id": run_id,
            "run_timestamp": run_timestamp,
            "epochs_budget": epochs,
            "method": method,
            "order": order,
            "model_type": model_type,
            "detection_power": f"{grp['detected_top_5pct'].mean()*100:.1f}%",
            "mean_causal_rank": f"{grp['mean_causal_rank'].mean():.1f} +/- {grp['mean_causal_rank'].std():.1f}",
            "mean_runtime_sec": f"{grp['runtime_sec'].mean():.1f}s",
        })
    df_summary = pd.DataFrame(summary_rows)

    print(f"\n=== SUMMARY | Run ID: {run_id} | order={order} model={model_type} ===", flush=True)
    print(df_summary.to_string(index=False), flush=True)

    suffix = f"order{order}_{model_type}"
    raw_path = f"reports/benchmarks/baseline_comparison_{suffix}_5seed.csv"
    summary_path = f"reports/benchmarks/baseline_summary_{suffix}_5seed.csv"

    os.makedirs("reports/benchmarks", exist_ok=True)
    df_raw.to_csv(raw_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    # Hard integrity assertions
    raw_mtime = os.path.getmtime(raw_path)
    summary_mtime = os.path.getmtime(summary_path)
    assert raw_mtime >= script_start_time, f"Integrity Failure: {raw_path} was not updated!"
    assert summary_mtime >= script_start_time, f"Integrity Failure: {summary_path} was not updated!"

    raw_hash = hashlib.sha256(open(raw_path, "rb").read()).hexdigest()
    summary_hash = hashlib.sha256(open(summary_path, "rb").read()).hexdigest()

    print(f"\n[INTEGRITY VERIFIED] Run ID: {run_id}", flush=True)
    print(f"  Raw CSV:     {raw_path} | SHA-256: {raw_hash}", flush=True)
    print(f"  Summary CSV: {summary_path} | SHA-256: {summary_hash}", flush=True)

    return df_raw, df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--model_type", type=str, default="additive",
                         choices=["additive", "multiplicative", "threshold", "xor"])
    parser.add_argument("--n_snps", type=int, default=1000)
    parser.add_argument("--n_samples", type=int, default=1600)
    parser.add_argument("--heritability", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_partitions", type=int, default=6)
    args = parser.parse_args()

    run_fair_benchmark(
        n_snps=args.n_snps, n_samples=args.n_samples, order=args.order,
        model_type=args.model_type, heritability=args.heritability,
        epochs=args.epochs, num_partitions=args.num_partitions,
    )
