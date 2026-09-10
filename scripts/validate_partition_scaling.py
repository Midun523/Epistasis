"""Validation script for Multi-Replicate Partition Scaling across P in [6, 10, 15, 20] on 1000 SNPs."""

import time
import torch
import numpy as np
import pandas as pd
from epistasis.data.gametes_simulator import EpistasisSimulator
from epistasis.training import run_full_pipeline

def run_validation(n_replicates: int = 5, epochs: int = 30):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running Multi-Replicate ({n_replicates} reps) Partition Scaling on {device.upper()} ===")
    
    partition_counts = [6, 10, 15, 20]
    seeds = [42, 101, 202, 303, 404][:n_replicates]
    
    raw_records = []
    summary_records = []
    
    for p in partition_counts:
        n_combs = p * (p - 1) // 2
        snps_per_comb = int(2 * (1000 / p))
        print(f"\n========================================================")
        print(f"--> Testing P={p} ({n_combs} combinations, ~{snps_per_comb} SNPs/comb, {n_replicates} replicates)")
        print(f"========================================================")
        
        rep_ranks = []
        rep_aucs = []
        rep_detected = []
        rep_times = []
        
        for rep_idx, seed in enumerate(seeds):
            # Seed PRNGs for complete reproducibility per replicate
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            
            # Generate simulated dataset
            sim = EpistasisSimulator(
                n_snps=1000,
                n_samples=1600,
                order=2,
                maf=0.2,
                heritability=0.4,
                model_type="additive",
                random_state=seed,
            )
            data = sim.generate_case_control()
            causal = data["causal_indices"]
            
            start_time = time.time()
            res = run_full_pipeline(
                data_dict=data,
                epochs=epochs,
                num_partitions=p,
                combination_size=2,
                sparsity_ratio=0.85,
                aggregation="gated",
                warmup_epochs=5,
                ema_decay=0.7,
                random_state=seed,
                device=device,
            )
            elapsed = time.time() - start_time
            
            d_eval = res["detection_eval"]
            best_auc = max(res["history"]["val_metric"])
            smoothed_auc = max(res["history"]["val_metric_smoothed"])
            mean_rank = d_eval["mean_causal_rank"]
            detected = d_eval["detected_all_top_5pct"]
            
            rep_ranks.append(mean_rank)
            rep_aucs.append(best_auc)
            rep_detected.append(detected)
            rep_times.append(elapsed)
            
            print(
                f"  [Rep {rep_idx+1}/{n_replicates} | Seed {seed}] "
                f"Time: {elapsed:5.1f}s | "
                f"Best Val AUC: {best_auc:.4f} (Smoothed: {smoothed_auc:.4f}) | "
                f"Causal Ranks: {str(d_eval['causal_ranks']):<14} | "
                f"Mean Rank: {mean_rank:5.1f} / 1000 | "
                f"Detected Top 5%: {detected}"
            )
            
            raw_records.append({
                "partitions": p,
                "combinations": n_combs,
                "replicate": rep_idx + 1,
                "seed": seed,
                "best_val_auc": round(best_auc, 4),
                "smoothed_val_auc": round(smoothed_auc, 4),
                "causal_ranks": str(d_eval["causal_ranks"]),
                "mean_causal_rank": round(mean_rank, 1),
                "detected_top_5pct": detected,
                "runtime_sec": round(elapsed, 1),
            })
            
        summary_records.append({
            "partitions": p,
            "combinations": n_combs,
            "snps_per_comb": snps_per_comb,
            "mean_val_auc": f"{np.mean(rep_aucs):.4f} +/- {np.std(rep_aucs):.4f}",
            "mean_causal_rank": f"{np.mean(rep_ranks):.1f} +/- {np.std(rep_ranks):.1f}",
            "detection_power": f"{np.mean(rep_detected)*100:.1f}%",
            "mean_runtime_sec": f"{np.mean(rep_times):.1f}s",
        })
        
    df_raw = pd.DataFrame(raw_records)
    df_summary = pd.DataFrame(summary_records)
    
    print("\n\n=== FINAL MULTI-REPLICATE SUMMARY TABLE ===")
    print(df_summary.to_string(index=False))
    
    df_raw.to_csv("reports/benchmarks/partition_scaling_raw_replicates.csv", index=False)
    df_summary.to_csv("reports/benchmarks/partition_scaling_validation.csv", index=False)
    print("\nSaved raw replicates to reports/benchmarks/partition_scaling_raw_replicates.csv")
    print("Saved summary table to reports/benchmarks/partition_scaling_validation.csv")

if __name__ == "__main__":
    run_validation(n_replicates=5, epochs=30)
