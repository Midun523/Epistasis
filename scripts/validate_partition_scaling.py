"""Validation script for Partition Scaling across P in [6, 10, 15, 20, 30] on 1000 SNPs."""

import time
import torch
import pandas as pd
from epistasis.data.gametes_simulator import EpistasisSimulator
from epistasis.training import run_full_pipeline

def run_validation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Running Partition Scaling Validation on {device.upper()} ===")
    
    # Generate 1000-SNP simulated dataset
    sim = EpistasisSimulator(
        n_snps=1000,
        n_samples=1600,
        order=2,
        maf=0.2,
        heritability=0.4,
        model_type="additive",
        random_state=42,
    )
    data = sim.generate_case_control()
    causal = data["causal_indices"]
    print(f"Dataset: N=1000 SNPs, n=1600 samples, Causal Loci={causal}\n")
    
    partition_counts = [6, 10, 15, 20]
    records = []
    
    for p in partition_counts:
        n_combs = p * (p - 1) // 2
        snps_per_comb = int(2 * (1000 / p))
        print(f"--> Testing P={p} ({n_combs} combinations, ~{snps_per_comb} SNPs/comb)...")
        
        start_time = time.time()
        res = run_full_pipeline(
            data_dict=data,
            epochs=50,
            num_partitions=p,
            combination_size=2,
            sparsity_ratio=0.85,
            aggregation="gated",
            device=device,
        )
        elapsed = time.time() - start_time
        
        d_eval = res["detection_eval"]
        best_auc = max(res["history"]["val_metric"])
        
        print(
            f"    [P={p:02d}] Finished in {elapsed:.1f}s | "
            f"Best Val AUC: {best_auc:.4f} | "
            f"Mean Rank: {d_eval['mean_causal_rank']:.1f} / 1000 | "
            f"Ranks: {d_eval['causal_ranks']} | "
            f"Detected Top 5%: {d_eval['detected_all_top_5pct']}"
        )
        
        records.append({
            "partitions": p,
            "combinations": n_combs,
            "snps_per_comb": snps_per_comb,
            "best_val_auc": best_auc,
            "mean_causal_rank": d_eval["mean_causal_rank"],
            "causal_ranks": str(d_eval["causal_ranks"]),
            "detected_top_5pct": d_eval["detected_all_top_5pct"],
            "runtime_sec": round(elapsed, 1),
        })
        
    df = pd.DataFrame(records)
    print("\n=== FINAL VALIDATION SUMMARY TABLE ===")
    print(df.to_string(index=False))
    
    df.to_csv("reports/benchmarks/partition_scaling_validation.csv", index=False)
    print("\nSaved validation summary to reports/benchmarks/partition_scaling_validation.csv")

if __name__ == "__main__":
    run_validation()
