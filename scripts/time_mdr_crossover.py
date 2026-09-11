"""
Times raw MDR combinatorial search cost (no model training, just the
search) across increasing SNP counts, to find where exhaustive search
actually becomes impractical -- not where we've assumed it does.
"""
import time
import numpy as np
from math import comb
from epistasis.models.baselines import MultifactorDimensionalityReduction

def time_mdr_search(n_snps, order, n_samples=1600, timeout_sec=300):
    np.random.seed(0)
    X = np.random.randint(0, 3, size=(n_samples, n_snps))
    y = np.random.randint(0, 2, size=n_samples)

    n_combos = comb(n_snps, order)
    print(f"N={n_snps}, order={order}: {n_combos:,} combinations...", flush=True)

    t0 = time.time()
    mdr = MultifactorDimensionalityReduction(order=order)
    mdr.fit(X, y, candidate_indices=None)  # exhaustive, no filtering
    elapsed = time.time() - t0

    print(f"  -> {elapsed:.1f}s ({n_combos/max(elapsed,0.001):,.0f} combos/sec)", flush=True)
    return elapsed

if __name__ == "__main__":
    configs = [
        (2, [100, 500, 1000, 1500, 2000, 5000, 10000, 20000, 50000]),
        (3, [50, 100, 150, 200, 300, 500, 1000, 2000, 5000]),
    ]
    for order, n_list in configs:
        print(f"\n=== ORDER {order} ===")
        combos_per_sec = 6500.0  # empirical rate
        for n_snps in n_list:
            n_combos = comb(n_snps, order)
            est_seconds = n_combos / combos_per_sec
            if est_seconds > 300:
                print(f"N={n_snps}, order={order}: {n_combos:,} combinations -> Estimated {est_seconds:.1f}s ({est_seconds/60:.1f} min / {est_seconds/3600:.1f} hrs) [Exceeds 5min threshold, skipping exact run]")
                continue
            try:
                elapsed = time_mdr_search(n_snps, order)
                combos_per_sec = n_combos / max(elapsed, 0.001)
                if elapsed > 300:
                    print(f"  Exceeded 5min threshold at N={n_snps}")
            except MemoryError:
                print(f"  MemoryError at N={n_snps}")
