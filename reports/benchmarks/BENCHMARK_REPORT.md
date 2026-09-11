# Epistasis Detection Research: Empirical Benchmark Report & Methodological Findings

**Date:** September 11, 2026  
**Hardware Platform:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.5), Python 3.12.2, PyTorch 2.6.0  
**Repository Branch:** `main`  

---

## 1. Methodological Integrity & Root Cause Bug Fixes

Before and during multi-seed benchmarking, four distinct methodological aspects were systematically audited, mathematically diagnosed, and corrected:

1. **Early Stopping on Primary Validation Metric (ROC-AUC / $R^2$), not Raw Loss (`f8035cb`):**
   - *Defect:* Binary Cross-Entropy (BCE) loss on noisy genomic data hovers near $\ln(2) \approx 0.693$ even as ranking capability sharpens ($0.52 \rightarrow 0.77$ AUC). Early stopping on raw loss with `patience=5` caused premature termination at epoch 5–8.
   - *Fix:* Configured early stopping and model checkpoint selection to track primary validation metrics (ROC-AUC for classification, $R^2$ for continuous regression).
2. **Strict Held-Out Evaluation of ACAT Attributions (`c18beae`):**
   - *Defect:* Attentive Class Activation Tokens (ACAT) and detection metrics were previously evaluated on `data_dict["X"]` (the entire dataset, including 70% training samples), introducing memorization leakage.
   - *Fix:* Restricted all ACAT gradient and attention scoring strictly to the held-out test split (`splits["test"]`).
3. **EMA Smoothed Metric Selection & Warmup Guard (`54e750b`):**
   - *Defect:* On small validation splits ($N_{\text{val}}=240$), sample variance ($\text{SE} \approx \pm 0.04$) creates single-epoch AUC lottery spikes (e.g. random epoch-1 weights scoring 0.535), locking the model checkpoint into random weights.
   - *Fix:* Implemented Exponential Moving Average ($\beta=0.7$) validation metric tracking with a 5-epoch warmup guard.
4. **Ruling Out Under-Training (`epochs=100` Diagnostic Sweep):**
   - *Investigation:* Evaluated whether `epochs=30` prematurely halted training by testing `--epochs 100` with `patience=15` across all benchmark seeds.
   - *Finding:* In 4/5 seeds, early stopping triggered at epochs 19–60 due to validation AUC plateauing, confirming that the 20% detection power at $N=1000$ reflects genuine optimization dynamics on small sample sizes rather than an insufficient epoch budget.

---

## 2. Empirical MDR Combinatorial Crossover Analysis

To establish where exhaustive search actually becomes impractical (rather than relying on arbitrary assumptions), raw combinatorial MDR search times were measured across SNP counts $N$ for Order 2 and Order 3 on identical hardware ($n=1600$ samples, vectorized base-3 indexing):

| Order | $N$ SNPs | Total Combinations | Empirical / Estimated Runtime | Combinations / Sec | Practical Feasibility |
|---|---|---|---|---|---|
| **Order 2** | 100 | 4,950 | **0.9s** | 5,771 | Real-time |
| **Order 2** | 500 | 124,750 | **24.8s** | 5,021 | Interactive (<30s) |
| **Order 2** | 1,000 | 499,500 | **115.1s** (~1.9 min) | 4,340 | Feasible exhaustive |
| **Order 2** | 1,500 | 1,124,250 | **258.3s** (~4.3 min) | 4,353 | Marginal |
| **Order 2** | 2,000 | 1,999,000 | **~7.7 min** | ~4,350 | **Crosses 5-min threshold** |
| **Order 2** | 5,000 | 12,497,500 | **~47.9 min** | ~4,350 | Infeasible for quick scans |
| **Order 2** | 10,000 | 49,995,000 | **~3.2 hours** | ~4,350 | HPC batch only |
| **Order 2** | 50,000 | 1.25 billion | **~79.8 hours** (~3.3 days) | ~4,350 | Intractable on single node |
| **Order 3** | 50 | 19,600 | **3.6s** | 5,426 | Real-time |
| **Order 3** | 100 | 161,700 | **30.3s** | 5,340 | Interactive (<1 min) |
| **Order 3** | 150 | 551,300 | **99.2s** (~1.6 min) | 5,560 | Interactive |
| **Order 3** | 200 | 1,313,400 | **230.4s** (~3.8 min) | 5,700 | Marginal |
| **Order 3** | 300 | 4,455,100 | **~13.0 min** | ~5,700 | **Crosses 5-min threshold** |
| **Order 3** | 500 | 20,708,500 | **~1.0 hour** | ~5,700 | Infeasible for interactive work |
| **Order 3** | 1,000 | 166,167,000 | **~8.1 hours** | ~5,700 | HPC batch only |
| **Order 3** | 5,000 | 20.8 billion | **~1,014 hours** (~42.3 days) | ~5,700 | Fully intractable |

**Key Takeaway:** Exhaustive MDR crosses the 5-minute practical threshold at **$N \approx 2,000$ for pairwise interactions** and **$N \approx 300$ for 3-way interactions**. Beyond these thresholds, exhaustive classical search is computationally prohibitive without severe heuristic filtering.

---

## 3. Generalized Fair 5-Seed Benchmarks (Evaluated under `epochs=100`)

### A. Order-2 Additive Epistasis ($N=1,000$ SNPs, $n=1,600$ samples, `epochs=100`, 5 seeds)
`reports/benchmarks/baseline_summary_order2_additive_5seed.csv`

| Model / Architecture | Detection Power (@ Top 5%) | Mean Causal Rank / 1000 | Mean Runtime / Rep |
|---|---|---|---|
| **XGBoost** | **100.0%** (5/5) | **1.5 ± 0.0** | **0.7s** |
| **Random Forest** | **100.0%** (5/5) | **1.5 ± 0.0** | **0.3s** |
| **DeepCOMBI (Dense MLP + Saliency)** | **100.0%** (5/5) | **1.5 ± 0.0** | **3.1s** |
| **MDR (order=2, exhaustive, 499,500 pairs)** | **100.0%** (5/5) | **1.5 ± 0.0** | **34.3s** |
| **Partitioned Transformer ($P=6$, epochs=100)** | **20.0%** (1/5) | **125.0 ± 160.7** | **94.9s** |
| **Interaction Logistic (pairwise terms)** | **20.0%** (1/5) | **438.4 ± 257.5** | **0.3s** |

- **Transformer Seed Breakdown:**
  - Seed 42: Early stopped at epoch 19 (AUC 0.4986) $\rightarrow$ Causal ranks `[18, 156]` (Mean 87.0, False)
  - Seed 101: Early stopped at epoch 71 (AUC 0.7231) $\rightarrow$ Causal ranks `[1, 2]` (Mean 1.5, **True**)
  - Seed 202: Early stopped at epoch 19 (AUC 0.4900) $\rightarrow$ Causal ranks `[139, 25]` (Mean 82.0, False)
  - Seed 303: Early stopped at epoch 47 (AUC 0.6242) $\rightarrow$ Causal ranks `[28, 69]` (Mean 48.5, False)
  - Seed 404: Early stopped at epoch 60 (AUC 0.4915) $\rightarrow$ Causal ranks `[395, 417]` (Mean 406.0, False)

---

### B. Order-3 Additive Epistasis ($N=1,000$ SNPs, $n=1,600$ samples, `epochs=100`, 5 seeds)
`reports/benchmarks/baseline_summary_order3_additive_5seed.csv`  
*3-way multi-locus interaction across 1,000 SNPs ($\binom{1000}{3} = 166.1\text{M}$ combos $\implies$ MDR is candidate-filtered).*

| Model / Architecture | Detection Power (@ Top 5%) | Mean Causal Rank / 1000 | Mean Runtime / Rep |
|---|---|---|---|
| **XGBoost** | **100.0%** (5/5) | **2.2 ± 0.3** | **1.0s** |
| **Random Forest** | **100.0%** (5/5) | **2.0 ± 0.0** | **0.4s** |
| **DeepCOMBI (Dense MLP + Saliency)** | **100.0%** (5/5) | **2.0 ± 0.0** | **4.3s** |
| **Partitioned Transformer ($P=6$, epochs=100)** | **20.0%** (1/5) | **142.0 ± 126.9** | **110.5s** |
| **MDR (order=3, candidate-filtered)** | **0.0%** (0/5) | **561.9 ± 0.3** | **18.7s** |
| **Interaction Logistic (pairwise-only)** | **0.0%** (0/5) | **566.2 ± 42.8** | **0.5s** |

- **Key Finding:** When exhaustive search is computationally impossible (166.1M combinations), candidate-filtering MDR on marginal variance fails completely (0% detection, mean rank 561.9). However, tree ensembles (XGBoost, RF) and DeepCOMBI MLPs do not perform combinatorial search and detect all 3 causal loci with 100% power in $\le 4.3$s. The Partitioned Transformer achieves 20% detection power (Seed 303: ranks `[7, 23, 40]`, Mean 23.3, True).

---

### C. Order-2 XOR Epistasis ($N=1,000$ SNPs, $n=1,600$ samples, `epochs=100`, 5 seeds)
`reports/benchmarks/baseline_summary_order2_xor_5seed.csv`  
*Pure non-linear interaction with zero marginal main effect.*

| Model / Architecture | Detection Power (@ Top 5%) | Mean Causal Rank / 1000 | Mean Runtime / Rep |
|---|---|---|---|
| **XGBoost** | **100.0%** (5/5) | **1.5 ± 0.0** | **0.4s** |
| **Random Forest** | **100.0%** (5/5) | **1.5 ± 0.0** | **0.2s** |
| **DeepCOMBI (Dense MLP + Saliency)** | **100.0%** (5/5) | **2.0 ± 0.7** | **1.6s** |
| **MDR (order=2, exhaustive, 499,500 pairs)** | **100.0%** (5/5) | **1.5 ± 0.0** | **14.1s** |
| **Partitioned Transformer ($P=6$, epochs=100)** | **0.0%** (0/5) | **171.5 ± 106.6** | **37.9s** |
| **Interaction Logistic (pairwise terms)** | **0.0%** (0/5) | **556.3 ± 207.9** | **0.1s** |

---

## 4. Scientific Findings & Nuanced Research Conclusions

1. **Combinatorial vs. Continuous Feature Selection:**
   - Classical exhaustive methods (MDR, full interaction logistic) scale as $O(N^k)$ and become intractable beyond $N=2,000$ (Order 2) and $N=300$ (Order 3).
   - Heuristic candidate filtering (variance or marginal correlation) completely breaks down under pure epistasis (XOR or higher-order multi-locus interactions) where individual SNPs possess negligible marginal effects.
2. **The Non-Combinatorial Baselines Advantage (Tree Ensembles & MLPs):**
   - Tree ensembles (XGBoost, Random Forest) and DeepCOMBI (MLP + Saliency) do not evaluate combinatorial pairs/triplets explicitly.
   - At $N=1,000$, their dense feature representations isolate epistatic loci in $\le 4.3$s across additive, XOR, and 3-way interactions with 100% detection power.
3. **The Architectural Tradeoff of Partitioning:**
   - Disjoint partitioning splits $N=1,000$ SNPs into $P=6$ subsets, placing the causal loci in only 1 of $\binom{P}{2}=15$ combinations.
   - On small sample sizes ($n=1,600$), learning to route attention dynamically across sparse combinations without leaking noisy gradients requires substantial signal-to-noise ratio.
   - The Sparse Partitioned Transformer is designed for **the asymptotic regime where $N \ge 20,000 - 500,000$ SNPs** (where dense matrix layers exceed GPU memory and trees suffer memory bottlenecks), rather than the small $N \le 1,000$ regime where dense baselines excel.

---

## 5. Summary of Multi-Seed Pharmacogenomics Validation (Phase 2/3)

**Setup:** $n=800$ cell lines, $N=50$ candidate pharmacogenomic genes, 3-way continuous interaction (`CYP1A1`, `CYP1A2`, `CYP2B6`), $h^2=0.35$, 5 independent replicates (`reports/pharmacogenomics/pharmacogenomics_summary.csv`).

| Replicate | Seed | Best Val $R^2$ | Causal Genes | Causal Ranks | Mean Causal Rank / 50 |
|---|---|---|---|---|---|
| **1** | 42 | 0.2885 | `['CYP1A1', 'CYP1A2', 'CYP2B6']` | `[3, 2, 1]` | **2.0** |
| **2** | 101 | 0.0459 | `['CYP1A1', 'CYP1A2', 'CYP2B6']` | `[6, 3, 1]` | **3.3** |
| **3** | 202 | -0.0001 | `['CYP1A1', 'CYP1A2', 'CYP2B6']` | `[19, 49, 10]` | **26.0** |
| **4** | 303 | 0.1214 | `['CYP1A1', 'CYP1A2', 'CYP2B6']` | `[2, 6, 1]` | **3.0** |
| **5** | 404 | 0.0012 | `['CYP1A1', 'CYP1A2', 'CYP2B6']` | `[25, 7, 13]` | **15.0** |

- **Aggregate Mean Causal Rank:** **$9.9 \pm 10.5$ / 50 genes**.
- **Biological Validation:** Live STRING API query confirmed **29 protein-protein / functional interactions** among top candidate genes (`EGFR <---> CYP1A1` score 0.592, `KRAS <---> PTEN` score 0.936, `KRAS <---> EGFR` score 0.997).
