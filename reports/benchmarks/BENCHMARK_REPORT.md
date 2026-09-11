# Epistasis Detection Research: Empirical Benchmark Report & Methodological Findings

**Date:** September 10, 2026  
**Hardware Platform:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 12.5), Python 3.12.2, PyTorch 2.6.0  
**Repository Branch:** `main` | Commit: `8823919`  

---

## 1. Methodological Integrity & Root Cause Bug Fixes

Before running multi-seed benchmarks, three distinct methodological defects in initial evaluation routines were identified, mathematically diagnosed, and corrected:

1. **Early Stopping on Validation Metric (ROC-AUC / $R^2$), not Raw Loss (`f8035cb`):**
   - *Defect:* Binary Cross-Entropy (BCE) loss on noisy genomic data hovers near $\ln(2) \approx 0.693$ even as ranking capability sharpens ($0.52 \rightarrow 0.77$ AUC). Early stopping on raw loss with `patience=5` caused premature termination at epoch 5–8.
   - *Fix:* Configured early stopping and model checkpoint selection to track primary validation metrics (ROC-AUC for classification, $R^2$ for continuous regression).
2. **Strict Held-Out Evaluation of ACAT Attributions (`c18beae`):**
   - *Defect:* Attentive Class Activation Tokens (ACAT) and detection metrics were previously evaluated on `data_dict["X"]` (the entire dataset, including 70% training samples), introducing memorization leakage.
   - *Fix:* Restricted all ACAT gradient and attention scoring strictly to the held-out test split (`splits["test"]`).
3. **EMA Smoothed Metric Selection & Warmup Guard (`54e750b`):**
   - *Defect:* On small validation splits ($N_{\text{val}}=240$), sample variance ($\text{SE} \approx \pm 0.04$) creates single-epoch AUC lottery spikes (e.g. random epoch-1 weights scoring 0.535), locking the model checkpoint into random weights.
   - *Fix:* Implemented Exponential Moving Average ($\beta=0.7$) validation metric tracking with a 5-epoch warmup guard.

---

## 2. Fair 5-Seed Baseline Comparisons at Scale

### A. Order-2 Additive Epistasis ($N=1,000$ SNPs, $n=1,600$ samples, 5 seeds)
`reports/benchmarks/baseline_summary_1000snp_5seed.csv`

| Model / Architecture | Detection Power (@ Top 5%) | Mean Causal Rank / 1000 | Mean Runtime / Rep |
|---|---|---|---|
| **XGBoost** | **100.0%** (5/5) | **1.5 ± 0.0** | **0.4s** |
| **Random Forest** | **100.0%** (5/5) | **1.5 ± 0.0** | **0.2s** |
| **DeepCOMBI (Dense MLP + Saliency)** | **100.0%** (5/5) | **1.5 ± 0.0** | **1.5s** |
| **Partitioned Transformer ($P=6$)** | **20.0%** (1/5) | **125.0 ± 160.7** | **38.8s** |
| **Interaction Logistic (Top 100 Variance)** | **20.0%** (1/5) | **438.4 ± 257.5** | **0.2s** |
| **MDR (Top 100 Variance Filtered)** | **20.0%** (1/5) | **435.0 ± 253.4** | **6.9s** |

---

### B. Order-2 XOR Epistasis ($N=1,000$ SNPs, $n=1,600$ samples, 5 seeds)
`reports/benchmarks/baseline_summary_order2_xor_5seed.csv`  
*Pure non-linear interaction with zero marginal main effect.*

| Model / Architecture | Detection Power (@ Top 5%) | Mean Causal Rank / 1000 | Mean Runtime / Rep |
|---|---|---|---|
| **XGBoost** | **100.0%** (5/5) | **1.5 ± 0.0** | **0.4s** |
| **Random Forest** | **100.0%** (5/5) | **1.5 ± 0.0** | **0.2s** |
| **DeepCOMBI (Dense MLP + Saliency)** | **100.0%** (5/5) | **2.0 ± 0.7** | **1.5s** |
| **MDR (order=2, exhaustive, $\binom{1000}{2}=499,500$ pairs)** | **100.0%** (5/5) | **1.5 ± 0.0** | **14.4s** |
| **Partitioned Transformer ($P=6$)** | **0.0%** (0/5) | **171.5 ± 106.6** | **37.5s** |
| **Interaction Logistic (Top 100 Variance)** | **0.0%** (0/5) | **556.3 ± 207.9** | **0.1s** |

*Note on classical baselines:* When evaluated without candidate pre-filtering, exhaustive MDR evaluates all 499,500 pairs in 14.4s via vectorized base-3 indexing, detecting the exact causal pair on 5/5 seeds with 100% power (Rank 1.5). Tree ensembles (XGBoost/RF) and DeepCOMBI also achieve 100% power in $<1.5$s. Interaction Logistic fails because its variance pre-filter eliminates zero-marginal XOR SNPs.

---

### C. Order-3 Additive Epistasis ($N=1,000$ SNPs, $n=1,600$ samples, 5 seeds)
`reports/benchmarks/baseline_summary_order3_additive_5seed.csv`  
*3-way multi-locus interaction across 1,000 SNPs.*

| Model / Architecture | Detection Power (@ Top 5%) | Mean Causal Rank / 1000 | Mean Runtime / Rep |
|---|---|---|---|
| **XGBoost** | **100.0%** (5/5) | **2.2 ± 0.3** | **0.6s** |
| **Random Forest** | **100.0%** (5/5) | **2.0 ± 0.0** | **0.2s** |
| **DeepCOMBI (Dense MLP + Saliency)** | **100.0%** (5/5) | **2.0 ± 0.0** | **1.6s** |
| **Partitioned Transformer ($P=6$)** | **20.0%** (1/5) | **142.4 ± 126.7** | **38.6s** |
| **MDR (order=3, candidate-filtered)** | **0.0%** (0/5) | **561.9 ± 0.3** | **5.4s** |
| **Interaction Logistic (pairwise terms only)** | **0.0%** (0/5) | **566.2 ± 42.8** | **0.3s** |

---

### D. Order-3 Additive Epistasis at Exhaustive Tractability Scale ($N=100$ SNPs, $n=500$ samples, 5 seeds)
`reports/benchmarks/baseline_summary_order3_additive_5seed.csv` (at $N=100$)  
*Combos $\binom{100}{3} = 161,700 < 2,000,000 \implies$ MDR runs fully exhaustive.*

| Model / Architecture | Detection Power (@ Top 5%) | Mean Causal Rank / 100 | Mean Runtime / Rep |
|---|---|---|---|
| **XGBoost** | **100.0%** (5/5) | **2.0 ± 0.0** | **0.1s** |
| **Random Forest** | **100.0%** (5/5) | **2.0 ± 0.0** | **0.2s** |
| **Interaction Logistic** | **100.0%** (5/5) | **2.0 ± 0.0** | **0.0s** |
| **DeepCOMBI (Dense MLP + Saliency)** | **100.0%** (5/5) | **2.0 ± 0.0** | **0.7s** |
| **MDR (order=3, exhaustive, 161,700 combos)** | **80.0%** (4/5) | **3.8 ± 4.0** | **2.6s** |
| **Partitioned Transformer ($P=6$)** | **0.0%** (0/5) | **29.9 ± 24.8** | **6.8s** |

---

## 3. Scientific Analysis & Architectural Takeaway (Path B)

1. **Why Dense MLPs and Tree Ensembles Dominate at $\le 1,000$ SNPs:**
   - At $N \le 1,000$ SNPs, all features fit simultaneously into GPU VRAM / CPU cache as a single dense matrix or decision tree split candidate set.
   - Dense models (XGBoost, RF, DeepCOMBI) isolate causal loci in $<1.5$ seconds with 100% detection power across additive, XOR, and 3-way interactions.
2. **The Disjoint Partitioning Tradeoff:**
   - Arbitrarily bucketing $N=1000$ SNPs into $P=6$ partitions isolates the interacting loci into only 1 of $\binom{P}{2}=15$ combinations. On small sample sizes ($n=1600$), learning to route attention to the active combination is strictly harder than dense feature evaluation.
3. **Where the Transformer Regime Lies:**
   - Classical regression / MDR explodes combinatorially at genome scale ($O(S^2)$ for 50,000 SNPs $\rightarrow 1.25 \times 10^9$ pairs; $O(S^3) \rightarrow 2.08 \times 10^{13}$ triplets).
   - The sparse partitioned transformer is an **asymptotic architecture for combinatorially intractable genome scale ($N \ge 20,000 - 500,000$ SNPs)** where dense matrix layers exceed GPU memory and exhaustive search is mathematically impossible.

---

## 4. Multi-Replicate Partition Scaling Sweep (20 GPU Runs)

**Configuration:** 1,000 SNPs, $n=1,600$ samples, Gated Combination Pooling, EMA smoothing, 5 seeds per partition count.

### Summary Table (`reports/benchmarks/partition_scaling_validation.csv`)

| $P$ | Combos ($\binom{P}{2}$) | SNPs / Comb | Mean Val AUC | Mean Causal Rank | Detection Power | Mean Runtime |
|---|---|---|---|---|---|---|
| **6** | 15 | 333 | $0.5540 \pm 0.0553$ | $298.6 \pm 179.0$ | **20.0%** | 41.8s |
| **10** | 45 | 200 | $0.5735 \pm 0.0444$ | $329.8 \pm 237.9$ | **20.0%** | 107.4s |
| **15** | 105 | 133 | $0.5499 \pm 0.0543$ | $195.2 \pm 125.4$ | **20.0%** | 377.5s |
| **20** | 190 | 100 | $0.5985 \pm 0.0236$ | $205.2 \pm 178.8$ | **20.0%** | 645.6s |

---

## 5. Phase 2/3: 3-Way Pharmacogenomics Epistasis ($IC_{50}$ Continuous Phenotype)

**Setup:** $n=800$ cell lines, $N=50$ candidate pharmacogenomic genes, 3-way continuous interaction (`CYP1A1`, `CYP1A2`, `CYP2B6`), $h^2=0.35$, 5 independent replicates (`reports/pharmacogenomics/pharmacogenomics_summary.csv`).

### Multi-Replicate Results

| Replicate | Seed | Best Val $R^2$ | Causal Genes | Causal Ranks | Mean Causal Rank / 50 |
|---|---|---|---|---|---|
| **1** | 42 | 0.2885 | `['CYP1A1', 'CYP1A2', 'CYP2B6']` | `[3, 2, 1]` | **2.0** |
| **2** | 101 | 0.0459 | `['CYP1A1', 'CYP1A2', 'CYP2B6']` | `[6, 3, 1]` | **3.3** |
| **3** | 202 | -0.0001 | `['CYP1A1', 'CYP1A2', 'CYP2B6']` | `[19, 49, 10]` | **26.0** |
| **4** | 303 | 0.1214 | `['CYP1A1', 'CYP1A2', 'CYP2B6']` | `[2, 6, 1]` | **3.0** |
| **5** | 404 | 0.0012 | `['CYP1A1', 'CYP1A2', 'CYP2B6']` | `[25, 7, 13]` | **15.0** |

**Aggregate Mean Causal Rank:** **$9.9 \pm 10.5$ / 50 genes**.  
**Biological Validation:** Live STRING API query confirmed **29 protein-protein / functional interactions** among top candidate genes (`EGFR <---> CYP1A1` score 0.592, `KRAS <---> PTEN` score 0.936, `KRAS <---> EGFR` score 0.997).
