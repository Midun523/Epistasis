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

## 2. Fair 5-Seed Baseline Comparison on 1,000 SNPs

**Experimental Setup:** $N=1,000$ SNPs, $n=1,600$ samples (Train: 1120, Val: 240, Test: 240), 2-way Additive Epistasis ($h^2=0.4, \text{MAF}=0.2$), 5 independent random seeds (`[42, 101, 202, 303, 404]`).

### Summary Table (`reports/benchmarks/baseline_summary_1000snp_5seed.csv`)

| Model / Architecture | Detection Power (@ Top 5%) | Mean Causal Rank / 1000 | Mean Runtime / Rep |
|---|---|---|---|
| **XGBoost** | **100.0%** (5/5) | **1.5 ± 0.0** | **0.4s** |
| **Random Forest** | **100.0%** (5/5) | **1.5 ± 0.0** | **0.2s** |
| **DeepCOMBI (Dense MLP + Saliency)** | **100.0%** (5/5) | **1.5 ± 0.0** | **1.5s** |
| **Partitioned Transformer ($P=6$)** | **20.0%** (1/5) | **125.0 ± 160.7** | **38.8s** |
| **Interaction Logistic (Top 100 Variance)** | **20.0%** (1/5) | **438.4 ± 257.5** | **0.2s** |
| **MDR (Top 100 Variance Filtered)** | **20.0%** (1/5) | **435.0 ± 253.4** | **6.9s** |

### Per-Replicate Breakdown (`reports/benchmarks/baseline_comparison_1000snp_5seed.csv`)

```text
Seed 42:
  XGBoost:                Causal: [1, 2]     | Mean Rank:   1.5 | Top 5%: True  (0.4s)
  Random Forest:          Causal: [2, 1]     | Mean Rank:   1.5 | Top 5%: True  (0.2s)
  DeepCOMBI MLP:          Causal: [1, 2]     | Mean Rank:   1.5 | Top 5%: True  (3.0s)
  Partitioned Transformer:Causal: [18, 156]  | Mean Rank:  87.0 | Top 5%: False (29.1s)
  Interaction Logistic:   Causal: [1000, 1]  | Mean Rank: 500.5 | Top 5%: False (0.2s)
  MDR:                    Causal: [1000, 2]  | Mean Rank: 501.0 | Top 5%: False (6.6s)

Seed 101:
  XGBoost:                Causal: [2, 1]     | Mean Rank:   1.5 | Top 5%: True  (0.4s)
  Random Forest:          Causal: [2, 1]     | Mean Rank:   1.5 | Top 5%: True  (0.2s)
  DeepCOMBI MLP:          Causal: [2, 1]     | Mean Rank:   1.5 | Top 5%: True  (1.2s)
  Partitioned Transformer:Causal: [1, 2]     | Mean Rank:   1.5 | Top 5%: True  (45.9s)
  Interaction Logistic:   Causal: [2, 1]     | Mean Rank:   1.5 | Top 5%: True  (0.1s)
  MDR:                    Causal: [1, 2]     | Mean Rank:   1.5 | Top 5%: True  (7.0s)

Seed 202:
  XGBoost:                Causal: [2, 1]     | Mean Rank:   1.5 | Top 5%: True  (0.4s)
  Random Forest:          Causal: [2, 1]     | Mean Rank:   1.5 | Top 5%: True  (0.2s)
  DeepCOMBI MLP:          Causal: [1, 2]     | Mean Rank:   1.5 | Top 5%: True  (1.1s)
  Partitioned Transformer:Causal: [139, 25]  | Mean Rank:  82.0 | Top 5%: False (29.1s)

Seed 303:
  XGBoost:                Causal: [2, 1]     | Mean Rank:   1.5 | Top 5%: True  (0.4s)
  Random Forest:          Causal: [2, 1]     | Mean Rank:   1.5 | Top 5%: True  (0.2s)
  DeepCOMBI MLP:          Causal: [2, 1]     | Mean Rank:   1.5 | Top 5%: True  (1.1s)
  Partitioned Transformer:Causal: [28, 69]   | Mean Rank:  48.5 | Top 5%: False (44.8s)

Seed 404:
  XGBoost:                Causal: [2, 1]     | Mean Rank:   1.5 | Top 5%: True  (0.3s)
  Random Forest:          Causal: [1, 2]     | Mean Rank:   1.5 | Top 5%: True  (0.2s)
  DeepCOMBI MLP:          Causal: [2, 1]     | Mean Rank:   1.5 | Top 5%: True  (1.2s)
  Partitioned Transformer:Causal: [395, 417] | Mean Rank: 406.0 | Top 5%: False (44.9s)
```

---

## 3. Scientific Analysis & Architectural Takeaway (Path B)

1. **Why Dense MLPs and Tree Ensembles Dominate at 1,000 SNPs:**
   - At $N=1,000$ SNPs, all 1,000 features can easily fit simultaneously into a single GPU/CPU matrix multiplication or decision tree node split.
   - Dense models (XGBoost, RF, DeepCOMBI) backpropagate gradients or evaluate split gains across all features concurrently, isolating the 2 causal SNPs instantly ($<1.5$ seconds, Rank 1.5).
2. **The Partitioning Tradeoff:**
   - Partitioning arbitrary features into $P$ buckets creates $\binom{P}{2}$ combinations, placing the interacting pair into only 1 combination while the remaining combinations process noise.
   - For un-prioritized 1,000-SNP genomes, partitioning adds optimization overhead without offering sample efficiency gains over dense models.
3. **Where the Transformer Regime Lies:**
   - Classical regression / MDR explodes combinatorially at genome scale ($O(S^k)$ for order-$k$ search over $50,000$ SNPs $\rightarrow 1.25 \times 10^9$ pairs).
   - The sparse partitioned transformer's value is in **Higher-Order Continuous Interactions ($k \ge 3$)** and **Candidate Panel Screening** where non-linear attention gating discovers multi-way epistasis without explicit feature matrix expansion.

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
