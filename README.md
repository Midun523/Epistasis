# Higher-Order Epistasis Detection: A Transformer-Based Framework

A high-performance deep learning framework for detecting higher-order epistatic interactions in genome-wide association studies (GWAS) and continuous pharmacogenomic phenotypes.

## Overview
- **Key-Vector Partitioned Transformer**: Scalable attention mechanism over SNP key partitions, avoiding combinatorial $O(S^k)$ explosion.
- **Attentive Class Activation Tokens (ACAT)**: Dual interpretability combining partitioned multi-head attention weights and input token gradients.
- **Continuous Phenotype Extension**: Regression head and attribution for cancer cell line drug sensitivity ($IC_{50}$ / AUC).
- **Comprehensive Benchmark Suite**: Side-by-side comparison against MDR, Logistic Regression with interaction terms, XGBoost, Random Forest, and DeepCOMBI (MLP + LRP).
- **Biological Validation**: Integrated queries to STRING database API and synthetic lethality databases.

## Installation
```bash
pip install -e .
```

## Running Simulations & Benchmarks
```bash
# Run simulation sweep across epistasis models
python scripts/run_simulation_experiment.py --order 2 --model xor --maf 0.2 --h2 0.2 --replicates 5

# Run baseline comparative benchmark
python scripts/run_baseline_benchmark.py --order 2 --model xor --replicates 3

# Run pharmacogenomic continuous drug response analysis
python scripts/run_pharmacogenomics.py --cell_lines 800 --genes 50 --order 3
```
