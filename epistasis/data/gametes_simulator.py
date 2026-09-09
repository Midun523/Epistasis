"""Synthetic population simulator for higher-order epistasis GWAS datasets."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from epistasis.data.penetrance import PenetranceCalculator, EpistasisModelType


class EpistasisSimulator:
    """
    Direct and exact synthetic dataset generator for epistatic interactions.
    Simulates balanced case-control or continuous phenotype cohorts with embedded ground-truth epistatic loci.
    """

    def __init__(
        self,
        n_snps: int = 1000,
        n_samples: int = 1600,
        order: int = 2,
        maf: float = 0.2,
        heritability: float = 0.2,
        model_type: EpistasisModelType = "additive",
        causal_indices: Optional[List[int]] = None,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            n_snps: Total number of SNPs (e.g., 1000).
            n_samples: Total number of samples (e.g., 1600 -> 800 cases, 800 controls).
            order: Interaction order k (e.g., 2 to 8).
            maf: Minor allele frequency.
            heritability: Broad-sense heritability H^2.
            model_type: 'additive', 'multiplicative', 'threshold', or 'xor'.
            causal_indices: Indices of interacting SNPs. Default: [0, 1, ..., order-1].
            random_state: Seed for reproducibility.
        """
        self.n_snps = n_snps
        self.n_samples = n_samples
        self.order = order
        self.maf = maf
        self.heritability = heritability
        self.model_type = model_type
        self.rng = np.random.RandomState(random_state)

        if causal_indices is None:
            self.causal_indices = list(range(order))
        else:
            if len(causal_indices) != order:
                raise ValueError(f"Number of causal indices ({len(causal_indices)}) must match order ({order})")
            self.causal_indices = list(causal_indices)

        self.penetrance_calculator = PenetranceCalculator(
            order=self.order, maf=self.maf, model_type=self.model_type
        )
        self.penetrance_table, self.h2_meta = self.penetrance_calculator.calibrate_penetrance_table(
            target_h2=self.heritability
        )

    def generate_case_control(self) -> Dict[str, Union[np.ndarray, List[int], Dict]]:
        """
        Generates a balanced case-control dataset using exact conditional posterior sampling.

        Returns:
            Dictionary containing:
                - 'X': Genotype matrix of shape (n_samples, n_snps) with values {0, 1, 2}
                - 'y': Binary label array of shape (n_samples,) with values {0, 1}
                - 'causal_indices': Ground-truth interacting SNP indices
                - 'metadata': Configuration and heritability metrics
        """
        n_cases = self.n_samples // 2
        n_controls = self.n_samples - n_cases

        # Joint prior P(g)
        p_joint = self.penetrance_calculator.joint_probabilities
        # Penetrance P(Y=1 | g)
        pen = self.penetrance_table
        K = self.h2_meta["prevalence"]

        # Conditional posterior probabilities:
        # P(g | Y=1) = P(Y=1 | g) * P(g) / K
        # P(g | Y=0) = (1 - P(Y=1 | g)) * P(g) / (1 - K)
        p_g_given_case = (pen * p_joint) / K
        p_g_given_case /= np.sum(p_g_given_case)

        p_g_given_control = ((1.0 - pen) * p_joint) / (1.0 - K)
        p_g_given_control /= np.sum(p_g_given_control)

        # Sample causal genotype configurations
        n_combs = len(self.penetrance_calculator.genotype_combinations)
        case_comb_indices = self.rng.choice(n_combs, size=n_cases, p=p_g_given_case)
        control_comb_indices = self.rng.choice(n_combs, size=n_controls, p=p_g_given_control)

        comb_array = np.array(self.penetrance_calculator.genotype_combinations, dtype=np.int8)
        causal_cases = comb_array[case_comb_indices]
        causal_controls = comb_array[control_comb_indices]

        # Allocate full genotype matrix
        X = np.empty((self.n_samples, self.n_snps), dtype=np.int8)
        y = np.empty(self.n_samples, dtype=np.int8)

        # Insert causal SNPs
        X[:n_cases, self.causal_indices] = causal_cases
        X[n_cases:, self.causal_indices] = causal_controls
        y[:n_cases] = 1
        y[n_cases:] = 0

        # Sample non-causal background noise SNPs under HWE
        non_causal_indices = [i for i in range(self.n_snps) if i not in self.causal_indices]
        if non_causal_indices:
            p = self.maf
            # Binomial sampling: B(2, p) corresponds exactly to HWE: (1-p)^2, 2p(1-p), p^2
            noise_genotypes = self.rng.binomial(2, p, size=(self.n_samples, len(non_causal_indices))).astype(np.int8)
            X[:, non_causal_indices] = noise_genotypes

        # Shuffle sample rows so cases/controls are randomized
        shuffle_idx = self.rng.permutation(self.n_samples)
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        return {
            "X": X,
            "y": y,
            "causal_indices": self.causal_indices,
            "metadata": {
                **self.h2_meta,
                "n_snps": self.n_snps,
                "n_samples": self.n_samples,
                "n_cases": n_cases,
                "n_controls": n_controls,
            },
        }

    def generate_continuous(self) -> Dict[str, Union[np.ndarray, List[int], Dict]]:
        """
        Generates a continuous phenotype dataset (e.g., drug response IC50 / viability).
        Phenotype y = f(G_causal) + N(0, sigma_e^2) calibrated to target H^2.
        """
        # Sample all SNPs from HWE
        p = self.maf
        X = self.rng.binomial(2, p, size=(self.n_samples, self.n_snps)).astype(np.int8)

        # Extract causal genotypes
        causal_X = X[:, self.causal_indices]

        # Compute genetic signal f(G_causal)
        comb_dict = self.penetrance_calculator.get_table_dict(self.penetrance_table)
        genotype_tuples = [tuple(row) for row in causal_X]
        genetic_signal = np.array([comb_dict[g] for g in genotype_tuples], dtype=np.float64)

        var_g = np.var(genetic_signal)
        if var_g < 1e-9:
            var_g = 1e-9

        # Target H^2 = var_g / (var_g + var_e) => var_e = var_g * (1 - H^2) / H^2
        var_e = var_g * (1.0 - self.heritability) / max(self.heritability, 1e-4)
        noise = self.rng.normal(0, np.sqrt(var_e), size=self.n_samples)
        y = genetic_signal + noise

        return {
            "X": X,
            "y": y.astype(np.float32),
            "causal_indices": self.causal_indices,
            "metadata": {
                **self.h2_meta,
                "phenotype_type": "continuous",
                "n_snps": self.n_snps,
                "n_samples": self.n_samples,
                "var_g": float(var_g),
                "var_e": float(var_e),
            },
        }

    def save_dataset(self, data: Dict, output_path: Union[str, Path]) -> None:
        """Saves generated dataset and metadata to an NPZ archive and JSON."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out.with_suffix(".npz"),
            X=data["X"],
            y=data["y"],
            causal_indices=np.array(data["causal_indices"], dtype=np.int32),
        )
        with open(out.with_suffix(".json"), "w") as f:
            json.dump(data["metadata"], f, indent=2)

    @staticmethod
    def load_dataset(path: Union[str, Path]) -> Dict[str, Union[np.ndarray, List[int], Dict]]:
        """Loads dataset from NPZ archive and companion metadata JSON."""
        p = Path(path)
        npz = np.load(p.with_suffix(".npz"))
        X = npz["X"]
        y = npz["y"]
        causal_indices = npz["causal_indices"].tolist()
        json_path = p.with_suffix(".json")
        metadata = {}
        if json_path.exists():
            with open(json_path, "r") as f:
                metadata = json.load(f)
        return {"X": X, "y": y, "causal_indices": causal_indices, "metadata": metadata}
