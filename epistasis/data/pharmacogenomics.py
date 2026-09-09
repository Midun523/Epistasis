"""Pharmacogenomic dataset loaders and preprocessors (GDSC, CCLE, DepMap, PharmGKB)."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


# Core pharmacokinetic and pharmacodynamic gene panels (PharmGKB / KEGG drug metabolism)
CORE_PHARMACOGENOMIC_GENES = [
    # Phase I drug metabolizing enzymes (CYP450 superfamily)
    "CYP1A1", "CYP1A2", "CYP2B6", "CYP2C8", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4", "CYP3A5",
    # Phase II conjugating enzymes
    "UGT1A1", "UGT2B7", "TPMT", "NUDT15", "NAT2", "G6PD", "GSTP1", "DPYD",
    # Transporters (SLC and ABC families)
    "ABCB1", "ABCC1", "ABCG2", "SLCO1B1", "SLCO1B3", "SLC22A1", "SLC22A2",
    # Key oncogenic drug targets & downstream effectors (EGFR, MAPK, PI3K/AKT, DNA repair)
    "EGFR", "ERBB2", "BRAF", "KRAS", "NRAS", "PIK3CA", "PTEN", "AKT1", "MTOR",
    "TP53", "BRCA1", "BRCA2", "PARP1", "ATM", "ATR", "CDK4", "CDK6", "RB1",
]


class PharmacogenomicsDataLoader:
    """Handles loading, pre-filtering, and formatting of cancer cell line drug response data."""

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/pharmacogenomics")

    def get_candidate_gene_panel(self, custom_genes: Optional[List[str]] = None) -> List[str]:
        """Returns the biologically-motivated candidate gene panel for Stage-1 candidate reduction."""
        if custom_genes:
            return sorted(list(set(CORE_PHARMACOGENOMIC_GENES + custom_genes)))
        return CORE_PHARMACOGENOMIC_GENES

    def generate_synthetic_pharmacogenomic_cohort(
        self,
        n_cell_lines: int = 800,
        n_genes: int = 100,
        causal_gene_indices: Optional[List[int]] = None,
        interaction_order: int = 3,
        heritability: float = 0.35,
        random_state: int = 42,
    ) -> Dict[str, Union[np.ndarray, List[str], List[int], Dict]]:
        """
        Generates a synthetic GDSC/CCLE-style cohort with realistic variant distributions
        and an embedded multi-gene epistatic drug-response interaction (continuous IC50).
        """
        rng = np.random.RandomState(random_state)
        gene_names = [f"GENE_{i:03d}" for i in range(n_genes)]
        for i, core_gene in enumerate(CORE_PHARMACOGENOMIC_GENES[: min(len(CORE_PHARMACOGENOMIC_GENES), n_genes)]):
            gene_names[i] = core_gene

        if causal_gene_indices is None:
            causal_gene_indices = list(range(interaction_order))

        # Binary mutation / discrete alteration states {0: wild-type, 1: heterozygous, 2: homozygous/amplified}
        # Rare variant distribution typical in cancer cell lines (MAF ~ 0.05 - 0.20)
        mafs = rng.uniform(0.05, 0.25, size=n_genes)
        X = np.zeros((n_cell_lines, n_genes), dtype=np.int8)
        for j in range(n_genes):
            X[:, j] = rng.binomial(2, mafs[j], size=n_cell_lines)

        # Multi-gene synergistic drug resistance / sensitivity model:
        # e.g., synergy when all causal mutations co-occur (synthetic lethality / collateral resistance)
        causal_X = X[:, causal_gene_indices].astype(np.float64)
        # Epistatic drug sensitivity: multiplicative synergy
        synergy_effect = np.prod(causal_X + 1.0, axis=1) - 1.0

        var_g = np.var(synergy_effect)
        if var_g < 1e-9:
            var_g = 1e-9

        var_e = var_g * (1.0 - heritability) / max(heritability, 1e-4)
        noise = rng.normal(0, np.sqrt(var_e), size=n_cell_lines)
        ic50_values = (synergy_effect + noise).astype(np.float32)

        return {
            "X": X,
            "y": ic50_values,
            "gene_names": gene_names,
            "causal_indices": causal_gene_indices,
            "causal_genes": [gene_names[i] for i in causal_gene_indices],
            "metadata": {
                "phenotype_type": "regression",
                "n_cell_lines": n_cell_lines,
                "n_genes": n_genes,
                "heritability": heritability,
                "interaction_order": interaction_order,
                "phenotype": "log(IC50)",
            },
        }
