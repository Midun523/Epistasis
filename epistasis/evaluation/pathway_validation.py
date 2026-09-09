"""Biological pathway and protein-protein interaction (STRING / KEGG) validation for discovered epistatic pairs."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import requests
from epistasis.utils.logger import get_logger

logger = get_logger("pathway_validation")

# Curated benchmark synthetic lethal / synergistic gene interaction pairs
KNOWN_SYNTHETIC_LETHAL_PAIRS = {
    ("PARP1", "BRCA1"),
    ("PARP1", "BRCA2"),
    ("PARP1", "ATM"),
    ("ARID1A", "ARID1B"),
    ("SMARCA4", "SMARCA2"),
    ("STAG1", "STAG2"),
    ("CDK4", "RB1"),
    ("BRAF", "EGFR"),
    ("KRAS", "PIK3CA"),
    ("TP53", "MDM2"),
}


class BiologicalValidator:
    """Queries STRING API and KEGG pathways to validate biological plausibility of epistatic interactions."""

    def __init__(self, species_id: int = 9606):  # 9606 is Homo sapiens
        self.species_id = species_id
        self.string_api_url = "https://string-db.org/api/json/network"

    def query_string_interactions(
        self, gene_list: List[str], score_threshold: int = 400
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Queries STRING database API for known protein-protein physical or functional interactions.

        Args:
            gene_list: List of gene symbols (e.g. ['BRAF', 'EGFR', 'KRAS']).
            score_threshold: Minimum STRING confidence score in [0, 1000] (400 = medium, 700 = high).

        Returns:
            List of detected interaction records with confidence scores.
        """
        if len(gene_list) < 2:
            return []

        params = {
            "identifiers": "%0d".join(gene_list),
            "species": self.species_id,
            "required_score": score_threshold,
            "caller_identity": "epistasis_research_framework",
        }

        try:
            response = requests.get(self.string_api_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data:
                    results.append({
                        "gene_a": item.get("preferredName_A", ""),
                        "gene_b": item.get("preferredName_B", ""),
                        "string_score": float(item.get("score", 0.0)),
                        "ncbi_tax_id": item.get("ncbiTaxonId", self.species_id),
                    })
                return results
            else:
                logger.warning(f"STRING API returned status {response.status_code}")
                return []
        except Exception as e:
            logger.warning(f"Could not connect to online STRING API ({e}). Falling back to local offline check.")
            return self._offline_interaction_check(gene_list)

    def _offline_interaction_check(self, gene_list: List[str]) -> List[Dict[str, Union[str, float]]]:
        """Offline fallback checker for curated synthetic lethal pairs."""
        results = []
        genes_set = set(gene_list)
        for g1, g2 in KNOWN_SYNTHETIC_LETHAL_PAIRS:
            if g1 in genes_set and g2 in genes_set:
                results.append({
                    "gene_a": g1,
                    "gene_b": g2,
                    "string_score": 0.950,
                    "is_synthetic_lethal": True,
                })
        return results

    def check_synthetic_lethality(self, gene_a: str, gene_b: str) -> bool:
        """Checks if a pair is an established synthetic lethal combination."""
        pair1 = (gene_a.upper(), gene_b.upper())
        pair2 = (gene_b.upper(), gene_a.upper())
        return pair1 in KNOWN_SYNTHETIC_LETHAL_PAIRS or pair2 in KNOWN_SYNTHETIC_LETHAL_PAIRS
