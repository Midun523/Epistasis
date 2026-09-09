"""Penetrance table generation and heritability calibration for higher-order epistasis."""

from __future__ import annotations
import itertools
from typing import Dict, List, Literal, Tuple, Union
import numpy as np
from scipy.optimize import root_scalar

EpistasisModelType = Literal["additive", "multiplicative", "threshold", "xor"]


class PenetranceCalculator:
    """Calculates penetrance tables P(Y=1 | G_1, ..., G_k) for k-way epistatic interactions."""

    def __init__(self, order: int, maf: float | List[float], model_type: EpistasisModelType):
        """
        Args:
            order: Epistatic interaction order k (e.g. 2 to 8).
            maf: Minor allele frequency (float or list of floats for each locus).
            model_type: One of 'additive', 'multiplicative', 'threshold', 'xor'.
        """
        if order < 2:
            raise ValueError(f"Order must be >= 2, got {order}")
        self.order = order
        if isinstance(maf, (int, float)):
            self.mafs = [float(maf)] * order
        else:
            if len(maf) != order:
                raise ValueError(f"Length of MAFs ({len(maf)}) must match order ({order})")
            self.mafs = [float(p) for p in maf]
        self.model_type = model_type
        self.genotype_combinations = list(itertools.product([0, 1, 2], repeat=order))
        self.joint_probabilities = self._compute_joint_probabilities()

    def _compute_joint_probabilities(self) -> np.ndarray:
        """Computes joint genotype probabilities assuming Hardy-Weinberg and Linkage Equilibrium."""
        probs = []
        for g_tuple in self.genotype_combinations:
            p_joint = 1.0
            for g, p in zip(g_tuple, self.mafs):
                if g == 0:
                    p_joint *= (1.0 - p) ** 2
                elif g == 1:
                    p_joint *= 2.0 * p * (1.0 - p)
                elif g == 2:
                    p_joint *= p ** 2
                else:
                    raise ValueError(f"Invalid genotype value: {g}")
            probs.append(p_joint)
        probs = np.array(probs, dtype=np.float64)
        return probs / np.sum(probs)

    def _raw_penetrance(self, scale: float, baseline: float = 0.1) -> np.ndarray:
        """Calculates raw unscaled or scaled penetrance values based on model_type."""
        penetrances = []
        for g_tuple in self.genotype_combinations:
            g_arr = np.array(g_tuple, dtype=np.float64)
            if self.model_type == "additive":
                # Linear additive sum of minor alleles + high-order interaction term
                main_effect = np.sum(g_arr)
                interaction = np.prod(g_arr)
                raw = baseline + scale * (0.2 * main_effect + 0.8 * interaction)
            elif self.model_type == "multiplicative":
                # Multiplicative synergy: (1 + alpha * g_1) * ... * (1 + alpha * g_k)
                synergy = np.prod(1.0 + g_arr) - 1.0
                raw = baseline + scale * synergy
            elif self.model_type == "threshold":
                # Threshold model: threshold at total alleles >= order
                total_alleles = np.sum(g_arr)
                threshold = self.order
                raw = baseline + scale * (1.0 if total_alleles >= threshold else 0.0)
            elif self.model_type == "xor":
                # Pure parity / XOR epistasis: disease risk elevated if allele sum is odd
                # This produces zero marginal single-SNP main effect
                total_alleles = int(np.sum(g_arr))
                is_odd = (total_alleles % 2) == 1
                raw = baseline + (scale if is_odd else 0.0)
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")
            penetrances.append(raw)

        pen_arr = np.clip(np.array(penetrances, dtype=np.float64), 0.0001, 0.9999)
        return pen_arr

    def compute_heritability(self, penetrances: np.ndarray) -> Tuple[float, float, float]:
        """
        Computes population prevalence (K), phenotypic variance (V_P), and broad-sense heritability (H^2).

        Returns:
            (prevalence, genetic_variance, heritability)
        """
        K = np.sum(self.joint_probabilities * penetrances)
        V_P = K * (1.0 - K)
        if V_P <= 1e-12:
            return K, 0.0, 0.0
        V_G = np.sum(self.joint_probabilities * ((penetrances - K) ** 2))
        H2 = V_G / V_P
        return K, V_G, H2

    def calibrate_penetrance_table(
        self, target_h2: float, baseline: float = 0.1, tolerance: float = 1e-3
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Calibrates the scale factor of the penetrance table to match the target broad-sense heritability H^2.

        Args:
            target_h2: Target heritability in (0, 1) (e.g. 0.01, 0.05, 0.20, 0.40).
            baseline: Baseline disease prevalence.
            tolerance: Optimization tolerance.

        Returns:
            (penetrance_table, metrics_dict)
        """
        if not (0.0 < target_h2 < 1.0):
            raise ValueError(f"Target heritability must be in (0, 1), got {target_h2}")

        def objective(scale: float) -> float:
            pen = self._raw_penetrance(scale, baseline=baseline)
            _, _, h2 = self.compute_heritability(pen)
            return h2 - target_h2

        # Binary search / root finding for optimal scale factor
        low, high = 0.0001, 5.0
        # Check boundary values
        obj_low = objective(low)
        obj_high = objective(high)

        if obj_low * obj_high > 0:
            # If sign doesn't change, perform grid search to bracket root
            scales = np.linspace(0.001, 20.0, 200)
            objs = [objective(s) for s in scales]
            min_idx = np.argmin(np.abs(objs))
            best_scale = scales[min_idx]
        else:
            sol = root_scalar(objective, bracket=[low, high], method="brentq", xtol=tolerance)
            best_scale = sol.root

        final_pen = self._raw_penetrance(best_scale, baseline=baseline)
        K, V_G, final_h2 = self.compute_heritability(final_pen)

        metrics = {
            "target_h2": target_h2,
            "actual_h2": float(final_h2),
            "prevalence": float(K),
            "genetic_variance": float(V_G),
            "optimal_scale": float(best_scale),
            "order": self.order,
            "model_type": self.model_type,
            "maf": self.mafs[0] if len(set(self.mafs)) == 1 else self.mafs,
        }
        return final_pen, metrics

    def get_table_dict(self, penetrance_array: np.ndarray) -> Dict[Tuple[int, ...], float]:
        """Returns mapping from genotype tuple (e.g., (0, 1, 2)) to penetrance P(Y=1)."""
        return {
            comb: float(prob)
            for comb, prob in zip(self.genotype_combinations, penetrance_array)
        }
