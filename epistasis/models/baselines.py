"""Classical and Machine Learning Baselines for Epistasis Detection."""

from __future__ import annotations
import itertools
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


class MultifactorDimensionalityReduction:
    """
    Multifactor Dimensionality Reduction (MDR) for detecting epistatic interactions.
    Pools multi-locus genotype cells into binary high-risk / low-risk classes.
    """

    def __init__(self, order: int = 2, threshold_ratio: float = 1.0):
        self.order = order
        self.threshold_ratio = threshold_ratio
        self.best_combination: Optional[Tuple[int, ...]] = None
        self.best_score: float = -1.0
        self.cell_labels: Dict[Tuple[int, ...], int] = {}

    def fit(self, X: np.ndarray, y: np.ndarray, candidate_indices: Optional[List[int]] = None) -> "MultifactorDimensionalityReduction":
        """
        Evaluates multi-locus combinations and finds the combination with highest balanced accuracy.
        Uses vectorized base-3 integer encoding and bincount for high performance.
        """
        n_samples, n_snps = X.shape
        eval_indices = candidate_indices if candidate_indices is not None else list(range(n_snps))
        combinations = list(itertools.combinations(eval_indices, self.order))

        y_int = y.astype(int)
        cases_mask = (y_int == 1)
        ctrls_mask = (y_int == 0)
        n_cases = np.sum(cases_mask)
        n_ctrls = np.sum(ctrls_mask)

        base_mult = np.array([3 ** (self.order - 1 - i) for i in range(self.order)], dtype=np.int64)
        num_bins = 3 ** self.order

        best_score = -1.0
        best_comb = None
        best_cells = {}

        X_int = X.astype(np.int64)

        for comb in combinations:
            sub_X = X_int[:, comb]
            cell_ids = np.dot(sub_X, base_mult)

            cases_per_cell = np.bincount(cell_ids[cases_mask], minlength=num_bins)
            ctrls_per_cell = np.bincount(cell_ids[ctrls_mask], minlength=num_bins)

            ratios = cases_per_cell / np.maximum(ctrls_per_cell, 1e-6)
            high_risk_bins = (ratios >= self.threshold_ratio)

            tp = np.sum(cases_per_cell[high_risk_bins])
            tn = np.sum(ctrls_per_cell[~high_risk_bins])

            sens = tp / max(n_cases, 1)
            spec = tn / max(n_ctrls, 1)
            bal_acc = 0.5 * (sens + spec)

            if bal_acc > best_score:
                best_score = bal_acc
                best_comb = comb

        if best_comb is not None:
            sub_X = X_int[:, best_comb]
            cell_ids = np.dot(sub_X, base_mult)
            cases_per_cell = np.bincount(cell_ids[cases_mask], minlength=num_bins)
            ctrls_per_cell = np.bincount(cell_ids[ctrls_mask], minlength=num_bins)
            ratios = cases_per_cell / np.maximum(ctrls_per_cell, 1e-6)
            high_risk_bins = (ratios >= self.threshold_ratio)

            unique_c_ids = np.unique(cell_ids)
            for cid in unique_c_ids:
                digits = []
                rem = cid
                for bm in base_mult:
                    digits.append(int(rem // bm))
                    rem = rem % bm
                best_cells[tuple(digits)] = int(high_risk_bins[cid])

        self.best_combination = best_comb
        self.best_score = best_score
        self.cell_labels = best_cells
        return self

    def score_features(self, n_snps: int) -> np.ndarray:
        """Returns feature importance ranking based on inclusion in top epistatic model."""
        scores = np.zeros(n_snps, dtype=np.float32)
        if self.best_combination is not None:
            for idx in self.best_combination:
                scores[idx] = self.best_score
        return scores


class InteractionRegressionBaseline:
    """
    Logistic or Linear Regression with explicit pairwise interaction terms (X_i * X_j).
    Restricted to candidate subset to prevent combinatorial O(S^2) explosion at genome scale.
    """

    def __init__(self, task: str = "classification", max_features: int = 100, penalty: str = "l2", C: float = 1.0):
        self.task = task
        self.max_features = max_features
        self.penalty = penalty
        self.C = C
        self.model = None
        self.selected_snps: List[int] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "InteractionRegressionBaseline":
        n_samples, n_snps = X.shape
        # If n_snps > max_features, pre-filter by single-locus variance/association
        if n_snps > self.max_features:
            variances = np.var(X, axis=0)
            self.selected_snps = np.argsort(variances)[-self.max_features:].tolist()
        else:
            self.selected_snps = list(range(n_snps))

        sub_X = X[:, self.selected_snps]
        # Generate pairwise interaction features: [X_i] + [X_i * X_j]
        n_sub = len(self.selected_snps)
        interactions = []
        feature_map = []  # Maps each interaction column back to constituent SNP indices

        # Main terms
        for i in range(n_sub):
            feature_map.append([self.selected_snps[i]])

        # Pairwise interaction terms
        for i, j in itertools.combinations(range(n_sub), 2):
            interactions.append(sub_X[:, i] * sub_X[:, j])
            feature_map.append([self.selected_snps[i], self.selected_snps[j]])

        if interactions:
            int_matrix = np.column_stack(interactions)
            full_X = np.column_stack([sub_X, int_matrix])
        else:
            full_X = sub_X

        self.scaler = StandardScaler()
        full_X_scaled = self.scaler.fit_transform(full_X)
        self.feature_map = feature_map

        if self.task == "classification":
            self.model = LogisticRegression(C=self.C, max_iter=1000, random_state=42)
        else:
            self.model = Ridge(alpha=1.0 / self.C, random_state=42)

        self.model.fit(full_X_scaled, y)
        return self

    def score_features(self, n_snps: int) -> np.ndarray:
        """Attributes regression coefficients back to constituent SNPs."""
        scores = np.zeros(n_snps, dtype=np.float32)
        if self.model is None:
            return scores

        coeffs = np.abs(self.model.coef_.ravel())
        for term_idx, weight in enumerate(coeffs):
            snps = self.feature_map[term_idx]
            for snp in snps:
                scores[snp] += weight / len(snps)
        return scores


class TreeEnsembleBaseline:
    """Random Forest and XGBoost baseline with feature importance scoring."""

    def __init__(self, model_type: str = "xgboost", task: str = "classification", n_estimators: int = 100):
        self.model_type = model_type
        self.task = task
        self.n_estimators = n_estimators
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TreeEnsembleBaseline":
        if self.model_type == "xgboost":
            if self.task == "classification":
                self.model = xgb.XGBClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=6,
                    learning_rate=0.05,
                    tree_method="hist",
                    random_state=42,
                    eval_metric="logloss",
                )
            else:
                self.model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=6,
                    learning_rate=0.05,
                    tree_method="hist",
                    random_state=42,
                )
        else:  # Random Forest
            if self.task == "classification":
                self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=10, random_state=42, n_jobs=-1)
            else:
                self.model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=10, random_state=42, n_jobs=-1)

        self.model.fit(X, y)
        return self

    def score_features(self, n_snps: int) -> np.ndarray:
        """Returns Gini / Gain feature importances."""
        if self.model is None:
            return np.zeros(n_snps, dtype=np.float32)
        return self.model.feature_importances_.astype(np.float32)
