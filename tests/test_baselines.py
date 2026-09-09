"""Unit tests for baseline models and evaluation metrics."""

import numpy as np
import pytest
from epistasis.models.baselines import (
    MultifactorDimensionalityReduction,
    InteractionRegressionBaseline,
    TreeEnsembleBaseline,
)
from epistasis.models.deepcombi_mlp import DeepCOMBIBaseline
from epistasis.evaluation.metrics import evaluate_epistasis_detection, compute_prediction_metrics


def test_baseline_models():
    N_samples, N_snps = 60, 20
    X = np.random.randint(0, 3, size=(N_samples, N_snps))
    y = np.random.randint(0, 2, size=N_samples)

    # MDR
    mdr = MultifactorDimensionalityReduction(order=2)
    mdr.fit(X, y, candidate_indices=list(range(5)))
    assert len(mdr.score_features(N_snps)) == N_snps

    # Logistic Regression with Interactions
    reg = InteractionRegressionBaseline(max_features=10)
    reg.fit(X, y)
    assert len(reg.score_features(N_snps)) == N_snps

    # XGBoost
    xgb_m = TreeEnsembleBaseline(model_type="xgboost", n_estimators=10)
    xgb_m.fit(X, y)
    assert len(xgb_m.score_features(N_snps)) == N_snps

    # DeepCOMBI
    deepc = DeepCOMBIBaseline(num_snps=N_snps, epochs=2, device="cpu")
    deepc.fit(X, y, batch_size=16)
    assert len(deepc.score_features(X)) == N_snps


def test_evaluation_metrics():
    # 10 SNPs, causal are [0, 1]
    scores = np.array([0.9, 0.8, 0.1, 0.05, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0])
    causal = [0, 1]

    # top 20% -> top 2 -> indices 0 and 1
    eval_res = evaluate_epistasis_detection(scores, causal, top_percent=0.20)
    assert eval_res["detected_all_top_5pct"] is True
    assert eval_res["recall_exact"] == 1.0
    assert eval_res["precision_exact"] == 1.0
    assert eval_res["causal_ranks"] == [1, 2]
