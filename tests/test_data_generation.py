"""Unit tests for penetrance calculation, heritability calibration, and synthetic data generation."""

import numpy as np
import pytest
from epistasis.data.penetrance import PenetranceCalculator
from epistasis.data.gametes_simulator import EpistasisSimulator


@pytest.mark.parametrize("model_type", ["additive", "multiplicative", "threshold", "xor"])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_penetrance_calibration(model_type, order):
    maf = 0.2
    target_h2 = 0.20
    calc = PenetranceCalculator(order=order, maf=maf, model_type=model_type)
    table, meta = calc.calibrate_penetrance_table(target_h2=target_h2)

    assert len(table) == 3 ** order
    assert np.all(table >= 0.0) and np.all(table <= 1.0)
    assert meta["target_h2"] == target_h2
    assert 0.0 < meta["prevalence"] < 1.0
    assert 0.0 < meta["actual_h2"] <= 1.0


def test_epistasis_simulator_case_control():
    sim = EpistasisSimulator(
        n_snps=100,
        n_samples=200,
        order=2,
        maf=0.2,
        heritability=0.20,
        model_type="xor",
        random_state=42,
    )
    data = sim.generate_case_control()

    X, y, causal = data["X"], data["y"], data["causal_indices"]
    assert X.shape == (200, 100)
    assert y.shape == (200,)
    assert len(causal) == 2
    assert np.sum(y == 1) == 100
    assert np.sum(y == 0) == 100
    assert np.all(np.isin(X, [0, 1, 2]))


def test_epistasis_simulator_continuous():
    sim = EpistasisSimulator(
        n_snps=50,
        n_samples=100,
        order=3,
        maf=0.2,
        heritability=0.30,
        model_type="multiplicative",
        random_state=42,
    )
    data = sim.generate_continuous()
    X, y = data["X"], data["y"]
    assert X.shape == (100, 50)
    assert y.shape == (100,)
    assert np.issubdtype(y.dtype, np.floating)
