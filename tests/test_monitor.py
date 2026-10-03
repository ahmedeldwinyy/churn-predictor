import numpy as np
import pandas as pd

from churn_predictor.monitor import psi_categorical, psi_numeric


def test_identical_distributions_have_zero_psi():
    x = pd.Series(np.random.default_rng(0).normal(size=2000))
    assert psi_numeric(x, x) < 0.001


def test_shifted_numeric_is_flagged():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 3000))
    b = pd.Series(rng.normal(1.5, 1, 3000))
    assert psi_numeric(a, b) > 0.25


def test_categorical_shift_is_flagged_and_unseen_category_is_handled():
    a = pd.Series(["x"] * 800 + ["y"] * 200)
    b = pd.Series(["x"] * 300 + ["y"] * 300 + ["z"] * 400)
    assert psi_categorical(a, a) < 0.001
    assert psi_categorical(a, b) > 0.25


def test_constant_column_does_not_crash():
    a = pd.Series([0] * 100)
    assert psi_numeric(a, a) < 0.001
