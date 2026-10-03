import numpy as np
import pytest

from churn_predictor.business import (
    CALL_COST,
    CAPACITY_SHARE,
    OFFER_SUCCESS,
    SAVE_VALUE,
    best_threshold,
    effective_threshold,
    profit_from_called,
)


def test_profit_hand_example():
    # 4 customers, we contact the first two: one real churner, one loyal
    got = profit_from_called([1, 0, 1, 0], [True, True, False, False])
    expected = (OFFER_SUCCESS * SAVE_VALUE - 2 * CALL_COST) / 4 * 1000
    assert got == pytest.approx(expected)


def test_contacting_nobody_earns_zero():
    assert profit_from_called([1, 0, 1, 0], [False] * 4) == 0.0


def test_capacity_limits_contacts():
    rng = np.random.default_rng(0)
    score = rng.random(2000)
    called = score >= effective_threshold(score, threshold=0.0)
    assert called.mean() == pytest.approx(CAPACITY_SHARE, abs=0.01)


def test_best_threshold_prefers_a_good_ranking():
    rng = np.random.default_rng(1)
    y = (rng.random(3000) < 0.25).astype(int)
    good = np.clip(y * 0.5 + rng.random(3000) * 0.5, 0, 1)  # churners tend to score higher
    random_score = rng.random(3000)
    _, p_good = best_threshold(y, good)
    _, p_random = best_threshold(y, random_score)
    assert p_good > p_random
