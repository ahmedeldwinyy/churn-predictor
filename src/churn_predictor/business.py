"""Business assumptions and the cost model.

REPLACE THE NUMBERS BELOW with real company figures. Everything downstream
(threshold, model choice, profit estimates) is only as good as these assumptions.
"""
import numpy as np

SAVE_VALUE = 400.0     # $ value of keeping a customer who would otherwise have churned
OFFER_SUCCESS = 0.30   # share of called churners who accept the retention offer
CALL_COST = 25.0       # $ cost of contacting one customer (agent time + discount)
CAPACITY_SHARE = 0.15  # the retention team can contact at most this share of customers


def effective_threshold(score, threshold: float) -> float:
    """The threshold actually applied: the team cannot contact more than CAPACITY_SHARE."""
    capacity_cutoff = float(np.quantile(np.asarray(score), 1 - CAPACITY_SHARE))
    return max(threshold, capacity_cutoff)


def profit_from_called(y_true, called) -> float:
    """Expected profit per 1,000 customers, given who was contacted (boolean array)."""
    y, called = np.asarray(y_true), np.asarray(called, dtype=bool)
    gain = (y[called] * OFFER_SUCCESS * SAVE_VALUE).sum() - CALL_COST * called.sum()
    return float(gain / len(y) * 1000)


def profit_per_1000(y_true, score, threshold: float) -> float:
    """Expected profit per 1,000 customers, contacting everyone above the effective threshold."""
    s = np.asarray(score)
    return profit_from_called(y_true, s >= effective_threshold(s, threshold))


def best_threshold(y_true, score, grid=None) -> tuple[float, float]:
    """Best (effective) threshold and its expected profit per 1,000 customers."""
    grid = np.arange(0.02, 0.95, 0.01) if grid is None else grid
    profits = [profit_per_1000(y_true, score, t) for t in grid]
    i = int(np.argmax(profits))
    return effective_threshold(score, float(grid[i])), float(profits[i])
