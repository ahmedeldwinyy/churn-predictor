"""Phase H: data drift check using the Population Stability Index (PSI).

PSI compares how a feature is distributed in the training data versus new data.
Rule of thumb: < 0.10 stable, 0.10-0.25 watch, > 0.25 significant shift.

Run:  uv run python -m churn_predictor.monitor                 (train vs test file)
      uv run python -m churn_predictor.monitor new_customers.csv
      uv run python -m churn_predictor.monitor --demo-shift     (simulated drift)
"""
import argparse

import numpy as np
import pandas as pd

from churn_predictor.models import CATEGORICAL, NUMERIC, PASSTHROUGH
from churn_predictor.split import TEST_PATH, load_train

WATCH, ALERT = 0.10, 0.25


def _psi(expected_share, actual_share, eps: float = 1e-4) -> float:
    e = np.clip(np.asarray(expected_share, dtype=float), eps, None)
    a = np.clip(np.asarray(actual_share, dtype=float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    cats = sorted(set(expected.dropna().unique()) | set(actual.dropna().unique()), key=str)
    e = expected.value_counts(normalize=True).reindex(cats, fill_value=0.0)
    a = actual.value_counts(normalize=True).reindex(cats, fill_value=0.0)
    return _psi(e.to_numpy(), a.to_numpy())


def psi_numeric(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (almost) constant column: compare as categories instead
        return psi_categorical(expected, actual)
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    return _psi(e, a)


def drift_report(train: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    rows = [(c, psi_numeric(train[c], new[c])) for c in NUMERIC]
    rows += [(c, psi_categorical(train[c], new[c])) for c in PASSTHROUGH + CATEGORICAL]
    out = pd.DataFrame(rows, columns=["feature", "psi"])
    out["status"] = np.where(out["psi"] > ALERT, "ALERT", np.where(out["psi"] > WATCH, "watch", "ok"))
    return out.sort_values("psi", ascending=False).reset_index(drop=True)


def simulate_shift(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Fake a market change: prices up 25% and 30% of customers move to month-to-month."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["MonthlyCharges"] = out["MonthlyCharges"] * 1.25
    out.loc[rng.random(len(out)) < 0.3, "Contract"] = "Month-to-month"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare new customer data against the training data.")
    ap.add_argument("csv", nargs="?", help="new data (default: the held-out test file)")
    ap.add_argument("--demo-shift", action="store_true", help="simulate drift to see alerts")
    args = ap.parse_args()

    train = load_train()
    new = pd.read_csv(args.csv) if args.csv else pd.read_csv(TEST_PATH)
    if args.demo_shift:
        new = simulate_shift(new)
    report = drift_report(train, new)
    print(report.round(3).to_string(index=False))
    flagged = report[report["status"] == "ALERT"]
    if len(flagged):
        print(f"\nALERT: {len(flagged)} feature(s) shifted a lot ({', '.join(flagged['feature'])}). "
              "Investigate, and consider retraining.")
    else:
        print("\nNo significant drift.")


if __name__ == "__main__":
    main()
