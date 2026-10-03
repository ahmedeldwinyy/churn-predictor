"""Load the saved model and score customers.

Batch use:  uv run python -m churn_predictor.predict data/raw/telco_churn.csv --top 20
"""
import argparse
import json
from functools import lru_cache

import joblib
import pandas as pd

from churn_predictor.config import DECISION_PATH, MODEL_PATH


@lru_cache(maxsize=1)
def load_model():
    """Return (fitted pipeline, decision dict). Cached so it loads once per process."""
    return joblib.load(MODEL_PATH), json.loads(DECISION_PATH.read_text())


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Same rule as cleaning: a blank TotalCharges means a brand-new customer (0.0)
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"].replace(r"^\s*$", "0", regex=True))
    return out


def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    """Churn probability and contact flag for each row (index is preserved)."""
    model, decision = load_model()
    prob = model.predict_proba(_prepare(df))[:, 1]
    return pd.DataFrame(
        {"churn_probability": prob.round(4), "contact": prob >= decision["threshold"]},
        index=df.index,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a CSV of customers and rank them by churn risk.")
    ap.add_argument("csv", help="CSV with the same columns as the raw Telco data")
    ap.add_argument("--top", type=int, default=None, help="keep only the N riskiest customers")
    ap.add_argument("--out", default="predictions.csv", help="where to save the ranked list")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "customerID" in df.columns:
        df = df.set_index("customerID")
    ranked = predict_df(df).sort_values("churn_probability", ascending=False)
    if args.top:
        ranked = ranked.head(args.top)
    ranked.to_csv(args.out)
    print(ranked.head(10).to_string())
    print(f"\nSaved {len(ranked)} rows to {args.out}")


if __name__ == "__main__":
    main()
