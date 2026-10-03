"""Loading and stateless cleaning of the Telco churn data."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "telco_churn.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "telco_clean.csv"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV exactly as delivered."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply stateless fixes only (nothing here learns from the data)."""
    out = df.copy()

    # TotalCharges: blanks are brand-new customers (tenure 0), so 0.0 is correct.
    # astype(float) raises on any OTHER bad value, so surprises fail loudly.
    out["TotalCharges"] = out["TotalCharges"].str.strip().replace("", "0").astype(float)

    # Target: Yes/No -> 1/0
    out["churn"] = (out["Churn"] == "Yes").astype(int)
    out = out.drop(columns=["Churn"])

    # The ID identifies a row; it is never a feature
    out = out.set_index("customerID")
    return out
