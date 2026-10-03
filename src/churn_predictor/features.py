"""Row-wise feature engineering (stateless: nothing is learned from other rows)."""
import numpy as np
import pandas as pd

ADDONS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def _price(df: pd.DataFrame) -> dict:
    # tenure 0 would divide by zero: fall back to the current monthly price
    avg_paid = (df["TotalCharges"] / df["tenure"].replace(0, np.nan)).fillna(df["MonthlyCharges"])
    return {"avg_paid": avg_paid, "price_gap": df["MonthlyCharges"] - avg_paid}


def _addons(df: pd.DataFrame) -> dict:
    return {"n_addons": (df[ADDONS] == "Yes").sum(axis=1)}


def _interaction(df: pd.DataFrame) -> dict:
    return {"tenure_if_mtm": df["tenure"].where(df["Contract"] == "Month-to-month", 0)}


GROUPS = {
    "price": (_price, ["avg_paid", "price_gap"]),
    "addons": (_addons, ["n_addons"]),
    "interaction": (_interaction, ["tenure_if_mtm"]),
}


def add_features(df: pd.DataFrame, groups: tuple = ()) -> pd.DataFrame:
    out = df.copy()
    for g in groups:
        for col, values in GROUPS[g][0](out).items():
            out[col] = values
    return out


def new_columns(groups: tuple = ()) -> list:
    return [c for g in groups for c in GROUPS[g][1]]
