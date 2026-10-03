"""Phase E: out-of-fold scores, calibration, cost-based threshold, error analysis.

Uses the TRAINING set only (5-fold out-of-fold predictions). The test set stays locked.
Writes artifacts/decision.json, which the training script reads.
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from churn_predictor.business import (
    CALL_COST,
    CAPACITY_SHARE,
    OFFER_SUCCESS,
    SAVE_VALUE,
    best_threshold,
    profit_per_1000,
)
from churn_predictor.config import DECISION_PATH
from churn_predictor.metrics import evaluate
from churn_predictor.models import build_hgb, build_logreg
from churn_predictor.split import SEED, load_train

BUILDERS = {"logreg": build_logreg, "hgb": build_hgb}
MARGIN = 0.03  # the complex model must beat the simple one by 3% profit to be chosen


def oof_scores(model, X, y) -> np.ndarray:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]


def calibration_table(y, score, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"score": score, "y": np.asarray(y)})
    df["bin"] = pd.qcut(df["score"], bins, duplicates="drop")
    out = df.groupby("bin", observed=True).agg(predicted=("score", "mean"), actual=("y", "mean"), n=("y", "size"))
    return out.round(3).reset_index(drop=True)


def segment_recall(X, y, score, threshold, column) -> pd.DataFrame:
    df = pd.DataFrame({"seg": X[column].to_numpy(), "y": np.asarray(y), "called": score >= threshold})
    churners = df[df["y"] == 1]
    return churners.groupby("seg", observed=True).agg(churners=("y", "size"), caught=("called", "mean")).round(3)


def main() -> None:
    train = load_train()
    X, y = train.drop(columns="churn"), train["churn"]
    print(f"Assumptions: save value ${SAVE_VALUE:.0f}, offer success {OFFER_SUCCESS:.0%}, "
          f"cost per contact ${CALL_COST:.0f}, capacity {CAPACITY_SHARE:.0%} of customers")
    print(f"Break-even probability if scores were perfectly calibrated: {CALL_COST / (OFFER_SUCCESS * SAVE_VALUE):.3f}\n")

    scores, results = {}, {}
    for name, builder in BUILDERS.items():
        scores[name] = oof_scores(builder(), X, y)
        thr, profit = best_threshold(y, scores[name])
        called = scores[name] >= thr
        m = evaluate(y, called.astype(int), scores[name])
        results[name] = {"threshold": thr, "profit": profit}
        print(f"{name:7s} threshold {thr:.2f} | profit/1000 customers ${profit:,.0f} | "
              f"contacts {called.mean():.1%} | recall {m['recall']:.3f} precision {m['precision']:.3f} "
              f"(lift {m['precision'] / y.mean():.1f}x vs random)")
    everyone = profit_per_1000(y, np.ones(len(y)), 0.0)
    print(f"\nReference: contact nobody $0 | contact a RANDOM {CAPACITY_SHARE:.0%} of customers ${everyone * CAPACITY_SHARE:,.0f}")

    chosen = "logreg"
    if results["hgb"]["profit"] > results["logreg"]["profit"] * (1 + MARGIN):
        chosen = "hgb"
    print(f"\nChosen model: {chosen} (complex model needs +{MARGIN:.0%} profit to replace the simple one)")

    print("\nCalibration of chosen model (do predicted probabilities match reality?)")
    print(calibration_table(y, scores[chosen]).to_string(index=False))

    thr = results[chosen]["threshold"]
    for col in ["Contract", "InternetService"]:
        print(f"\nShare of real churners caught, by {col} (threshold {thr:.2f})")
        print(segment_recall(X, y, scores[chosen], thr, col).to_string())
    tenure_bin = pd.cut(X["tenure"], [-1, 6, 12, 24, 48, 72]).astype(str)
    seg = pd.DataFrame({"seg": tenure_bin.to_numpy(), "y": y.to_numpy(), "called": scores[chosen] >= thr})
    print("\nShare of real churners caught, by tenure bin")
    print(seg[seg["y"] == 1].groupby("seg").agg(churners=("y", "size"), caught=("called", "mean")).round(3).to_string())

    DECISION_PATH.parent.mkdir(parents=True, exist_ok=True)
    decision = {
        "model": chosen,
        "threshold": round(thr, 2),
        "cv_profit_per_1000": round(results[chosen]["profit"], 1),
        "assumptions": {
            "save_value": SAVE_VALUE,
            "offer_success": OFFER_SUCCESS,
            "call_cost": CALL_COST,
            "capacity_share": CAPACITY_SHARE,
        },
    }
    DECISION_PATH.write_text(json.dumps(decision, indent=2) + "\n")
    print(f"\nSaved {DECISION_PATH}")


if __name__ == "__main__":
    main()
