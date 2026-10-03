"""Phase F: fit the chosen model on the training set, evaluate ONCE on the locked test set,
and save the model plus its metadata to artifacts/.

Run:  uv run python -m churn_predictor.train
Needs artifacts/decision.json (created by churn_predictor.analysis).
"""
import json
from datetime import UTC, datetime

import joblib
import pandas as pd
import sklearn

from churn_predictor.analysis import BUILDERS
from churn_predictor.business import CAPACITY_SHARE, profit_from_called
from churn_predictor.config import ARTIFACTS, DECISION_PATH, METRICS_PATH, MODEL_PATH
from churn_predictor.metrics import evaluate
from churn_predictor.split import TEST_PATH, load_train


def main() -> None:
    decision = json.loads(DECISION_PATH.read_text())
    threshold = decision["threshold"]

    train = load_train()
    X_train, y_train = train.drop(columns="churn"), train["churn"]
    model = BUILDERS[decision["model"]]().fit(X_train, y_train)

    # The one and only look at the test set.
    test = pd.read_csv(TEST_PATH, index_col="customerID")
    X_test, y_test = test.drop(columns="churn"), test["churn"]
    score = model.predict_proba(X_test)[:, 1]
    called = score >= threshold
    m = evaluate(y_test, called.astype(int), score)

    profit = profit_from_called(y_test, called)
    everyone = profit_from_called(y_test, [True] * len(y_test))
    metrics = {
        "model": decision["model"],
        "threshold": threshold,
        "trained_on": datetime.now(tz=UTC).date().isoformat(),
        "sklearn_version": sklearn.__version__,
        "n_train": len(train),
        "n_test": len(test),
        "test": {k: round(float(v), 4) if isinstance(v, float) else v for k, v in m.items()},
        "contact_share": round(float(called.mean()), 4),
        "lift_vs_random": round(float(m["precision"] / y_test.mean()), 2),
        "profit_per_1000_test": round(profit, 1),
        "profit_per_1000_random_contact": round(everyone * CAPACITY_SHARE, 1),
        "profit_per_1000_cv_estimate": decision["cv_profit_per_1000"],
        "assumptions": decision["assumptions"],
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"Model: {metrics['model']}   threshold: {threshold}")
    print(f"TEST  ROC-AUC {m['roc_auc']:.3f}  PR-AUC {m['pr_auc']:.3f}  "
          f"precision {m['precision']:.3f}  recall {m['recall']:.3f}  contacts {called.mean():.1%}")
    print(f"TEST  profit/1000 customers ${profit:,.0f} "
          f"(cross-validation estimate was ${decision['cv_profit_per_1000']:,.0f}; "
          f"random contact would earn ${everyone * CAPACITY_SHARE:,.0f})")
    print(f"Saved {MODEL_PATH} and {METRICS_PATH}")


if __name__ == "__main__":
    main()
