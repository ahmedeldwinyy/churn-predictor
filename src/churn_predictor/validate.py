"""Cross-validation on the TRAINING set only."""
from sklearn.model_selection import StratifiedKFold, cross_validate

from churn_predictor.models import build_logreg
from churn_predictor.split import SEED, load_train

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}


def cv_report(model, X, y, n_splits: int = 5) -> dict:
    """Mean and standard deviation of each metric across folds."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    res = cross_validate(model, X, y, cv=cv, scoring=SCORING)
    return {k: (res[f"test_{k}"].mean(), res[f"test_{k}"].std()) for k in SCORING}


def main() -> None:
    train = load_train()
    X, y = train.drop(columns="churn"), train["churn"]
    models = {
        "logreg": build_logreg(),
        "logreg balanced": build_logreg(class_weight="balanced"),
    }
    for name, model in models.items():
        print(f"\n{name}")
        for metric, (mean, std) in cv_report(model, X, y).items():
            print(f"  {metric:10s} {mean:.3f} +/- {std:.3f}")


if __name__ == "__main__":
    main()
