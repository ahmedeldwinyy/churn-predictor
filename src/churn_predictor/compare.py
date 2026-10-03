"""Paired comparison of top candidates with repeated cross-validation (train set only)."""
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from churn_predictor.experiments import RUNS
from churn_predictor.split import SEED, load_train

CANDIDATES = ["logreg", "logreg_interaction", "hgb", "hgb_addons"]


def main() -> None:
    train = load_train()
    X, y = train.drop(columns="churn"), train["churn"]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)  # same 15 splits for all
    scores = {
        name: cross_val_score(RUNS[name][0](), X, y, cv=cv, scoring="average_precision")
        for name in CANDIDATES
    }
    base = scores["logreg"]
    for name, s in scores.items():
        diff = s - base
        print(f"{name:20s} PR-AUC {s.mean():.4f}   diff vs logreg {diff.mean():+.4f}   wins {(diff > 0).sum()}/{len(diff)} folds")


if __name__ == "__main__":
    main()
