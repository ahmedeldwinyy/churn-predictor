"""Experiment definitions. Add new runs here, one change at a time.

Usage: uv run python -m churn_predictor.experiments hgb_price logreg_all
       uv run python -m churn_predictor.experiments d4      (all feature experiments)
"""
import sys

from churn_predictor.models import build_forest, build_hgb, build_logreg
from churn_predictor.split import load_train
from churn_predictor.tracking import run_experiment

FEATURE_SETS = {
    "base": (),
    "price": ("price",),
    "addons": ("addons",),
    "interaction": ("interaction",),
    "all": ("price", "addons", "interaction"),
}
BUILDERS = {"logreg": build_logreg, "hgb": build_hgb}


def _make(builder, groups):
    return lambda: builder(groups=groups)


RUNS = {
    "logreg_balanced": (
        lambda: build_logreg(class_weight="balanced"),
        {"model": "logreg", "class_weight": "balanced", "features": "base"},
    ),
    "forest": (build_forest, {"model": "random_forest", "features": "base"}),
}
D4 = []
for model_name, builder in BUILDERS.items():
    for feat_name, groups in FEATURE_SETS.items():
        run_name = model_name if feat_name == "base" else f"{model_name}_{feat_name}"
        RUNS[run_name] = (_make(builder, groups), {"model": model_name, "features": feat_name})
        if feat_name != "base":
            D4.append(run_name)


def main(names: list[str]) -> None:
    names = D4 if names == ["d4"] else names
    unknown = [n for n in names if n not in RUNS]
    if not names or unknown:
        sys.exit(f"Choose from: {', '.join(RUNS)}, d4  (unknown: {unknown})")
    train = load_train()
    X, y = train.drop(columns="churn"), train["churn"]
    for name in names:
        builder, params = RUNS[name]
        report = run_experiment(name, builder(), X, y, params)
        print(f"{name:20s} PR-AUC {report['pr_auc'][0]:.3f} +/- {report['pr_auc'][1]:.3f}   "
              f"ROC-AUC {report['roc_auc'][0]:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:])
