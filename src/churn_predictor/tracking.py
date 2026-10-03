"""Run a cross-validated experiment and log it to MLflow."""
import mlflow

from churn_predictor.data import ROOT
from churn_predictor.validate import cv_report

TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
EXPERIMENT = "churn-predictor"


def run_experiment(run_name: str, model, X, y, params: dict | None = None) -> dict:
    """Cross-validate `model` on training data and log params + mean/std metrics."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    report = cv_report(model, X, y)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params or {})
        for metric, (mean, std) in report.items():
            mlflow.log_metric(f"{metric}_mean", mean)
            mlflow.log_metric(f"{metric}_std", std)
    return report
