"""Paths to the files that travel with the deployed model."""
from churn_predictor.data import ROOT

ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "model.joblib"
DECISION_PATH = ARTIFACTS / "decision.json"
METRICS_PATH = ARTIFACTS / "metrics.json"
