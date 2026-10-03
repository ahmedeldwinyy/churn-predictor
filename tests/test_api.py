import pytest
from fastapi.testclient import TestClient

from churn_predictor.api import app
from churn_predictor.config import DECISION_PATH, MODEL_PATH

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and DECISION_PATH.exists()), reason="train the model first"
)
client = TestClient(app)

HIGH_RISK = {
    "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.7, "TotalCharges": 151.65,
}
LOW_RISK = {
    **HIGH_RISK, "Partner": "Yes", "Dependents": "Yes", "tenure": 60, "InternetService": "DSL",
    "OnlineSecurity": "Yes", "TechSupport": "Yes", "Contract": "Two year",
    "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 55.0, "TotalCharges": 3300.0,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_valid_probability():
    r = client.post("/predict", json=HIGH_RISK)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert set(body) == {"churn_probability", "contact", "threshold", "model"}


def test_risk_goes_in_the_right_direction():
    high = client.post("/predict", json=HIGH_RISK).json()["churn_probability"]
    low = client.post("/predict", json=LOW_RISK).json()["churn_probability"]
    assert high > low


def test_new_customer_with_zero_total_charges_works():
    r = client.post("/predict", json={**HIGH_RISK, "tenure": 0, "TotalCharges": 0})
    assert r.status_code == 200


def test_invalid_input_is_rejected():
    assert client.post("/predict", json={**HIGH_RISK, "Contract": "Lifetime"}).status_code == 422
    assert client.post("/predict", json={**HIGH_RISK, "tenure": -5}).status_code == 422
    bad = {k: v for k, v in HIGH_RISK.items() if k != "Contract"}
    assert client.post("/predict", json=bad).status_code == 422
