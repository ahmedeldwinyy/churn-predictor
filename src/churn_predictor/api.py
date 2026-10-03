"""FastAPI service.  Run locally:  uv run uvicorn churn_predictor.api:app --reload
Docs (try it in the browser):  http://127.0.0.1:8000/docs
"""
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from churn_predictor.predict import load_model, predict_df

YN = Literal["Yes", "No"]
YNI = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    """One customer, using the same columns and labels as the training data."""

    gender: str | None = None  # accepted but never used by the model
    SeniorCitizen: Literal[0, 1]
    Partner: YN
    Dependents: YN
    tenure: int = Field(ge=0, le=120, description="months as a customer")
    PhoneService: YN
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YNI
    OnlineBackup: YNI
    DeviceProtection: YNI
    TechSupport: YNI
    StreamingTV: YNI
    StreamingMovies: YNI
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YN
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
                "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
                "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check", "MonthlyCharges": 70.7, "TotalCharges": 151.65,
            }
        }
    }


class Prediction(BaseModel):
    churn_probability: float
    contact: bool
    threshold: float
    model: str


app = FastAPI(
    title="Telco Churn Predictor",
    version="1.0.0",
    description="Scores a customer's risk of cancelling. `contact` is true when the "
    "probability passes the cost-based threshold chosen for the retention team.",
)


@app.get("/health")
def health() -> dict:
    _, decision = load_model()
    return {"status": "ok", "model": decision["model"], "threshold": decision["threshold"]}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    result = predict_df(pd.DataFrame([customer.model_dump()])).iloc[0]
    _, decision = load_model()
    return Prediction(
        churn_probability=float(result["churn_probability"]),
        contact=bool(result["contact"]),
        threshold=decision["threshold"],
        model=decision["model"],
    )
