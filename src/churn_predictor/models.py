"""Model pipelines. Every learning step lives INSIDE the pipeline."""
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
PASSTHROUGH = ["SeniorCitizen"]
CATEGORICAL = [
    "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
# gender is intentionally NOT listed: no signal + fairness concern (see docs/data_notes.md)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL),
            ("pass", "passthrough", PASSTHROUGH),
        ],
        remainder="drop",
    )


def build_logreg(**kwargs) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, **kwargs)),
        ]
    )
