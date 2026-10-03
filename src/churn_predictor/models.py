"""Model pipelines. Every learning step lives INSIDE the pipeline."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn_predictor.features import add_features, new_columns
from churn_predictor.split import SEED

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
PASSTHROUGH = ["SeniorCitizen"]
CATEGORICAL = [
    "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
# gender is intentionally NOT listed: no signal + fairness concern (see docs/data_notes.md)


def build_preprocessor(groups: tuple = ()) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC + new_columns(groups)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
                CATEGORICAL,
            ),
            ("pass", "passthrough", PASSTHROUGH),
        ],
        remainder="drop",
    )


def _pipeline(model, groups: tuple) -> Pipeline:
    groups = tuple(groups)
    return Pipeline(
        [
            ("feat", FunctionTransformer(add_features, kw_args={"groups": groups})),
            ("prep", build_preprocessor(groups)),
            ("model", model),
        ]
    )


def build_logreg(groups: tuple = (), **kwargs) -> Pipeline:
    return _pipeline(LogisticRegression(max_iter=1000, **kwargs), groups)


def build_forest(groups: tuple = (), **kwargs) -> Pipeline:
    params = {"n_estimators": 300, "min_samples_leaf": 5, "n_jobs": -1, "random_state": SEED}
    return _pipeline(RandomForestClassifier(**{**params, **kwargs}), groups)


def build_hgb(groups: tuple = (), **kwargs) -> Pipeline:
    params = {"learning_rate": 0.05, "max_iter": 200, "max_depth": 3, "random_state": SEED}
    return _pipeline(HistGradientBoostingClassifier(**{**params, **kwargs}), groups)
