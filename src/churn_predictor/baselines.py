"""Baseline models. Every real model must beat these."""
from sklearn.dummy import DummyClassifier

from churn_predictor.metrics import evaluate
from churn_predictor.split import load_train


def main() -> None:
    train = load_train()
    X, y = train.drop(columns="churn"), train["churn"]

    # Always predicts the majority class ("no churn")
    model = DummyClassifier(strategy="most_frequent").fit(X, y)
    pred = model.predict(X)
    score = model.predict_proba(X)[:, 1]

    for name, value in evaluate(y, pred, score).items():
        print(f"{name:10s} {value:.4f}" if isinstance(value, float) else f"{name:10s} {value}")


if __name__ == "__main__":
    main()
