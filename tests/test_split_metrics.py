import pandas as pd
import pytest

from churn_predictor.metrics import evaluate
from churn_predictor.split import make_split


def fake_df():
    return pd.DataFrame(
        {"x": range(200), "churn": [1] * 50 + [0] * 150},
        index=[f"id{i}" for i in range(200)],
    )


def test_split_is_stratified():
    train, test = make_split(fake_df())
    assert abs(train["churn"].mean() - 0.25) < 0.01
    assert abs(test["churn"].mean() - 0.25) < 0.01


def test_split_has_no_overlap_and_loses_no_rows():
    df = fake_df()
    train, test = make_split(df)
    assert set(train.index).isdisjoint(test.index)
    assert len(train) + len(test) == len(df)


def test_split_is_deterministic():
    a, _ = make_split(fake_df())
    b, _ = make_split(fake_df())
    assert a.index.tolist() == b.index.tolist()


def test_metrics_hand_example():
    out = evaluate([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.6, 0.7, 0.9])
    assert (out["tn"], out["fp"], out["fn"], out["tp"]) == (1, 1, 0, 2)
    assert out["recall"] == 1.0
    assert out["accuracy"] == 0.75
    assert out["roc_auc"] == pytest.approx(1.0)


def test_metrics_all_negative_predictions_do_not_crash():
    out = evaluate([0, 1, 0, 1], [0, 0, 0, 0], [0.5, 0.5, 0.5, 0.5])
    assert out["recall"] == 0.0
    assert out["precision"] == 0.0
