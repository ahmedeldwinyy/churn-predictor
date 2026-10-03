import numpy as np
import pandas as pd
import pytest

from churn_predictor.features import ADDONS, add_features
from churn_predictor.models import build_logreg
from churn_predictor.split import TRAIN_PATH, load_train


def addon_df(values):
    d = pd.DataFrame(
        {
            "tenure": [0, 10],
            "TotalCharges": [0.0, 500.0],
            "MonthlyCharges": [50.0, 60.0],
            "Contract": ["Month-to-month", "Two year"],
        }
    )
    for a in ADDONS:
        d[a] = values
    return d


def test_price_features_handle_tenure_zero():
    out = add_features(addon_df(["Yes", "No"]), ("price",))
    assert np.isfinite(out[["avg_paid", "price_gap"]]).all().all()
    assert out.loc[0, "avg_paid"] == 50.0  # falls back to current price
    assert out.loc[1, "avg_paid"] == 50.0
    assert out.loc[1, "price_gap"] == 10.0


def test_n_addons_counts_yes():
    out = add_features(addon_df(["Yes", "No"]), ("addons",))
    assert out["n_addons"].tolist() == [6, 0]


def test_tenure_if_mtm_only_for_month_to_month():
    d = addon_df(["No", "No"])
    d["tenure"] = [3, 10]
    out = add_features(d, ("interaction",))
    assert out["tenure_if_mtm"].tolist() == [3, 0]


def test_add_features_does_not_modify_input():
    d = addon_df(["Yes", "No"])
    before = d.copy()
    add_features(d, ("price", "addons", "interaction"))
    pd.testing.assert_frame_equal(d, before)


@pytest.mark.skipif(not TRAIN_PATH.exists(), reason="run the split first")
def test_pipeline_fits_with_all_features():
    train = load_train().head(500)
    X, y = train.drop(columns="churn"), train["churn"]
    model = build_logreg(groups=("price", "addons", "interaction")).fit(X, y)
    assert model.predict_proba(X).shape == (500, 2)
