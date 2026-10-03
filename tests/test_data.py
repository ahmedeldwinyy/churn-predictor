import pandas as pd
import pytest

from churn_predictor.data import RAW_PATH, clean, load_raw


def make_df():
    return pd.DataFrame(
        {
            "customerID": ["a", "b", "c"],
            "tenure": [0, 5, 10],
            "TotalCharges": [" ", "100.5", "250"],
            "Churn": ["No", "Yes", "No"],
        }
    )


def test_blank_total_charges_becomes_zero():
    out = clean(make_df())
    assert out.loc["a", "TotalCharges"] == 0.0
    assert out["TotalCharges"].dtype == float


def test_target_is_binary():
    out = clean(make_df())
    assert out["churn"].tolist() == [0, 1, 0]
    assert "Churn" not in out.columns


def test_customer_id_is_index():
    out = clean(make_df())
    assert out.index.name == "customerID"
    assert "customerID" not in out.columns


def test_input_not_modified():
    df = make_df()
    before = df.copy()
    clean(df)
    pd.testing.assert_frame_equal(df, before)


def test_bad_value_fails_loudly():
    df = make_df()
    df.loc[1, "TotalCharges"] = "abc"
    with pytest.raises(ValueError):
        clean(df)


@pytest.mark.skipif(not RAW_PATH.exists(), reason="raw data not downloaded")
def test_real_data_is_valid():
    out = clean(load_raw())
    assert out.shape == (7043, 20)
    assert out.index.is_unique
    assert out.isna().sum().sum() == 0
    assert set(out["churn"].unique()) == {0, 1}
    assert (out.loc[out["tenure"] == 0, "TotalCharges"] == 0).all()
