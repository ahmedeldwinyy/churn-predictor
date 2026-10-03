
## Data

The dataset is not stored in Git. Download it once:

    mkdir -p data/raw
    curl -L -o data/raw/telco_churn.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

Then clean it:

    uv run python -m churn_predictor.prepare

See `docs/problem_spec.md` for the problem definition and `docs/data_notes.md` for data decisions.
