# Telco Churn Predictor

An end-to-end machine learning project: from raw customer data to a deployed API that tells a
retention team **whom to contact first**, with the business cost of mistakes built in.

- **Problem:** a telecom company loses about 27% of customers; the retention team can only contact a limited share each month.
- **Solution:** a calibrated logistic regression pipeline ranks customers by churn risk. The contact threshold is chosen by expected profit under the team's capacity, not by accuracy.
- **Honest evaluation:** fixed stratified split, cross-validation on the training set only, the test set touched exactly once. Results and limits: [`docs/model_card.md`](docs/model_card.md).
- **Production habits:** tested code, pipelines with no data leakage, experiment tracking (MLflow), typed API, Docker, CI, drift monitoring.

## Live demo
API docs: `https://churn-predictor-xxxx.onrender.com/docs`  (free hosting may take about a minute to wake up)

## Quickstart
```bash
uv sync                                   # install dependencies
mkdir -p data/raw
curl -L -o data/raw/telco_churn.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

uv run python -m churn_predictor.prepare  # clean the data
uv run python -m churn_predictor.split    # fixed train/test split
uv run python -m churn_predictor.analysis # cost-based threshold + error analysis (train set only)
uv run python -m churn_predictor.train    # fit, evaluate once on test, save artifacts/
uv run pytest                             # run the tests
```

## Use the model
```bash
# API (interactive docs at http://127.0.0.1:8000/docs)
uv run uvicorn churn_predictor.api:app --reload

# Batch: rank a CSV of customers by risk
uv run python -m churn_predictor.predict data/raw/telco_churn.csv --top 50

# Docker
docker build -t churn-api .
docker run --rm -p 8000:8000 churn-api

# Drift check on new data
uv run python -m churn_predictor.monitor new_customers.csv
```

## Project layout
| Path | What it is |
|---|---|
| `src/churn_predictor/data.py` | loading and stateless cleaning |
| `src/churn_predictor/split.py`, `metrics.py` | fixed split, one scoring function for every model |
| `src/churn_predictor/features.py`, `models.py` | feature engineering and sklearn pipelines |
| `src/churn_predictor/experiments.py`, `tracking.py` | cross-validated experiments logged to MLflow |
| `src/churn_predictor/business.py`, `analysis.py` | cost model, capacity-aware threshold, calibration, error analysis |
| `src/churn_predictor/train.py`, `predict.py`, `api.py` | final training, batch scoring, FastAPI service |
| `src/churn_predictor/monitor.py` | PSI drift detection |
| `artifacts/` | trained model, chosen threshold, test metrics (committed so the image is self-contained) |
| `docs/` | problem spec, data notes, baselines, experiments, model card |

## Key decisions (details in `docs/`)
- Accuracy is misleading (73% of customers stay); the model is judged by PR-AUC, recall and expected profit.
- More complex models and engineered features gave no gain beyond noise, so the simpler model was kept.
- The dollar assumptions in `business.py` are placeholders and must be replaced with real company numbers.
