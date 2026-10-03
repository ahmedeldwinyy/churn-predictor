# Model Card: Telco Churn Predictor

## Purpose
Rank customers by their risk of cancelling in the next billing cycle, so the retention team
can contact the riskiest ones first. It supports a decision (whom to contact); it does not
decide anything by itself.

## Model
- Algorithm: **logreg** (scikit-learn pipeline: feature step, scaling/one-hot encoding, classifier)
- Trained on 5,634 customers on 2026-10-03 (scikit-learn 1.9.1)
- Decision threshold: **0.6** on the churn probability. It was chosen on the training set with
  out-of-fold predictions, using the cost assumptions below and the team's contact capacity.

## Data
IBM Telco Customer Churn sample (7,043 customers, 21 columns, 26.5% churn).
Cleaning and decisions: `docs/data_notes.md`. The `gender` column is NOT used.

## Performance (held-out test set, 1,409 customers, evaluated once)
| Metric | Value |
|---|---|
| ROC-AUC | 0.842 |
| PR-AUC | 0.634 |
| Precision at threshold | 0.710 |
| Recall at threshold | 0.398 |
| Share of customers contacted | 14.9% |
| Lift vs contacting at random | 2.67x |

## Business value (per 1,000 customers, using the assumptions below)
| Policy | Expected profit |
|---|---|
| This model (test set) | $8,964 |
| Cross-validation estimate made before seeing the test set | $9,094 |
| Contacting the same number of customers at random | $1,028 |

Assumptions (placeholders, replace with real company numbers): a saved customer is worth
$400; 30% of contacted churners accept the offer; each contact costs
$25; the team can contact at most 15% of customers.

## Limitations
- The model finds correlation, not causes. Add-on services predict loyalty, but that does not prove
  that offering them reduces churn.
- Recall is about 0.398: most churners are not in the contacted group because capacity is limited.
- Customers on long contracts are rarely flagged, and the data holds few of their churn cases.
- Trained on one snapshot of one dataset; behaviour on a different market or period is unknown.
- The dollar figures depend entirely on the assumptions above.

## Fairness
`gender` is excluded. `SeniorCitizen` is used as a feature and should be reviewed for fair treatment
before any real deployment. Contact decisions only offer a discount, but this check is still required.

## Monitoring and retraining
- Weekly: run `churn_predictor.monitor` on the latest customers. Any feature with PSI above 0.25 is an alert.
- Monthly, once real outcomes arrive: recompute PR-AUC and recall on the newest customers.
- Retrain if PR-AUC drops by more than 0.05 from the value above, or if drift alerts persist.
