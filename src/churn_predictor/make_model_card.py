"""Generate docs/model_card.md from artifacts/metrics.json (so the numbers are never typed by hand).

Run:  uv run python -m churn_predictor.make_model_card
"""
import json

from churn_predictor.config import METRICS_PATH
from churn_predictor.data import ROOT

CARD_PATH = ROOT / "docs" / "model_card.md"

TEMPLATE = """# Model Card: Telco Churn Predictor

## Purpose
Rank customers by their risk of cancelling in the next billing cycle, so the retention team
can contact the riskiest ones first. It supports a decision (whom to contact); it does not
decide anything by itself.

## Model
- Algorithm: **{model}** (scikit-learn pipeline: feature step, scaling/one-hot encoding, classifier)
- Trained on {n_train:,} customers on {trained_on} (scikit-learn {sklearn_version})
- Decision threshold: **{threshold}** on the churn probability. It was chosen on the training set with
  out-of-fold predictions, using the cost assumptions below and the team's contact capacity.

## Data
IBM Telco Customer Churn sample (7,043 customers, 21 columns, 26.5% churn).
Cleaning and decisions: `docs/data_notes.md`. The `gender` column is NOT used.

## Performance (held-out test set, {n_test:,} customers, evaluated once)
| Metric | Value |
|---|---|
| ROC-AUC | {roc_auc} |
| PR-AUC | {pr_auc} |
| Precision at threshold | {precision} |
| Recall at threshold | {recall} |
| Share of customers contacted | {contact_share:.1%} |
| Lift vs contacting at random | {lift}x |

## Business value (per 1,000 customers, using the assumptions below)
| Policy | Expected profit |
|---|---|
| This model (test set) | ${profit_test:,.0f} |
| Cross-validation estimate made before seeing the test set | ${profit_cv:,.0f} |
| Contacting the same number of customers at random | ${profit_random:,.0f} |

Assumptions (placeholders, replace with real company numbers): a saved customer is worth
${save_value:.0f}; {offer_success:.0%} of contacted churners accept the offer; each contact costs
${call_cost:.0f}; the team can contact at most {capacity:.0%} of customers.

## Limitations
- The model finds correlation, not causes. Add-on services predict loyalty, but that does not prove
  that offering them reduces churn.
- Recall is about {recall}: most churners are not in the contacted group because capacity is limited.
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
"""


def main() -> None:
    m = json.loads(METRICS_PATH.read_text())
    t, a = m["test"], m["assumptions"]
    card = TEMPLATE.format(
        model=m["model"], n_train=m["n_train"], n_test=m["n_test"], trained_on=m["trained_on"],
        sklearn_version=m["sklearn_version"], threshold=m["threshold"],
        roc_auc=f"{t['roc_auc']:.3f}", pr_auc=f"{t['pr_auc']:.3f}",
        precision=f"{t['precision']:.3f}", recall=f"{t['recall']:.3f}",
        contact_share=m["contact_share"], lift=m["lift_vs_random"],
        profit_test=m["profit_per_1000_test"], profit_cv=m["profit_per_1000_cv_estimate"],
        profit_random=m["profit_per_1000_random_contact"],
        save_value=a["save_value"], offer_success=a["offer_success"],
        call_cost=a["call_cost"], capacity=a["capacity_share"],
    )
    CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    CARD_PATH.write_text(card)
    print(f"Wrote {CARD_PATH}")


if __name__ == "__main__":
    main()
