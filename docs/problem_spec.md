# Problem Spec: Telecom Churn Predictor

**Goal:** Predict which customers will cancel in the next billing cycle.
**User:** Retention team (calls customers with an offer, capacity about 500 calls/month).
**Decision supported:** Whom to call this month, highest risk first.

## Cost of mistakes
- False negative (missed churner): lose the customer's lifetime revenue (expensive).
- False positive (called a loyal customer): one call plus a small discount (cheap).

## Metrics
Accuracy is misleading (73% of customers do not churn).
Primary: recall and precision on the churn class, ROC-AUC, PR-AUC. Later: cost-based score.

## Baselines to beat
1. Always predict "no churn".
2. Logistic regression.

## Data
IBM Telco Customer Churn (7,043 rows, 21 columns, 26.5% churn). Not stored in Git.
Download: see the curl command in the README.
