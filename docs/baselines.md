# Baselines (5-fold stratified CV on the training set; test set untouched)

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Always "no churn" | 0.500 | 0.265 | 0.000 | 0.000 | 0.000 |
| Logistic regression | 0.846 +/- 0.013 | 0.661 +/- 0.019 | 0.655 | 0.546 | 0.595 |
| Logistic regression, balanced | 0.846 +/- 0.013 | 0.660 +/- 0.019 | 0.517 | 0.802 | 0.629 |

## Takeaways
- Accuracy is misleading: the dummy model gets 73.5% accuracy while catching 0 churners.
- class_weight="balanced" does not improve ranking (AUCs identical). It moves the operating point:
  recall 0.55 -> 0.80, precision 0.66 -> 0.52. The right trade-off is a business decision (Phase E).
- Score to beat in Phase D: PR-AUC 0.661, ROC-AUC 0.846.
