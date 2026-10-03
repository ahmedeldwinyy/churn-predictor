# Phase D: Experiments (5-fold stratified CV on the training set; test set untouched)

Bar to beat: logistic regression, PR-AUC 0.661 (fold-to-fold spread about +/- 0.02).

| Model | base | +price | +addons | +interaction | +all |
|---|---|---|---|---|---|
| Logistic regression | 0.661 | 0.659 | 0.662 | 0.663 | 0.661 |
| Gradient boosting | 0.669 | 0.665 | 0.669 | 0.669 | 0.666 |
| Random forest | 0.658 | | | | |

## Repeated CV (5 folds x 3 repeats, same splits, PR-AUC)
logreg 0.6620 | logreg+interaction 0.6639 (+0.0019, 10/15 folds) | hgb 0.6681 (+0.0061, 10/15) | hgb+addons 0.6679 (+0.0059, 11/15)

## Conclusions
- Neither model complexity nor engineered features moved the score beyond noise. The signal in the 19 columns is mostly captured by a plain linear model.
- Features failed because they were redundant with existing columns: avg_paid ~ MonthlyCharges, n_addons = sum of existing columns, and trees already find the tenure x contract interaction.
- Gradient boosting shows a tiny edge (+0.006), suggestive but weak (wins 10-11 of 15 folds, folds not independent).
- Decision rule: when models are within noise, prefer the simpler one.
  Finalist: logistic regression. Challenger: gradient boosting. Phase E compares both on business cost.
- Hyperparameter tuning skipped on purpose: expected gains are below the noise level.
