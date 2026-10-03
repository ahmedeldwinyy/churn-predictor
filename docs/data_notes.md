# Data Notes: Telco Churn

## Dataset
7,043 customers, 21 columns, no duplicates, `customerID` unique. Churn rate 26.5%.

## Cleaning decisions (stateless fixes only)
1. `TotalCharges` is text. 11 rows contain a blank string (all have tenure = 0, Churn = No: new customers, not yet billed).
   Decision: convert to float and fill blanks with 0.0. Reason: business logic (nothing billed yet), no learning from data.
2. `Churn` Yes/No becomes the binary target 1/0.
3. `SeniorCitizen` is already 0/1 and model-ready. No change.
4. `customerID` is an identifier. Kept as the row index, never used as a feature.
5. `gender` has no signal (26.9% vs 26.2%) and raises fairness concerns. Excluded from features.

## Decisions deferred to the modeling pipeline (Phase C)
- Encoding categoricals, scaling, and whether to collapse "No internet service" / "No phone service" into "No".

## Findings so far
- Contract: month-to-month 42.7% churn vs two-year 2.8%.
- Tenure: churn falls from about 53% (0-6 months) to about 10% (48+ months).
- InternetService: fiber optic 41.9% vs no internet 7.4%.
- PaymentMethod: electronic check 45.3% vs automatic credit card 15.2%.
- Churners pay more per month (median about 80 vs 65).

## Open questions (to verify)
- Do add-ons (OnlineSecurity, TechSupport) lower churn among internet customers only?
- How much of the Contract effect is really a tenure effect?
- TotalCharges is close to tenure x MonthlyCharges (redundant). Does the gap (price change) predict churn?

## Answered
- Add-ons: among internet customers only, OnlineSecurity (41.8% without vs 14.6% with) and TechSupport (41.6% vs 15.2%) still show large gaps. Correlation only, not proof of causation.
- Contract vs tenure: average tenure is 18 (month-to-month), 42 (one year) and 56.7 (two year) months, so the two are entangled. Checked within tenure bins (see next entry).
- Contract vs tenure result: within every tenure bin month-to-month churn is far higher (51% vs 10% vs 0% in the first 12 months), so Contract has an effect of its own. Tenure matters mostly for month-to-month (51% falling to 26%); for one/two-year contracts it is nearly flat. This is an interaction effect: try tree models and/or an explicit interaction feature in Phase D.
