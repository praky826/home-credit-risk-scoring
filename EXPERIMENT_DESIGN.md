# LendingClub Default Risk - Experiment Protocol

## Objective

Predict final default/charge-off risk for unsecured LendingClub personal loans using borrower and loan-origination features.

## Target Definition

`TARGET` is derived from `loan_status`:

- `0`: final paid outcome
- `1`: final bad outcome such as `Charged Off` or `Default`

Non-final statuses are excluded from supervised training.

## Split Strategy

The feature builder sorts accepted loans by `issue_d` before writing processed features. The training loader then creates a stable temporal-style split:

| Split | Rows |
|---|---:|
| Train | 970,631 |
| Validation | 121,328 |
| Test | 121,330 |

This better reflects a lending setting where the model is trained on older loans and evaluated on newer loans.

## Current First-Pass Metrics

| Model | Validation ROC-AUC | Test ROC-AUC | Test KS | Test Brier | Calibration Slope |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.70302 | 0.70647 | 0.30042 | 0.24021 | 0.68141 |
| LightGBM | 0.70358 | 0.70616 | 0.29888 | 0.24802 | 0.66812 |

These are untuned migration metrics. They prove the LendingClub pipeline, split, training code, and serving artifact contract work. They are not final benchmark claims.

## Next Experiments

1. Tune XGBoost and LightGBM on the LendingClub split.
2. Recompute class weights from the active training split.
3. Run probability calibration analysis.
4. Refit low/review/high decision thresholds.
5. Retrain neural models only after tree baselines are stable.
6. Regenerate plots and final model comparison artifacts.

## Leakage Policy

Post-origination repayment, recovery, collection, and final outcome columns are excluded from training. LendingClub `grade`, `sub_grade`, and `int_rate` remain disabled by default because they reflect LendingClub's own underwriting/pricing decisions.
