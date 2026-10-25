# LendingClub Default Risk - Serving Layer Specification

## Service

FastAPI serves LendingClub default-risk predictions from a locally trained model artifact.

```text
GET  /health
POST /predict
GET  /drift
GET  /model-comparison
```

Local default:

```text
http://127.0.0.1:8080
```

## Prediction Request

Required fields:

| Field | Type | Meaning |
|---|---|---|
| `loan_amnt` | float | Requested loan amount |
| `annual_inc` | float | Annual income |
| `dti` | float | Debt-to-income ratio |
| `fico_range_low` | float | Lower FICO bound |
| `fico_range_high` | float | Upper FICO bound |

Recommended fields:

| Field | Type | Meaning |
|---|---|---|
| `term` | float | Loan term in months |
| `installment` | float | Monthly installment |
| `emp_length` | float | Employment length in years |
| `purpose` | string | Stated loan purpose |
| `home_ownership` | string | Housing status |
| `verification_status` | string | Income verification status |
| `delinq_2yrs` | float | Delinquencies in last two years |
| `inq_last_6mths` | float | Recent credit inquiries |
| `pub_rec` | float | Public records |
| `revol_bal` | float | Revolving balance |
| `revol_util` | float | Revolving utilization percentage |

Optional lookup field:

| Field | Type | Meaning |
|---|---|---|
| `loan_id` | int | Looks up precomputed features in the local feature store |

## Example

```json
{
  "loan_amnt": 12000,
  "annual_inc": 72000,
  "term": 36,
  "installment": 390.5,
  "dti": 18.4,
  "fico_range_low": 690,
  "fico_range_high": 694,
  "emp_length": 5,
  "purpose": "debt_consolidation",
  "home_ownership": "RENT",
  "revol_util": 42.3
}
```

## Adapter Logic

The serving adapter:

1. Optionally looks up `loan_id` in processed feature-store parquet files.
2. Applies request fields as overrides.
3. Defaults missing clean-credit-history fields to zero for synthetic applicants.
4. Recomputes LendingClub derived features: `fico_avg`, `loan_income_ratio`, `installment_income_ratio`, and `revol_balance_income_ratio`.
5. Encodes request-time categorical values with `data/processed/category_mappings.json`.
6. Aligns the request to `data/processed/feature_names.json`.
7. Imputes missing fields from `data/drift/reference_stats.json`.
8. Runs model inference.
9. Applies class-weight probability unbiasing.
10. Adds the scored record to the PSI drift window.

## Current Model

```text
mlflow_artifacts/model_entity_embedding_011460b22e6b48e1ade83348241d7ea2.pt
```

Current holdout ROC-AUC:

```text
0.70679
```

The active local champion is now the Entity Embedding model with test
ROC-AUC 0.70679. Serving applies the class-weight correction with weight 4.29
and uses decision thresholds of 0.15 and 0.30 for Low Risk, Needs Review, and
High Attention labels.
