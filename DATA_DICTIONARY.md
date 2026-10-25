# LendingClub Default Risk - Dataset Reference

This project now uses LendingClub accepted-loan data for unsecured personal-loan default prediction.

## Source File

| File | Grain | Notes |
|---|---|---|
| `data/raw/lendingclub/accepted_2007_to_2018Q4.csv.gz` | 1 row per accepted LendingClub loan | Downloaded from Kaggle `wordsforthewise/lending-club` |

The rejected-loan file is downloaded but not used for supervised default modeling because it does not contain final repayment outcomes.

## Target

The binary target is derived from `loan_status`.

| `TARGET` | Loan statuses |
|---|---|
| 0 | `Fully Paid`, `Does not meet the credit policy. Status:Fully Paid` |
| 1 | `Charged Off`, `Default`, `Does not meet the credit policy. Status:Charged Off` |

Rows with non-final statuses such as `Current`, `Issued`, or grace/late interim states are excluded from the first-pass supervised dataset.

## Model Features

The active feature builder keeps fields that are available at or near origination time.

| Feature | Meaning |
|---|---|
| `loan_amnt` | Requested loan amount |
| `term` | Loan term in months, parsed from strings such as `36 months` |
| `installment` | Scheduled monthly payment |
| `annual_inc` | Borrower annual income |
| `dti` | Debt-to-income ratio |
| `fico_range_low`, `fico_range_high` | Borrower FICO range at origination |
| `emp_length` | Employment length parsed to 0-10 years |
| `home_ownership` | Encoded housing status |
| `verification_status` | Encoded income verification status |
| `purpose` | Encoded stated loan purpose |
| `addr_state` | Encoded borrower state |
| `delinq_2yrs` | Delinquencies in the last two years |
| `inq_last_6mths` | Recent credit inquiries |
| `open_acc` | Open credit accounts |
| `pub_rec` | Public records |
| `revol_bal` | Revolving credit balance |
| `revol_util` | Revolving utilization percentage |
| `total_acc` | Total credit accounts |
| `initial_list_status` | Encoded initial listing status |
| `application_type` | Individual or joint application |
| `mort_acc` | Mortgage account count |
| `pub_rec_bankruptcies` | Public record bankruptcies |

## Engineered Features

| Feature | Definition |
|---|---|
| `fico_avg` | Average of `fico_range_low` and `fico_range_high` |
| `loan_income_ratio` | `loan_amnt / annual_inc` |
| `installment_income_ratio` | `installment * 12 / annual_inc` |
| `revol_balance_income_ratio` | `revol_bal / annual_inc` |
| `credit_history_years` | Years between `earliest_cr_line` and `issue_d` |
| `issue_year`, `issue_month` | Origination period derived from `issue_d` |

## Leakage Columns Excluded

The feature builder drops repayment, recovery, and post-origination fields including:

```text
loan_status
funded_amnt
funded_amnt_inv
last_pymnt_d
last_pymnt_amnt
total_pymnt
total_pymnt_inv
total_rec_prncp
total_rec_int
total_rec_late_fee
recoveries
collection_recovery_fee
last_credit_pull_d
last_fico_range_high
last_fico_range_low
out_prncp
out_prncp_inv
next_pymnt_d
debt_settlement_flag
settlement_status
```

## Artifacts

The pipeline writes the same downstream contract used by training, serving, and monitoring:

```text
data/processed/train_features.parquet
data/processed/test_features.parquet
data/processed/feature_names.json
data/processed/cat_feature_indices.json
data/processed/cat_cardinalities.json
data/processed/category_mappings.json
data/drift/reference_stats.json
```

Current first-pass feature matrix:

| Dataset artifact | Rows | Features |
|---|---:|---:|
| `train_features.parquet` | 1,213,289 | 30 plus `TARGET` |
| `test_features.parquet` | 134,810 | 30 |
