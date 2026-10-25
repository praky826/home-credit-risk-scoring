# LendingClub Default Risk - Machine Learning & Production Scoring Suite

An end-to-end credit risk scoring system for unsecured LendingClub personal-loan default prediction. The project keeps the original production ML architecture: feature engineering, model training, evaluation, FastAPI serving, PSI drift monitoring, tests, Docker support, and a React underwriter frontend.

## Current Status

- Active dataset: LendingClub accepted loans from Kaggle `wordsforthewise/lending-club`.
- Active raw file: `data/raw/lendingclub/accepted_2007_to_2018Q4.csv.gz`.
- Active target: `TARGET=1` for final bad outcomes such as `Charged Off` or `Default`; `TARGET=0` for `Fully Paid`.
- Active feature count: 30 LendingClub origination and engineered features.
- Current local serving model: XGBoost artifact at `mlflow_artifacts/model_xgboost_213ee9bcca664eb78cfb4bb7d6de27f5.pkl`.

## First-Pass Results

These are migration smoke-test results, not final tuned benchmark claims.

| Model | Validation ROC-AUC | Test ROC-AUC | Test KS | Test Brier | Status |
|---|---:|---:|---:|---:|---|
| Entity Embedding | 0.70184 | 0.70679 | 0.29967 | 0.26739 | Neural challenger |
| LightGBM | 0.70358 | 0.70616 | 0.29888 | 0.24802 | Production champion |
| XGBoost | 0.70291 | 0.70487 | 0.29824 | 0.24073 | GPU baseline |
| MLP | 0.69435 | 0.69755 | 0.28861 | 0.23261 | Neural baseline |
| TabTransformer | 0.65182 | 0.65670 | 0.22281 | 0.21660 | Experimental neural model |

Training split used 970,631 rows, validation used 121,328 rows, and holdout test used 121,330 rows.

## Feature Contract

The model uses LendingClub application/origination fields such as:

```text
loan_amnt
term
installment
annual_inc
dti
fico_range_low
fico_range_high
emp_length
home_ownership
verification_status
purpose
addr_state
delinq_2yrs
inq_last_6mths
open_acc
pub_rec
revol_bal
revol_util
total_acc
mort_acc
pub_rec_bankruptcies
```

The feature builder also creates:

```text
fico_avg
loan_income_ratio
installment_income_ratio
revol_balance_income_ratio
credit_history_years
issue_year
issue_month
```

Post-origination repayment and recovery columns are excluded to avoid leakage.

## Project Structure

```text
configs/                  Model, feature, serving, and MLOps configuration
data/raw/lendingclub/      LendingClub Kaggle files
data/processed/            Generated feature matrix and metadata
src/features/              LendingClub feature builder plus legacy modules
src/models/                XGBoost, LightGBM, MLP, entity embedding, TabTransformer wrappers
src/evaluation/            Metrics and plots
src/monitoring/            PSI drift monitoring and alerting
src/serving/               FastAPI prediction service
frontend/                  React + TypeScript underwriting UI
tests/                     Feature, model, serving, and monitoring tests
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install kaggle
```

## Download Data

Configure Kaggle credentials, then run:

```powershell
.\.venv\Scripts\kaggle.exe datasets download -d wordsforthewise/lending-club -p data/raw/lendingclub --unzip
```

The active config expects:

```text
data/raw/lendingclub/accepted_2007_to_2018Q4.csv.gz
```

## Build Features

```powershell
$env:PYTHONPATH="."
.venv\Scripts\python -m src.features.pipeline
```

Generated artifacts:

```text
data/processed/train_features.parquet
data/processed/test_features.parquet
data/processed/feature_names.json
data/processed/category_mappings.json
data/drift/reference_stats.json
```

## Train Models

```powershell
$env:PYTHONPATH="."
.venv\Scripts\python -m src.models.train --model xgboost
.venv\Scripts\python -m src.models.train --model lgbm
```

For final metrics, rerun tree models with tuning:

```powershell
.venv\Scripts\python -m src.models.train --model xgboost --tune
.venv\Scripts\python -m src.models.train --model lgbm --tune
```

## Run Backend

```powershell
$env:PYTHONPATH="."
.venv\Scripts\python -m uvicorn src.serving.app:app --host 127.0.0.1 --port 8080 --reload
```

Endpoints:

```text
GET  http://127.0.0.1:8080/health
POST http://127.0.0.1:8080/predict
GET  http://127.0.0.1:8080/drift
GET  http://127.0.0.1:8080/model-comparison
```

Example prediction payload:

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

## Run Frontend

```powershell
$env:VITE_API_BASE_URL="http://127.0.0.1:8080"
npm --prefix frontend run dev -- --host 127.0.0.1
```

Open:

```text
http://127.0.0.1:5173/
```

## Test

```powershell
$env:PYTHONPATH="."
.venv\Scripts\pytest -q
npm --prefix frontend run build
```

Current local verification:

```text
26 Python tests passed
Frontend production build passed
Live FastAPI prediction passed
```
