# LendingClub Default Risk - File Specifications

## Active Configuration

| File | Purpose |
|---|---|
| `configs/features.yaml` | Selects the LendingClub dataset and raw accepted-loans path |
| `configs/serving.yaml` | Selects the local serving model artifact, drift settings, feature-store path, and decision thresholds |
| `configs/mlops.yaml` | MLflow experiment and registry names |
| `configs/xgboost.yaml` | XGBoost training parameters |
| `configs/lgbm.yaml` | LightGBM training parameters |

## Feature Engineering

| File | Status | Purpose |
|---|---|---|
| `src/features/lendingclub.py` | Active | Parses LendingClub raw data, builds `TARGET`, removes leakage fields, encodes categoricals, engineers ratios |
| `src/features/pipeline.py` | Active | Dispatches to the LendingClub builder when `dataset: lendingclub` |
| `src/features/utils.py` | Active | Shared imputation, encoding, memory reduction, and safe division helpers |
| `src/features/application.py` and auxiliary legacy modules | Inactive | Kept only as historical reference until a later cleanup |

## Generated Data Artifacts

| File | Purpose |
|---|---|
| `data/processed/train_features.parquet` | LendingClub model matrix with `loan_id`, `TARGET`, and 30 features |
| `data/processed/test_features.parquet` | LendingClub holdout feature-store copy without `TARGET` |
| `data/processed/feature_names.json` | Exact model input column order |
| `data/processed/cat_feature_indices.json` | Categorical feature positions for neural models |
| `data/processed/cat_cardinalities.json` | Cardinalities for categorical features |
| `data/processed/category_mappings.json` | Request-time categorical encoding maps |
| `data/drift/reference_stats.json` | Numeric reference distributions for PSI monitoring |

## Training

| File | Purpose |
|---|---|
| `src/models/train.py` | Loads processed data, creates stable ID-based splits, trains selected model, logs MLflow metrics/artifacts |
| `src/models/xgboost_model.py` | XGBoost wrapper |
| `src/models/lgbm_model.py` | LightGBM wrapper |
| `src/models/mlp.py` | MLP wrapper |
| `src/models/entity_embedding.py` | Entity embedding wrapper |
| `src/models/tab_transformer.py` | TabTransformer wrapper |

## Serving

| File | Purpose |
|---|---|
| `src/serving/app.py` | FastAPI app |
| `src/serving/router.py` | Prediction, drift, health, and model-comparison endpoints |
| `src/serving/schemas.py` | Dynamic LendingClub request schema and response models |
| `src/serving/feature_store.py` | Local parquet lookup by `loan_id` |
| `src/serving/calibration.py` | Probability adjustment for class-weighted training |
| `src/serving/model_loader.py` | Loads local/registry model artifacts |

## Frontend

| File | Purpose |
|---|---|
| `frontend/src/App.tsx` | LendingClub risk checker UI |
| `frontend/src/domain/types.ts` | Frontend API and form types |
| `frontend/src/domain/inputMapping.ts` | Maps form values to LendingClub API payload |
| `frontend/src/domain/validation.ts` | Form validation for LendingClub inputs |
| `frontend/src/domain/featureCatalog.ts` | Optional LendingClub input catalog |
| `frontend/src/domain/modelComparisonData.ts` | Fallback model comparison metrics |
| `frontend/src/api/client.ts` | API client |

## Tests

| Path | Purpose |
|---|---|
| `tests/test_features/test_lendingclub.py` | LendingClub parser tests |
| `tests/test_features/test_pipeline.py` | Active full pipeline smoke test |
| `tests/test_serving/` | LendingClub API contract, prediction, drift, and model comparison tests |
| `tests/test_models/` | Model wrapper tests |
| `tests/test_monitoring/` | Drift alert tests |
