# PROJECT_STRUCTURE.md
## Home Credit Default Risk — Complete File Tree

```
home-credit/
│
├── data/
│   ├── raw/                        # Immutable source CSVs from Kaggle
│   │   ├── application_train.csv   # Main training table (307,511 rows)
│   │   ├── application_test.csv    # Main test table (48,744 rows)
│   │   ├── bureau.csv              # Credit Bureau history per applicant
│   │   ├── bureau_balance.csv      # Monthly balance snapshots for bureau loans
│   │   ├── previous_application.csv# Prior HC applications per applicant
│   │   ├── POS_CASH_balance.csv    # Monthly POS and cash loan snapshots
│   │   ├── installments_payments.csv# Repayment history for prior HC loans
│   │   ├── credit_card_balance.csv # Monthly credit card snapshots
│   │   └── HomeCredit_columns_description.csv  # Official column glossary
│   │
│   ├── processed/                  # Outputs of feature engineering pipeline
│   │   ├── train_features.parquet  # Final training matrix (SK_ID_CURR + features + TARGET)
│   │   ├── test_features.parquet   # Final test matrix (SK_ID_CURR + features)
│   │   ├── feature_names.json      # Ordered list of feature column names
│   │   ├── cat_feature_indices.json# Indices of categorical columns (for embeddings)
│   │   └── cat_cardinalities.json  # {col_name: n_unique} for embedding dim calc
│   │
│   ├── splits/                     # Train/val/test index files (reproducible)
│   │   ├── train_idx.npy           # np array of SK_ID_CURR for train fold
│   │   ├── val_idx.npy             # np array of SK_ID_CURR for validation fold
│   │   └── test_idx.npy            # np array of SK_ID_CURR for holdout test fold
│   │
│   └── drift/                      # Reference distributions for monitoring
│       └── reference_stats.json    # Per-feature mean, std, bin edges for PSI
│
├── src/
│   ├── __init__.py
│   │
│   ├── features/                   # Feature engineering — raw → processed
│   │   ├── __init__.py
│   │   ├── pipeline.py             # Orchestrator: calls all sub-builders, merges, saves
│   │   ├── application.py          # Features from application_train/test
│   │   ├── bureau.py               # Aggregated features from bureau + bureau_balance
│   │   ├── previous_app.py         # Aggregated features from previous_application
│   │   ├── pos_cash.py             # Aggregated features from POS_CASH_balance
│   │   ├── installments.py         # Aggregated features from installments_payments
│   │   ├── credit_card.py          # Aggregated features from credit_card_balance
│   │   └── utils.py                # Shared helpers: label encoding, imputation, dtype casting
│   │
│   ├── models/                     # Model definitions, training loops, wrappers
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract BaseModel class (fit/predict_proba interface)
│   │   ├── xgboost_model.py        # XGBoost wrapper + Optuna tuning
│   │   ├── lgbm_model.py           # LightGBM wrapper + Optuna tuning
│   │   ├── mlp.py                  # PyTorch MLP definition + training loop
│   │   ├── entity_embedding.py     # PyTorch Entity Embedding Network definition + training loop
│   │   ├── tab_transformer.py      # PyTorch TabTransformer definition + training loop
│   │   └── train.py                # Entry point: loads data, calls model, logs to MLflow
│   │
│   ├── evaluation/                 # Metrics, plots, comparison reporting
│   │   ├── __init__.py
│   │   ├── metrics.py              # ROC-AUC, KS stat, Brier score, calibration
│   │   ├── plots.py                # ROC curve, PR curve, feature importance, calibration plot
│   │   └── comparison.py           # Side-by-side model comparison table + Wilcoxon test
│   │
│   ├── serving/                    # FastAPI application
│   │   ├── __init__.py
│   │   ├── app.py                  # FastAPI app instantiation, router registration
│   │   ├── router.py               # Endpoint definitions (/predict, /health, /drift)
│   │   ├── schemas.py              # Pydantic request/response models
│   │   ├── model_loader.py         # Loads MLflow artifact or local .pkl/.pt at startup
│   │   └── drift.py                # PSI computation against reference_stats.json
│   │
│   └── monitoring/                 # Drift detection and alerting logic
│       ├── __init__.py
│       ├── psi.py                  # Population Stability Index calculation
│       └── alert.py                # Threshold checking; emits structured log / SNS message
│
├── notebooks/                      # Exploration only — not imported by src/
│   ├── 01_data_exploration.ipynb   # Schema audit, missingness heatmap, target rate
│   ├── 02_feature_analysis.ipynb   # Feature distributions, correlation, IV/WoE
│   ├── 03_model_comparison.ipynb   # Final result visualization and writeup
│   └── 04_error_analysis.ipynb     # False positive / false negative deep-dives
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Shared pytest fixtures (mini dataframes, tmp paths)
│   ├── test_features/
│   │   ├── test_application.py     # Unit tests for application feature builder
│   │   ├── test_bureau.py          # Unit tests for bureau feature builder
│   │   └── test_pipeline.py        # Integration test: raw CSVs → processed parquet shape
│   ├── test_models/
│   │   ├── test_mlp.py             # Forward pass shape, loss decreases over 5 steps
│   │   ├── test_entity_embedding.py# Embedding lookup + forward pass
│   │   └── test_xgboost_model.py   # Fit on toy data, predict_proba returns [0,1]
│   └── test_serving/
│       ├── test_router.py          # FastAPI TestClient: /health, /predict, /drift
│       └── test_schemas.py         # Pydantic validation: required fields, type coercion
│
├── configs/
│   ├── features.yaml               # Feature engineering toggles (which tables to include)
│   ├── xgboost.yaml                # Default XGBoost hyperparameters + Optuna search space
│   ├── lgbm.yaml                   # Default LightGBM hyperparameters + Optuna search space
│   ├── mlp.yaml                    # MLP architecture config + training hyperparameters
│   ├── entity_embedding.yaml       # EEN architecture config + training hyperparameters
│   ├── tab_transformer.yaml        # TabTransformer config + training hyperparameters
│   ├── serving.yaml                # FastAPI host/port, model artifact path, drift threshold
│   └── mlops.yaml                  # MLflow tracking URI, S3 bucket name, SNS ARN
│
├── mlruns/                         # MLflow tracking directory (local backend)
│   └── .gitkeep
│
├── mlflow_artifacts/               # Local artifact store (models, plots, feature lists)
│   └── .gitkeep
│
├── docker/
│   ├── Dockerfile.serving          # Production image for FastAPI service
│   ├── Dockerfile.training         # Training image with full ML deps
│   └── docker-compose.yml          # Compose: serving + mlflow-ui services
│
├── .github/
│   └── workflows/
│       ├── ci.yml                  # PR gate: lint, unit tests, type check
│       └── train_and_register.yml  # Manual/scheduled: full train → MLflow → S3
│
├── scripts/
│   ├── download_data.sh            # Kaggle API download + unzip to data/raw/
│   ├── build_features.sh           # Wrapper: python -m src.features.pipeline
│   ├── train_all.sh                # Trains all 5 models sequentially, logs to MLflow
│   └── promote_model.sh            # Promotes best MLflow run to "Production" stage
│
├── requirements.txt                # Pinned runtime deps (pandas, torch, xgboost, fastapi…)
├── requirements-dev.txt            # Dev-only deps (pytest, black, mypy, optuna…)
├── pyproject.toml                  # Black + isort + mypy config
├── .env.example                    # Template for MLFLOW_TRACKING_URI, AWS_*, KAGGLE_*
├── .gitignore                      # data/raw/, data/processed/, mlruns/, __pycache__/, .env
└── README.md                       # Project overview, quickstart, experiment results table
```

### Directory Purpose Summary

| Directory | Purpose |
|---|---|
| `data/raw/` | Immutable source data — never written to by pipeline code |
| `data/processed/` | Output of `src/features/pipeline.py` — versioned by git tag |
| `data/splits/` | Deterministic index arrays — guarantees reproducible train/val/test |
| `data/drift/` | Snapshot of training distribution for serving-time drift detection |
| `src/features/` | All feature engineering; each module handles one source table |
| `src/models/` | Model classes + training orchestration; no data logic here |
| `src/evaluation/` | Pure metrics and plotting — no model training, no data loading |
| `src/serving/` | FastAPI app — imports model_loader and drift; no training code |
| `src/monitoring/` | PSI and alerting — importable by both serving and offline batch jobs |
| `notebooks/` | EDA and result analysis — never imported, never tested |
| `tests/` | Mirrors `src/` structure; fixtures in `conftest.py` |
| `configs/` | All hyperparameters and infra config — no hardcoded values in `src/` |
| `mlruns/` | MLflow local backend — excluded from git, backed up to S3 |
| `docker/` | All Dockerfiles and Compose — serving image is minimal, training image is full |
| `.github/workflows/` | CI (every PR) and training pipeline (manual trigger or schedule) |
| `scripts/` | Shell wrappers for common tasks — thin, just invoke Python modules |
