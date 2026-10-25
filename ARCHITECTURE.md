# LendingClub Default Risk - Architecture

## Pipeline

```text
LendingClub accepted loans
-> src/features/lendingclub.py
-> data/processed/train_features.parquet
-> src/models/train.py
-> MLflow/model artifact
-> src/serving/app.py
-> frontend React risk checker
```

## Main Components

| Layer | Location | Purpose |
|---|---|---|
| Feature engineering | `src/features/lendingclub.py` | Build LendingClub model matrix and metadata |
| Model training | `src/models/` | Train XGBoost, LightGBM, and neural tabular models |
| Evaluation | `src/evaluation/` | ROC-AUC, KS, Brier score, calibration slope, plots |
| Serving | `src/serving/` | FastAPI prediction, feature alignment, calibration, drift window |
| Monitoring | `src/monitoring/` | PSI drift computation and alerting |
| Frontend | `frontend/src/` | LendingClub loan risk checker and model comparison dashboard |

## Active Data Contract

The downstream contract is intentionally stable:

```text
train_features.parquet
test_features.parquet
feature_names.json
cat_feature_indices.json
cat_cardinalities.json
category_mappings.json
reference_stats.json
```

This allows dataset-specific feature logic to change without rewriting model wrappers, evaluation utilities, or the serving alignment path.
