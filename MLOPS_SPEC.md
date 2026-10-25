# LendingClub Default Risk - MLOps Specification

## Tracking

MLflow experiments:

| Model family | Experiment |
|---|---|
| XGBoost, LightGBM | `lendingclub_trees` |
| MLP, Entity Embedding, TabTransformer | `lendingclub_nn` |

Registry model name:

```text
lendingclub_default_risk
```

The local serving config currently points directly to the first-pass XGBoost artifact. Registry promotion can be re-enabled after tuned models are selected.

## Artifacts

Training writes model artifacts under:

```text
mlflow_artifacts/
```

Feature engineering writes:

```text
data/processed/train_features.parquet
data/processed/test_features.parquet
data/processed/feature_names.json
data/processed/category_mappings.json
data/drift/reference_stats.json
```

## Drift Monitoring

Population Stability Index compares recent scored requests against LendingClub reference statistics generated from the processed training feature matrix.

Current drift config:

```yaml
reference_stats_path: data/drift/reference_stats.json
drift_threshold: 0.2
rolling_window_size: 100
```

## Calibration

The serving layer uses `scale_pos_weight_unbias` with a first-pass LendingClub class-weight estimate:

```yaml
scale_pos_weight: 4.29
```

This should be recalibrated after final tuned training.

For the current Entity Embedding holdout, the calibrated probability bands
produce approximately 21% Low Risk, 44% Needs Review, and 36% High Attention
using thresholds of 0.15 and 0.30. These are operational bands, not approval
decisions.
