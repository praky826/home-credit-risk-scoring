# MLOPS_SPEC.md
## Home Credit Default Risk — MLOps Layer Specification

GitHub Actions workflows, MLflow tracking schema, AWS infrastructure, and drift monitoring logic.

---

## 1. GitHub Actions Workflows

### Workflow 1: `ci.yml` — Pull Request Gate

**File:** `.github/workflows/ci.yml`

**Trigger:**
```yaml
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]
```

**Runner:** `ubuntu-22.04`

**Steps (in order):**

| Step | Command | Fail condition |
|---|---|---|
| Checkout | `actions/checkout@v4` | — |
| Setup Python | `actions/setup-python@v5`, python-version: `3.11` | — |
| Cache pip | `actions/cache@v4` on `~/.cache/pip` keyed by `requirements*.txt` hash | — |
| Install deps | `pip install -r requirements-dev.txt` | pip error |
| Lint (black) | `black --check src/ tests/` | Any formatting diff |
| Sort imports | `isort --check-only src/ tests/` | Any import order diff |
| Type check | `mypy src/ --ignore-missing-imports` | Any type error |
| Unit tests | `pytest tests/ -v --tb=short -x` | Any test failure |
| Coverage | `pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=70` | Coverage < 70% |

**What does NOT run in CI:**
- Feature engineering pipeline (too slow, requires data)
- Model training (no GPU, no data)
- Docker build (handled in training workflow)

**Full workflow YAML:**
```yaml
name: CI

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements*.txt') }}

      - name: Install dependencies
        run: pip install -r requirements-dev.txt

      - name: Lint
        run: black --check src/ tests/

      - name: Sort imports
        run: isort --check-only src/ tests/

      - name: Type check
        run: mypy src/ --ignore-missing-imports

      - name: Test
        run: pytest tests/ -v --tb=short -x --cov=src --cov-report=term-missing --cov-fail-under=70
```

---

### Workflow 2: `train_and_register.yml` — Training Pipeline

**File:** `.github/workflows/train_and_register.yml`

**Trigger:**
```yaml
on:
  workflow_dispatch:       # Manual trigger with inputs
    inputs:
      model_type:
        description: 'Model to train'
        required: true
        default: 'lgbm'
        type: choice
        options: [xgboost, lgbm, mlp, entity_embedding, tab_transformer, all]
      tune:
        description: 'Run hyperparameter tuning'
        required: false
        default: 'false'
        type: boolean
  schedule:
    - cron: '0 2 * * 1'   # Every Monday at 2am UTC (weekly retraining)
```

**Runner:** `ubuntu-22.04` (or self-hosted GPU runner if available)

**Required Secrets:**
```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
KAGGLE_USERNAME
KAGGLE_KEY
MLFLOW_TRACKING_URI        # e.g. http://mlflow-server:5000 or s3://... for remote
```

**Steps (in order):**

| Step | What it does |
|---|---|
| Checkout | Clone repo |
| Setup Python 3.11 | — |
| Install deps | `pip install -r requirements.txt` |
| Download data | `bash scripts/download_data.sh` (uses KAGGLE_* secrets) |
| Build features | `python -m src.features.pipeline` |
| Train model(s) | `python -m src.models.train --model $MODEL_TYPE [--tune]` (loop if "all") |
| Run comparison | `python -m src.evaluation.comparison` (if model_type == "all") |
| Push to S3 | Upload best model artifact to `s3://home-credit-mlops/models/{run_id}/` |
| Register in MLflow | Transition best run to "Staging" in MLflow Model Registry |
| Build Docker image | `docker build -f docker/Dockerfile.serving -t home-credit-serving:{run_id} .` |
| Push Docker image | Push to ECR `home-credit-serving:{run_id}` and `:latest` |

**Pass/Fail:**
- Workflow **passes** if: training completes, val_roc_auc > 0.75, artifact uploaded to S3.
- Workflow **fails** if: training crashes, AUC below threshold, S3 upload fails, Docker build fails.
- On failure: GitHub sends email to repo owner (configured in repo notification settings).

---

## 2. MLflow Tracking Schema

**Tracking URI:** Set via `MLFLOW_TRACKING_URI` env var. Defaults to `mlruns/` (local file-based).

### Experiment Names
| Models | Experiment Name |
|---|---|
| XGBoost, LightGBM | `home_credit_trees` |
| MLP, EEN, TabTransformer | `home_credit_nn` |

### Per-Run Logged Content

#### Tags (set via `mlflow.set_tag`)
```
model_type:         str   # xgboost | lgbm | mlp | entity_embedding | tab_transformer
dataset_version:    str   # git commit hash of data/processed/ generation
git_commit:         str   # current HEAD commit hash
tuned:              str   # "true" | "false"
```

#### Params (set via `mlflow.log_params`)
All hyperparameters from the config yaml, flattened.

**For XGBoost:**
```
n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
min_child_weight, reg_alpha, reg_lambda, scale_pos_weight,
early_stopping_rounds, tree_method
```

**For LightGBM:**
```
n_estimators, num_leaves, learning_rate, subsample, colsample_bytree,
min_child_samples, reg_alpha, reg_lambda, is_unbalance, early_stopping_rounds
```

**For MLP/EEN/TabTransformer:**
```
hidden_dims (as string, e.g. "[512, 256, 128]"),
dropout, learning_rate, weight_decay, batch_size,
epochs, patience, optimizer, scheduler,
# TabTransformer additional:
d_model, n_heads, n_layers, ff_dim
```

#### Metrics (set via `mlflow.log_metric`)
Logged at end of training (not per-epoch; for per-epoch use `step=` param):

```
val_roc_auc:         float  # primary selection metric
test_roc_auc:        float  # only computed if training workflow; not in CI
val_ks_stat:         float
val_brier_score:     float
val_log_loss:        float
best_val_epoch:      int    # NNs only: epoch where early stopping triggered
n_estimators_used:   int    # Trees only: actual rounds trained (after early stopping)
training_time_sec:   float
```

**Key naming rules:**
- All lowercase with underscores
- Prefix `val_` for validation set metrics, `test_` for test set metrics
- No spaces, no slashes

#### Artifacts (logged via `mlflow.log_artifact`)
```
model.pkl | model.pt        # serialized model
feature_names.json          # list of feature names used
configs/{model_type}.yaml   # exact config used for this run
optuna_study.pkl            # Optuna study object (trees only, if tuned=true)
val_roc_curve.png           # ROC curve plot on validation set
val_calibration.png         # Calibration curve on validation set
feature_importance.png      # Trees only
```

### Model Registry
After training, best run is registered:
```python
mlflow.register_model(
    f"runs:/{run_id}/model",
    name="home_credit_default_risk"
)
# Then transition to Staging:
client.transition_model_version_stage(
    name="home_credit_default_risk",
    version=version_number,
    stage="Staging"
)
```
Promotion from Staging → Production requires manual approval (`scripts/promote_model.sh`).

---

## 3. AWS Infrastructure Specification

### S3 Buckets

| Bucket | Purpose | Versioning |
|---|---|---|
| `home-credit-mlops` | Model artifacts, MLflow artifact store | Enabled |
| `home-credit-data` | Raw data backup (optional) | Disabled |

### Artifact Naming Convention
```
s3://home-credit-mlops/models/{run_id}/model.pkl
s3://home-credit-mlops/models/{run_id}/feature_names.json
s3://home-credit-mlops/models/{run_id}/configs/{model_type}.yaml
s3://home-credit-mlops/mlflow-artifacts/{experiment_id}/{run_id}/artifacts/
```

`run_id` = MLflow run ID (e.g. `abc123def456`). 32-char hex string.

### IAM Permissions Required

**For training pipeline (GitHub Actions runner):**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::home-credit-mlops",
        "arn:aws:s3:::home-credit-mlops/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload"
      ],
      "Resource": "*"
    }
  ]
}
```

**For serving container (EC2/ECS task role):**
```json
{
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": ["arn:aws:s3:::home-credit-mlops", "arn:aws:s3:::home-credit-mlops/*"]
    },
    {
      "Effect": "Allow",
      "Action": ["sns:Publish"],
      "Resource": "arn:aws:sns:us-east-1:ACCOUNT_ID:home-credit-drift-alerts"
    }
  ]
}
```

### ECR Repository
```
Repository name:  home-credit-serving
Image tags:       {run_id} (immutable), latest (mutable)
Region:           us-east-1
Scan on push:     enabled
```

### SNS Topic (Drift Alerts)
```
Topic name:   home-credit-drift-alerts
Type:         Standard
Subscribers:  Email to ops team (configured manually)
Message format (published by src/monitoring/alert.py):
{
  "alert_type": "feature_drift",
  "timestamp": "2024-01-15T14:23:00Z",
  "features_above_threshold": ["DAYS_EMPLOYED", "EXT_SOURCE_2"],
  "max_psi": 0.34,
  "threshold": 0.2,
  "n_records_evaluated": 1000,
  "model_version": "lgbm_run_abc123"
}
```

---

## 4. Drift Monitoring Logic

### PSI Computation Detail

```python
def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    eps: float = 1e-6
) -> float:
    # Step 1: compute decile edges from reference
    percentiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(reference, percentiles)
    bin_edges[0] = -np.inf    # open left edge
    bin_edges[-1] = np.inf    # open right edge

    # Step 2: bin both distributions
    ref_counts = np.histogram(reference, bins=bin_edges)[0]
    cur_counts = np.histogram(current, bins=bin_edges)[0]

    # Step 3: convert to proportions
    ref_props = ref_counts / len(reference)
    cur_props = cur_counts / len(current)

    # Step 4: add epsilon to avoid log(0) or division by zero
    ref_props = ref_props + eps
    cur_props = cur_props + eps

    # Step 5: normalize (ensure sums to 1 after eps addition)
    ref_props = ref_props / ref_props.sum()
    cur_props = cur_props / cur_props.sum()

    # Step 6: compute PSI
    psi = np.sum((ref_props - cur_props) * np.log(ref_props / cur_props))
    return float(psi)
```

### PSI Interpretation
| PSI Value | Status | Action |
|---|---|---|
| < 0.1 | Stable | No action |
| 0.1 – 0.2 | Moderate shift | Log warning, increase monitoring frequency |
| ≥ 0.2 | Significant shift | Fire SNS alert, schedule retraining review |

### Drift Monitoring Scope
PSI computed for **all numeric features** in `feature_names.json`. Categorical features are excluded from PSI (PSI is defined for continuous distributions). Categorical drift is monitored separately via chi-square test if needed (not in MVP scope).

### Reference Stats Generation
```python
# Run once after Phase 2 feature engineering, before serving starts
def generate_reference_stats(
    train_features_path: str = "data/processed/train_features.parquet",
    output_path: str = "data/drift/reference_stats.json",
    bins: int = 10
) -> None:
    """
    For each numeric feature column:
    - Compute mean, std
    - Compute bin_edges = np.percentile(col.dropna(), np.linspace(0, 100, bins+1))
    Save to output_path as JSON.
    Skips TARGET and SK_ID_CURR.
    """
```
Call this at end of Phase 2 pipeline. Commit the output. This is the reference distribution; it should not change until the model is retrained from scratch.

### Alert Deduplication
`alert.py` tracks last alert timestamp in a module-level variable. Will not fire a second SNS alert within 1 hour even if `/drift` is called repeatedly with the same high-PSI window. Reset on service restart.

---

## 5. `configs/mlops.yaml`

```yaml
mlflow:
  tracking_uri: "mlruns"              # Override with MLFLOW_TRACKING_URI env var
  experiment_trees: "home_credit_trees"
  experiment_nn: "home_credit_nn"
  registry_model_name: "home_credit_default_risk"

aws:
  region: "us-east-1"
  s3_bucket: "home-credit-mlops"
  s3_model_prefix: "models"
  ecr_repo: "home-credit-serving"
  sns_drift_arn: ""                   # Set to SNS ARN when deploying to AWS

drift:
  psi_threshold: 0.2
  alert_cooldown_seconds: 3600
  bins: 10
  eps: 1.0e-6

training:
  min_acceptable_val_auc: 0.75        # Workflow fails if below this
  promote_to_staging_auto: true       # Auto-promote to Staging after training
  promote_to_production_auto: false   # Production requires manual approval
```

---

## 6. `scripts/promote_model.sh`

```bash
#!/bin/bash
# Usage: bash scripts/promote_model.sh --run-id abc123 --stage Production

set -e

RUN_ID=""
STAGE="Production"

while [[ $# -gt 0 ]]; do
  case $1 in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --stage)  STAGE="$2";  shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [ -z "$RUN_ID" ]; then
  echo "Error: --run-id required"
  exit 1
fi

python - <<EOF
import mlflow
client = mlflow.tracking.MlflowClient()
model_name = "home_credit_default_risk"

# Find the model version with this run_id
versions = client.search_model_versions(f"name='{model_name}'")
target = next((v for v in versions if v.run_id == "${RUN_ID}"), None)
if not target:
    raise ValueError(f"No model version found for run_id=${RUN_ID}")

client.transition_model_version_stage(
    name=model_name,
    version=target.version,
    stage="${STAGE}",
    archive_existing_versions=True   # archive other versions in same stage
)
print(f"Promoted {model_name} v{target.version} (run {target.run_id}) to ${STAGE}")
EOF
```
