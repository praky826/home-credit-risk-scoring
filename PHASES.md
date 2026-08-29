# PHASES.md
## Home Credit Default Risk — Build Order & Phase Acceptance Tests

Each phase produces a working, testable artifact before the next begins.
Never start Phase N+1 until Phase N's acceptance test passes.

---

## Phase 1 — Data Ingestion & Exploration
**Goal:** Understand the schema completely. No unknowns going into feature engineering.

### What you build
- `notebooks/01_data_exploration.ipynb` — full EDA
- `data/DATA_DICTIONARY.md` reference (populated, not generated) — see `DATA_DICTIONARY.md`
- `scripts/download_data.sh` — idempotent Kaggle download script

### Files created/modified
```
scripts/download_data.sh          [new]
notebooks/01_data_exploration.ipynb  [new]
data/raw/*                        [downloaded, not committed]
```

### Done looks like
- All 8 CSVs present in `data/raw/`
- Notebook runs cell-by-cell without errors
- Every table has: row count confirmed, dtypes audited, missingness % per column logged, SK_ID_CURR join cardinality verified
- Target rate confirmed: ~8.1% positive class
- No table has unresolved "what is this column?" questions

### Acceptance Test
```bash
python - <<'EOF'
import pandas as pd, os
tables = [
    "application_train.csv","application_test.csv","bureau.csv",
    "bureau_balance.csv","previous_application.csv","POS_CASH_balance.csv",
    "installments_payments.csv","credit_card_balance.csv"
]
for t in tables:
    path = f"data/raw/{t}"
    assert os.path.exists(path), f"Missing: {path}"
    df = pd.read_csv(path, nrows=1)
    assert "SK_ID_CURR" in df.columns, f"No SK_ID_CURR in {t}"
train = pd.read_csv("data/raw/application_train.csv")
assert train["TARGET"].mean() < 0.12, "Target rate unexpected"
print("Phase 1 PASS")
EOF
```

---

## Phase 2 — Feature Engineering Pipeline
**Goal:** Transform all raw tables into a single clean training matrix.

### What you build
- `src/features/application.py`
- `src/features/bureau.py`
- `src/features/previous_app.py`
- `src/features/pos_cash.py`
- `src/features/installments.py`
- `src/features/credit_card.py`
- `src/features/utils.py`
- `src/features/pipeline.py`
- `configs/features.yaml`
- `tests/test_features/` (unit + integration)
- `scripts/build_features.sh`

### Files created/modified
```
src/features/*.py                 [new — all modules]
configs/features.yaml             [new]
tests/test_features/*.py          [new]
scripts/build_features.sh         [new]
data/processed/train_features.parquet    [generated artifact]
data/processed/test_features.parquet     [generated artifact]
data/processed/feature_names.json        [generated artifact]
data/processed/cat_feature_indices.json  [generated artifact]
data/processed/cat_cardinalities.json    [generated artifact]
```

### Done looks like
- `python -m src.features.pipeline` runs without errors in < 15 minutes
- Output shape: `train_features.parquet` has 307,511 rows, ≥ 200 feature columns + SK_ID_CURR + TARGET
- No column is 100% null
- No data leakage: TARGET not in test features, SK_ID_CURR not in feature_names.json
- All unit tests pass

### Acceptance Test
```bash
pytest tests/test_features/ -v

python - <<'EOF'
import pandas as pd, json
train = pd.read_parquet("data/processed/train_features.parquet")
test  = pd.read_parquet("data/processed/test_features.parquet")
names = json.load(open("data/processed/feature_names.json"))

assert train.shape[0] == 307_511, f"Train row count wrong: {train.shape[0]}"
assert "TARGET" in train.columns
assert "TARGET" not in test.columns
assert "SK_ID_CURR" not in names, "SK_ID_CURR leaked into feature list"
assert train[names].isnull().all().sum() == 0, "Column(s) are 100% null"
assert len(names) >= 200, f"Too few features: {len(names)}"
print(f"Phase 2 PASS — {len(names)} features, {train.shape[0]} train rows")
EOF
```

---

## Phase 3 — Tree Models (XGBoost + LightGBM)
**Goal:** Establish strong baselines; get a leaderboard-competitive AUC.

### What you build
- `src/models/base.py`
- `src/models/xgboost_model.py`
- `src/models/lgbm_model.py`
- `src/models/train.py`
- `configs/xgboost.yaml`
- `configs/lgbm.yaml`
- `tests/test_models/test_xgboost_model.py`

### Files created/modified
```
src/models/base.py                [new]
src/models/xgboost_model.py       [new]
src/models/lgbm_model.py          [new]
src/models/train.py               [new]
configs/xgboost.yaml              [new]
configs/lgbm.yaml                 [new]
tests/test_models/test_xgboost_model.py  [new]
mlruns/                           [populated with 2 experiment runs]
```

### Done looks like
- `python -m src.models.train --model xgboost` completes and logs to MLflow
- `python -m src.models.train --model lgbm` completes and logs to MLflow
- Optuna tuning (50 trials each) completes without crashing
- Validation ROC-AUC: XGBoost ≥ 0.775, LightGBM ≥ 0.780
- MLflow UI shows both runs with params, metrics, and model artifact

### Acceptance Test
```bash
pytest tests/test_models/test_xgboost_model.py -v

python - <<'EOF'
import mlflow
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name("home_credit_trees")
runs = client.search_runs(exp.experiment_id, order_by=["metrics.val_roc_auc DESC"])
assert len(runs) >= 2, "Need at least 2 logged runs"
best_auc = float(runs[0].data.metrics["val_roc_auc"])
assert best_auc >= 0.775, f"Best AUC too low: {best_auc}"
print(f"Phase 3 PASS — best val AUC: {best_auc:.4f}")
EOF
```

---

## Phase 4 — Neural Network Models
**Goal:** Implement and train MLP, Entity Embedding Network, and TabTransformer.

### What you build
- `src/models/mlp.py`
- `src/models/entity_embedding.py`
- `src/models/tab_transformer.py`
- `configs/mlp.yaml`
- `configs/entity_embedding.yaml`
- `configs/tab_transformer.yaml`
- `tests/test_models/test_mlp.py`
- `tests/test_models/test_entity_embedding.py`

### Files created/modified
```
src/models/mlp.py                 [new]
src/models/entity_embedding.py    [new]
src/models/tab_transformer.py     [new]
configs/mlp.yaml                  [new]
configs/entity_embedding.yaml     [new]
configs/tab_transformer.yaml      [new]
tests/test_models/test_mlp.py     [new]
tests/test_models/test_entity_embedding.py  [new]
mlruns/                           [3 more runs added]
```

### Done looks like
- All three models train for the configured number of epochs without NaN loss
- `python -m src.models.train --model mlp` logs to MLflow with val AUC
- `python -m src.models.train --model entity_embedding` same
- `python -m src.models.train --model tab_transformer` same
- Validation ROC-AUC: MLP ≥ 0.760, EEN ≥ 0.770
- Both PyTorch model tests pass (forward pass shape, loss decreases)

### Acceptance Test
```bash
pytest tests/test_models/test_mlp.py tests/test_models/test_entity_embedding.py -v

python - <<'EOF'
import mlflow
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name("home_credit_nn")
runs = client.search_runs(exp.experiment_id, order_by=["metrics.val_roc_auc DESC"])
assert len(runs) >= 3, "Need at least 3 NN runs"
for r in runs:
    auc = float(r.data.metrics["val_roc_auc"])
    model = r.data.params["model_type"]
    print(f"  {model}: {auc:.4f}")
    assert not (auc != auc), f"NaN AUC for {model}"  # nan check
print("Phase 4 PASS")
EOF
```

---

## Phase 5 — Comparison Analysis
**Goal:** Rigorous, reproducible comparison of all 5 models on the holdout test set.

### What you build
- `src/evaluation/metrics.py`
- `src/evaluation/plots.py`
- `src/evaluation/comparison.py`
- `notebooks/03_model_comparison.ipynb`
- `notebooks/04_error_analysis.ipynb`

### Files created/modified
```
src/evaluation/metrics.py         [new]
src/evaluation/plots.py           [new]
src/evaluation/comparison.py      [new]
notebooks/03_model_comparison.ipynb  [new]
notebooks/04_error_analysis.ipynb    [new]
```

### Done looks like
- `comparison.py` produces a markdown table: model × {AUC, KS, Brier, calibration slope}
- ROC curves for all 5 models plotted on one axis and saved as PNG
- Wilcoxon signed-rank test run between best tree and best NN; p-value reported
- Feature importance plot saved for both tree models
- Calibration curves plotted for all 5 models
- Clear narrative answer to: "Do NNs beat trees on this dataset?"

### Acceptance Test
```bash
python - <<'EOF'
from src.evaluation.comparison import run_comparison
table = run_comparison()   # returns dict keyed by model name
assert len(table) == 5, "Need results for all 5 models"
for name, metrics in table.items():
    assert "roc_auc" in metrics
    assert "ks_stat" in metrics
    assert "brier_score" in metrics
    assert 0.5 < metrics["roc_auc"] < 1.0, f"Implausible AUC for {name}"
print("Phase 5 PASS")
print({k: round(v["roc_auc"],4) for k,v in table.items()})
EOF
```

---

## Phase 6 — Serving Layer
**Goal:** Productionised FastAPI service that loads the best model and serves predictions.

### What you build
- `src/serving/app.py`
- `src/serving/router.py`
- `src/serving/schemas.py`
- `src/serving/model_loader.py`
- `src/serving/drift.py`
- `configs/serving.yaml`
- `docker/Dockerfile.serving`
- `docker/docker-compose.yml`
- `tests/test_serving/test_router.py`
- `tests/test_serving/test_schemas.py`

### Files created/modified
```
src/serving/*.py                  [new — all modules]
configs/serving.yaml              [new]
docker/Dockerfile.serving         [new]
docker/docker-compose.yml         [new]
tests/test_serving/*.py           [new]
```

### Done looks like
- `uvicorn src.serving.app:app` starts without error
- `/health` returns 200
- `/predict` returns a JSON response with `default_probability` for a valid payload
- `/predict` returns 422 for a malformed payload
- `/drift` returns PSI values for the last N scored records
- Docker image builds: `docker build -f docker/Dockerfile.serving . -t home-credit-serving`
- Container starts and `/health` responds on port 8080

### Acceptance Test
```bash
pytest tests/test_serving/ -v

# Docker smoke test
docker build -f docker/Dockerfile.serving . -t home-credit-serving-test -q
docker run -d -p 8080:8080 --name hc-test home-credit-serving-test
sleep 5
curl -sf http://localhost:8080/health | python -c "import sys,json; d=json.load(sys.stdin); assert d['status']=='ok'"
docker stop hc-test && docker rm hc-test
echo "Phase 6 PASS"
```

---

## Phase 7 — MLOps Layer
**Goal:** Automated training pipeline, model registry, drift monitoring, and AWS deployment.

### What you build
- `.github/workflows/ci.yml`
- `.github/workflows/train_and_register.yml`
- `src/monitoring/psi.py`
- `src/monitoring/alert.py`
- `configs/mlops.yaml`
- `scripts/promote_model.sh`
- `docker/Dockerfile.training`

### Files created/modified
```
.github/workflows/ci.yml          [new]
.github/workflows/train_and_register.yml  [new]
src/monitoring/psi.py             [new]
src/monitoring/alert.py           [new]
configs/mlops.yaml                [new]
scripts/promote_model.sh          [new]
docker/Dockerfile.training        [new]
```

### Done looks like
- CI workflow runs on every PR: lint (black --check), type check (mypy), unit tests (pytest tests/)
- Training workflow: triggered manually, runs full train → logs to MLflow → pushes best model artifact to S3 → registers in MLflow Model Registry
- PSI computed on `/drift` endpoint for each numeric feature; threshold alert fires at PSI > 0.2
- `scripts/promote_model.sh` transitions a run to "Production" stage in MLflow registry
- AWS S3 bucket exists at `s3://home-credit-mlops/models/{run_id}/model.pkl`

### Acceptance Test
```bash
# CI workflow test (local act runner or GitHub push)
# Verify PSI module works standalone
python - <<'EOF'
import numpy as np
from src.monitoring.psi import compute_psi
ref = np.random.normal(0, 1, 10000)
same = np.random.normal(0, 1, 1000)   # same dist → low PSI
shifted = np.random.normal(2, 1, 1000) # shifted dist → high PSI
psi_same = compute_psi(ref, same, bins=10)
psi_shifted = compute_psi(ref, shifted, bins=10)
assert psi_same < 0.1, f"PSI on same dist too high: {psi_same:.3f}"
assert psi_shifted > 0.2, f"PSI on shifted dist too low: {psi_shifted:.3f}"
print(f"Phase 7 PASS — PSI same={psi_same:.3f}, shifted={psi_shifted:.3f}")
EOF
```

---

## Phase Summary Table

| Phase | Key Artifact | Acceptance Gate |
|---|---|---|
| 1 | 8 CSVs in `data/raw/` | Row counts + SK_ID_CURR present |
| 2 | `train_features.parquet` (307,511 × 200+) | pytest + shape/null checks |
| 3 | 2 MLflow runs, val AUC ≥ 0.775 | MLflow API query |
| 4 | 3 more MLflow runs, no NaN loss | pytest + MLflow API query |
| 5 | Comparison table + plots | `run_comparison()` returns 5 models |
| 6 | Docker container serving `/predict` | curl `/health` on running container |
| 7 | CI green, PSI module works | PSI unit test + GitHub Actions green |
