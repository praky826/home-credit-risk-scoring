# SERVING_SPEC.md
## Home Credit Default Risk — Serving Layer Specification

FastAPI contract, Docker build spec, and drift monitoring endpoint.

---

## 1. FastAPI Service Overview

| Property | Value |
|---|---|
| Framework | FastAPI 0.110+ |
| Python | 3.11 |
| Server | Uvicorn |
| Base path | `/` |
| Port | `8080` |
| Auth | None (internal service; add API key if exposed) |

### Startup Sequence
1. FastAPI app created in `src/serving/app.py`
2. `@app.on_event("startup")` calls `model_loader.load_model()`
3. `model_loader.load_model()` reads `configs/serving.yaml`, loads artifact, sets module singleton
4. Loads `data/processed/feature_names.json` → stored as module-level list
5. Loads `data/processed/feature medians` (from `data/drift/reference_stats.json`) → used for missing value imputation at inference time
6. Initializes in-memory rolling window (deque, maxlen=1000) for drift monitoring
7. Registers router from `src/serving/router.py`

---

## 2. Endpoints

### `GET /health`

**Purpose:** Liveness + readiness check for container orchestration.

**Request:** No body, no params.

**Response: HTTP 200 (always)**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "lgbm_run_abc123"
}
```
If model failed to load:
```json
{
  "status": "degraded",
  "model_loaded": false,
  "model_version": ""
}
```
**Never returns 5xx.** Health endpoint must always respond.

---

### `POST /predict`

**Purpose:** Score a single loan application.

**Request Body (JSON):**

All fields from `feature_names.json` are accepted. Minimum required fields:

| Field | Type | Required | Notes |
|---|---|---|---|
| `AMT_CREDIT` | float | ✓ | Loan amount |
| `AMT_INCOME_TOTAL` | float | ✓ | Annual income |
| `DAYS_BIRTH` | int | ✓ | Negative integer (days before application) |
| `DAYS_EMPLOYED` | int | ✓ | Negative integer or 365243 for unemployed |
| `EXT_SOURCE_2` | float | ✗ | Filled with median if missing |
| All other features | float/int | ✗ | Filled with training median if missing |
| `SK_ID_CURR` | int | ✗ | For logging only; not passed to model |

Full example request:
```json
{
  "AMT_CREDIT": 450000.0,
  "AMT_INCOME_TOTAL": 135000.0,
  "AMT_ANNUITY": 22500.0,
  "DAYS_BIRTH": -12000,
  "DAYS_EMPLOYED": -2500,
  "EXT_SOURCE_1": 0.512,
  "EXT_SOURCE_2": 0.623,
  "EXT_SOURCE_3": 0.491,
  "CODE_GENDER": 1,
  "FLAG_OWN_CAR": 0,
  "SK_ID_CURR": 100002
}
```

**Response: HTTP 200**
```json
{
  "default_probability": 0.0723,
  "model_version": "lgbm_run_abc123",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```
- `default_probability`: float in [0.0, 1.0]. Higher = higher default risk.
- `model_version`: string identifying the loaded model artifact.
- `request_id`: UUID4 generated per request for logging correlation.

**Error Responses:**

| Code | Condition | Body |
|---|---|---|
| 422 | Pydantic validation failure (wrong type, missing required field) | FastAPI default validation error |
| 503 | Model not loaded | `{"detail": "Model not available"}` |

**Internal behavior (order matters):**
1. Parse request via `PredictRequest` Pydantic model.
2. Build feature dict from request fields.
3. Compute derived features: `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_TERM`, `DAYS_EMPLOYED_PERCENT`, `EMPLOYED_ANOM`, `EXT_SOURCE_MEAN`, `EXT_SOURCE_MIN`, `EXT_SOURCE_PRODUCT`.
4. Construct DataFrame with columns = `feature_names.json` order.
5. Fill NaN columns with medians from `reference_stats.json`.
6. Call `model.predict_proba(df)` → get scalar probability.
7. Append `{features_dict, probability}` to rolling window deque.
8. Return `PredictResponse`.

---

### `GET /drift`

**Purpose:** Monitor feature distribution shift between training and recent traffic.

**Request:** No body. Optional query param:
- `?min_records=50` (default: 50) — minimum records in window before computing drift

**Response: HTTP 200**
```json
{
  "psi_values": {
    "AMT_CREDIT": 0.032,
    "AMT_INCOME_TOTAL": 0.018,
    "EXT_SOURCE_2": 0.041,
    "DAYS_EMPLOYED": 0.087,
    "...": "..."
  },
  "n_records_evaluated": 1000,
  "alert": false,
  "threshold": 0.2
}
```
- `psi_values`: PSI score for each numeric feature with at least 50 non-null values in window.
- `n_records_evaluated`: number of records in rolling window.
- `alert`: `true` if **any** feature has PSI ≥ `threshold`.
- `threshold`: from `configs/serving.yaml` → `drift_threshold` (default: 0.2).

**Error Responses:**

| Code | Condition | Body |
|---|---|---|
| 503 | Fewer than `min_records` in window | `{"detail": "Insufficient data: N records in window, need M"}` |

---

## 3. Request/Response Pydantic Models

```python
# src/serving/schemas.py

from pydantic import BaseModel, Field
from typing import Optional, Dict
import uuid

class PredictRequest(BaseModel):
    # Required
    AMT_CREDIT: float
    AMT_INCOME_TOTAL: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int

    # Common optional fields
    SK_ID_CURR: Optional[int] = None
    AMT_ANNUITY: Optional[float] = None
    AMT_GOODS_PRICE: Optional[float] = None
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None
    CODE_GENDER: Optional[int] = None      # 1=M, 0=F, -1=unknown
    FLAG_OWN_CAR: Optional[int] = None     # 1/0
    FLAG_OWN_REALTY: Optional[int] = None  # 1/0
    CNT_CHILDREN: Optional[int] = None
    NAME_EDUCATION_TYPE: Optional[int] = None   # label-encoded int
    NAME_INCOME_TYPE: Optional[int] = None
    NAME_FAMILY_STATUS: Optional[int] = None
    NAME_HOUSING_TYPE: Optional[int] = None
    OCCUPATION_TYPE: Optional[int] = None
    ORGANIZATION_TYPE: Optional[int] = None
    REGION_RATING_CLIENT: Optional[int] = None
    # ... all remaining features from feature_names.json as Optional[float | int] = None

    class Config:
        extra = "allow"   # Accept unknown fields (ignored, not passed to model)


class PredictResponse(BaseModel):
    default_probability: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class HealthResponse(BaseModel):
    status: str           # "ok" | "degraded"
    model_loaded: bool
    model_version: str


class DriftResponse(BaseModel):
    psi_values: Dict[str, float]
    n_records_evaluated: int
    alert: bool
    threshold: float
```

---

## 4. `configs/serving.yaml`

```yaml
# Model artifact location
model_type: lgbm                         # xgboost | lgbm | mlp | entity_embedding | tab_transformer
artifact_path: mlflow_artifacts/lgbm/model.pkl   # or s3://home-credit-mlops/models/{run_id}/model.pkl
model_version: lgbm_run_abc123

# Feature reference
feature_names_path: data/processed/feature_names.json
reference_stats_path: data/drift/reference_stats.json

# Server
host: 0.0.0.0
port: 8080
workers: 1                               # single worker for stateful rolling window

# Drift monitoring
drift_threshold: 0.2
rolling_window_size: 1000
min_records_for_drift: 50
```

---

## 5. Docker Build Specification

### `docker/Dockerfile.serving`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \                     # required by LightGBM
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps (serving only — no training deps)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn[standard]==0.29.0 \
    pydantic==2.7.0 \
    lightgbm==4.3.0 \
    xgboost==2.0.3 \
    torch==2.3.0+cpu \             # CPU-only torch for serving (smaller image)
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.4.2 \
    joblib==1.4.0 \
    boto3==1.34.0 \
    pyyaml==6.0.1

# Copy application code
COPY src/ src/
COPY configs/ configs/
COPY data/processed/feature_names.json data/processed/
COPY data/drift/reference_stats.json data/drift/

# Copy model artifact (or mount via volume / download from S3 at startup)
COPY mlflow_artifacts/ mlflow_artifacts/

# Expose port
EXPOSE 8080

# Environment variables (overridden at runtime)
ENV MODEL_TYPE=lgbm
ENV ARTIFACT_PATH=mlflow_artifacts/lgbm/model.pkl
ENV MODEL_VERSION=unknown

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
```

### `docker/docker-compose.yml`

```yaml
version: "3.9"

services:
  serving:
    build:
      context: ..
      dockerfile: docker/Dockerfile.serving
    ports:
      - "8080:8080"
    environment:
      - MODEL_TYPE=${MODEL_TYPE:-lgbm}
      - ARTIFACT_PATH=${ARTIFACT_PATH:-mlflow_artifacts/lgbm/model.pkl}
      - MODEL_VERSION=${MODEL_VERSION:-local}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
    volumes:
      - ../data/processed:/app/data/processed:ro
      - ../data/drift:/app/data/drift:ro
      - ../mlflow_artifacts:/app/mlflow_artifacts:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  mlflow-ui:
    image: python:3.11-slim
    working_dir: /mlruns
    command: >
      sh -c "pip install mlflow -q &&
             mlflow ui --host 0.0.0.0 --port 5000"
    ports:
      - "5000:5000"
    volumes:
      - ../mlruns:/mlruns
```

---

## 6. Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `MODEL_TYPE` | No | `lgbm` | Model class to load |
| `ARTIFACT_PATH` | No | `mlflow_artifacts/lgbm/model.pkl` | Local path or `s3://` URI |
| `MODEL_VERSION` | No | `unknown` | Displayed in `/health` and `/predict` responses |
| `AWS_ACCESS_KEY_ID` | If using S3 | — | AWS credential |
| `AWS_SECRET_ACCESS_KEY` | If using S3 | — | AWS credential |
| `AWS_DEFAULT_REGION` | If using S3 | `us-east-1` | AWS region |
| `DRIFT_THRESHOLD` | No | `0.2` | Overrides serving.yaml value |

---

## 7. Monitoring Endpoint Detail

### PSI Computation at `/drift`

```python
# Pseudo-code for the /drift endpoint handler

async def drift() -> DriftResponse:
    window_records = list(rolling_window)   # deque of {feature: value} dicts
    
    if len(window_records) < min_records:
        raise HTTPException(503, f"Insufficient data: {len(window_records)} records")
    
    # Build DataFrame from rolling window
    window_df = pd.DataFrame(window_records)
    
    # Load reference stats
    ref_stats = json.load(open(reference_stats_path))
    
    # Compute PSI for each feature
    psi_values = compute_all_psi(ref_stats, window_df, bins=10)
    
    # Check threshold
    alert = any(v >= drift_threshold for v in psi_values.values())
    
    if alert:
        alert_module.check_and_alert(psi_values, drift_threshold, sns_arn=None)
    
    return DriftResponse(
        psi_values=psi_values,
        n_records_evaluated=len(window_records),
        alert=alert,
        threshold=drift_threshold
    )
```

### Reference Stats Format (`data/drift/reference_stats.json`)
```json
{
  "AMT_CREDIT": {
    "mean": 538638.0,
    "std": 369446.0,
    "percentiles": [45000.0, 112500.0, 270000.0, 450000.0, 675000.0, 900000.0, 1125000.0, 1350000.0, 1575000.0, 4050000.0]
  },
  "EXT_SOURCE_2": {
    "mean": 0.5143,
    "std": 0.1939,
    "percentiles": [0.054, 0.268, 0.406, 0.514, 0.606, 0.687, 0.757, 0.821, 0.879, 0.999]
  },
  "...": "..."
}
```
`percentiles` array = 10 decile edges (used as bin boundaries for PSI calculation).
Generated during Phase 2 by `src/features/pipeline.py` from the training split.
