import os
import json
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, create_model

# 1. Base Responses
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    model_name: Optional[str] = "XGBoost Risk Classifier (Production Champion)"
    model_type: Optional[str] = "xgboost"
    model_stage: Optional[str] = "Production"

class PredictResponse(BaseModel):
    default_probability: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    model_name: Optional[str] = "XGBoost Risk Classifier (Production Champion)"
    request_id: str

class DriftResponse(BaseModel):
    psi_values: Dict[str, float]
    n_records_evaluated: int
    alert: bool
    threshold: float

class ModelBenchmarkItem(BaseModel):
    model_id: str
    name: str
    architecture: str
    roc_auc: float
    ks_stat: float
    brier_score: float
    calibration_slope: float
    inference_latency_ms: float
    status: str
    is_champion: bool = False

class WilcoxonResultItem(BaseModel):
    comparison: str
    statistic: float
    p_value: float
    significant: bool
    interpretation: str

class SelectionRationaleItem(BaseModel):
    title: str
    summary: str
    details: str

class ModelComparisonResponse(BaseModel):
    champion_model_id: str
    champion_model_name: str
    models: list[ModelBenchmarkItem]
    selection_rationales: list[SelectionRationaleItem]
    significance_tests: list[WilcoxonResultItem]
    evaluation_dataset: Dict[str, Any]

# 2. Dynamic PredictRequest Creation
default_features = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_ANNUITY",
    "AMT_GOODS_PRICE", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "CODE_GENDER",
    "NAME_FAMILY_STATUS", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "FLAG_DOCUMENT_3",
    "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO",
    "CREDIT_TERM", "DAYS_EMPLOYED_PERCENT", "EMPLOYED_ANOM"
]

feature_names_path = "data/processed/feature_names.json"
feature_names = []
if os.path.exists(feature_names_path):
    try:
        with open(feature_names_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
    except Exception:
        feature_names = default_features
else:
    feature_names = default_features

# Create fields dictionary for Pydantic model
fields = {
    "SK_ID_CURR": (Optional[int], Field(default=None, description="Client ID for logging")),
    "AMT_CREDIT": (float, Field(..., description="Credit amount of the loan")),
    "AMT_INCOME_TOTAL": (float, Field(..., description="Income of the client")),
    "DAYS_BIRTH": (int, Field(..., description="Client's age in days (negative)")),
    "DAYS_EMPLOYED": (int, Field(..., description="How many days before application the person started current job")),
}

# Add all other features from feature_names.json as optional floats
for feat in feature_names:
    if feat not in fields:
        fields[feat] = (Optional[float], Field(default=None))

# Create config class to allow extra parameters
class CustomConfig:
    extra = "allow"

PredictRequest = create_model("PredictRequest", __config__=CustomConfig, **fields)
print(f"Dynamically registered PredictRequest schema with {len(fields)} fields.")
