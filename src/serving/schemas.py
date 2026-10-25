import os
import json
from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, create_model

# 1. Base Responses
class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_version: str
    model_name: Optional[str] = "LendingClub XGBoost Default Risk Classifier"
    model_type: Optional[str] = "xgboost"
    model_stage: Optional[str] = "Production"

class PredictionWarning(BaseModel):
    code: str
    field: Optional[str] = None
    message: str

class FeatureLookupMetadata(BaseModel):
    customer_id: Optional[int] = None
    found: bool = False
    source: Optional[str] = None

class SolvencyDiagnostics(BaseModel):
    repayment_coverage_ratio: float
    disposable_monthly_cashflow: float
    debt_service_burden_pct: float
    salary_multiplier: float
    amortized_installment: float
    policy_knockout: bool = False
    policy_reasons: list[str] = []

class PredictResponse(BaseModel):
    default_probability: float = Field(..., ge=0.0, le=1.0)
    raw_model_probability: Optional[float] = None
    calibration_method: Optional[str] = None
    decision_label: Optional[str] = None
    model_version: str
    model_name: Optional[str] = "LendingClub XGBoost Default Risk Classifier"
    request_id: str
    feature_lookup: Optional[FeatureLookupMetadata] = None
    solvency: Optional[SolvencyDiagnostics] = None
    warnings: list[PredictionWarning] = []

class DriftResponse(BaseModel):
    psi_values: Dict[str, float]
    n_records_evaluated: int
    alert: bool
    threshold: float

class ModelBenchmarkItem(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

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
    model_config = ConfigDict(protected_namespaces=())

    champion_model_id: str
    champion_model_name: str
    models: list[ModelBenchmarkItem]
    selection_rationales: list[SelectionRationaleItem]
    significance_tests: list[WilcoxonResultItem]
    evaluation_dataset: Dict[str, Any]

# 2. Dynamic PredictRequest Creation
default_features = [
    "loan_amnt", "term", "installment", "annual_inc", "dti",
    "fico_range_low", "fico_range_high", "fico_avg", "emp_length",
    "home_ownership", "verification_status", "purpose", "addr_state",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "initial_list_status", "application_type",
    "mort_acc", "pub_rec_bankruptcies", "loan_income_ratio",
    "installment_income_ratio", "revol_balance_income_ratio",
    "credit_history_years", "issue_year", "issue_month"
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
    "loan_id": (Optional[int], Field(default=None, description="LendingClub loan ID for lookup/logging")),
    "loan_amnt": (float, Field(..., description="Requested personal loan amount")),
    "annual_inc": (float, Field(..., description="Self-reported annual income")),
    "dti": (float, Field(..., description="Debt-to-income ratio")),
    "term": (Optional[float], Field(default=36.0, description="Loan term in months (36 or 60)")),
    "installment": (Optional[float], Field(default=None, description="Scheduled monthly installment")),
    "fico_range_low": (Optional[float], Field(default=None, description="Lower bound of borrower FICO range")),
    "fico_range_high": (Optional[float], Field(default=None, description="Upper bound of borrower FICO range")),
}

# Add all other features from feature_names.json as optional floats
for feat in feature_names:
    if feat not in fields:
        fields[feat] = (Optional[float], Field(default=None))

request_model_config = ConfigDict(extra="allow", protected_namespaces=())

PredictRequest = create_model("PredictRequest", __config__=request_model_config, **fields)
print(f"Dynamically registered PredictRequest schema with {len(fields)} fields.")
