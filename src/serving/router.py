import os
import json
import uuid
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.serving.schemas import (
    HealthResponse,
    PredictResponse,
    PredictionWarning,
    FeatureLookupMetadata,
    SolvencyDiagnostics,
    DriftResponse,
    ModelComparisonResponse,
    ModelBenchmarkItem,
    WilcoxonResultItem,
    SelectionRationaleItem,
)
from src.serving.model_loader import (
    get_model,
    get_model_version,
    get_model_name,
    get_model_type,
    get_model_stage,
)
from src.serving.drift import add_request, get_window_records, get_window_size
from src.monitoring.psi import compute_all_psi
from src.serving.calibration import calibrate_probability
from src.serving.feature_store import get_feature_store
from src.serving.underwriting_rules import (
    calculate_amortized_installment,
    evaluate_solvency_and_guardrails,
    PLATFORM_MAX_LOAN,
    PLATFORM_MIN_LOAN,
)

router = APIRouter()

# Global variables cached at startup/import
FEATURE_NAMES_PATH = "data/processed/feature_names.json"
CATEGORY_MAPPINGS_PATH = "data/processed/category_mappings.json"
REF_STATS_PATH = "data/drift/reference_stats.json"
SERVING_CONFIG_PATH = "configs/serving.yaml"

_feature_names = []
_feature_medians = {}
_category_mappings = {}
_reference_stats = {}
_drift_threshold = 0.2
_serving_config = {}

# Underwriting settings
_cal_enabled = True
_cal_method = "scale_pos_weight_unbias"
_cal_weight = 4.0
_low_max = 0.15
_review_max = 0.30
_clean_history_synthetic = True
_ood_enabled = True

CLEAN_HISTORY_PATTERNS = [
    "delinq",
    "inq_",
    "pub_rec",
    "bankrupt",
]

REQUIRED_LENDINGCLUB_INPUTS = ["loan_amnt", "annual_inc", "dti"]


def _init_metadata():
    global _feature_names, _feature_medians, _category_mappings, _reference_stats, _drift_threshold
    global _serving_config, _cal_enabled, _cal_method, _cal_weight
    global _low_max, _review_max, _clean_history_synthetic, _ood_enabled
    
    # Load feature names
    if os.path.exists(FEATURE_NAMES_PATH):
        try:
            with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
                _feature_names = json.load(f)
        except Exception as e:
            print(f"Error loading feature names: {e}")
            _feature_names = []

    if os.path.exists(CATEGORY_MAPPINGS_PATH):
        try:
            with open(CATEGORY_MAPPINGS_PATH, "r", encoding="utf-8") as f:
                _category_mappings = json.load(f)
        except Exception as e:
            print(f"Error loading category mappings: {e}")
            _category_mappings = {}
            
    # Load reference stats and extract medians
    if os.path.exists(REF_STATS_PATH):
        try:
            with open(REF_STATS_PATH, "r", encoding="utf-8") as f:
                _reference_stats = json.load(f)
            for col, stats in _reference_stats.items():
                if "percentiles" in stats and len(stats["percentiles"]) >= 6:
                    _feature_medians[col] = stats["percentiles"][5]  # 50th percentile
                else:
                    _feature_medians[col] = stats.get("mean", 0.0)
        except Exception as e:
            print(f"Error loading reference stats: {e}")
            _reference_stats = {}
            _feature_medians = {}

    # Load configuration
    if os.path.exists(SERVING_CONFIG_PATH):
        try:
            with open(SERVING_CONFIG_PATH, "r", encoding="utf-8") as f:
                _serving_config = yaml.safe_load(f) or {}
                _drift_threshold = _serving_config.get("drift_threshold", 0.2)
                
                cal_cfg = _serving_config.get("probability_calibration", {})
                _cal_enabled = cal_cfg.get("enabled", True)
                _cal_method = cal_cfg.get("method", "scale_pos_weight_unbias")
                _cal_weight = float(cal_cfg.get("scale_pos_weight", 4.0))
                
                thresh_cfg = _serving_config.get("decision_thresholds", {})
                _low_max = float(thresh_cfg.get("low_max", 0.15))
                _review_max = float(thresh_cfg.get("review_max", 0.30))
                
                def_cfg = _serving_config.get("defaults", {})
                _clean_history_synthetic = def_cfg.get("clean_credit_history_for_missing_loan", True)
                
                ood_cfg = _serving_config.get("ood", {})
                _ood_enabled = ood_cfg.get("enabled", True)
        except Exception as e:
            print(f"Error loading serving config: {e}")
            _drift_threshold = 0.2
            
    # Environment variable overrides
    _drift_threshold = float(os.environ.get("DRIFT_THRESHOLD", _drift_threshold))


# Initialize metadata
_init_metadata()


def build_base_feature_vector(request_data: Dict[str, Any]) -> tuple[Dict[str, Any], FeatureLookupMetadata]:
    """Looks up an existing loan from feature store or returns a synthetic profile."""
    customer_id = request_data.get("loan_id")
    if customer_id is not None:
        try:
            cid_int = int(customer_id)
            fs = get_feature_store(SERVING_CONFIG_PATH)
            row_dict, source = fs.get_customer_features(cid_int)
            if row_dict is not None:
                return dict(row_dict), FeatureLookupMetadata(customer_id=cid_int, found=True, source=source)
            return {}, FeatureLookupMetadata(customer_id=cid_int, found=False, source=None)
        except (ValueError, TypeError):
            pass
    return {}, FeatureLookupMetadata(customer_id=None, found=False, source=None)


def apply_request_overrides(base_features: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Applies UI form inputs as explicit overrides over base features."""
    features = base_features.copy()
    for k, v in request_data.items():
        if v is not None:
            features[k] = v
    return features


def apply_business_defaults(features: Dict[str, Any], is_synthetic: bool = True) -> Dict[str, Any]:
    """Applies LendingClub-oriented defaults instead of blind population medians."""
    if is_synthetic and _clean_history_synthetic:
        for col in _feature_names:
            if any(pattern in col for pattern in CLEAN_HISTORY_PATTERNS):
                if col not in features or features[col] is None or pd.isna(features[col]):
                    features[col] = 0.0

    return features


def recompute_derived_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """Recomputes LendingClub borrower and credit-history ratios."""
    loan_amnt = float(features.get("loan_amnt", 0.0) or 0.0)
    annual_inc = float(features.get("annual_inc", 0.0) or 0.0)
    term = int(float(features.get("term", 36) or 36))
    features["term"] = float(term)
    installment = float(features.get("installment", 0.0) or 0.0)
    revol_bal = float(features.get("revol_bal", 0.0) or 0.0)
    fico_low = features.get("fico_range_low")
    fico_high = features.get("fico_range_high")

    # If installment is missing or impossible (total repayments < principal), auto-calculate amortized payment
    if installment <= 0.0 or (installment * term < loan_amnt and loan_amnt > 0):
        installment = calculate_amortized_installment(loan_amnt, term)
        features["installment"] = installment

    # Cashflow-first FICO handling (support unbanked/thin-file applicants without FICO)
    has_valid_fico = (
        fico_low is not None
        and fico_high is not None
        and not pd.isna(fico_low)
        and not pd.isna(fico_high)
        and float(fico_low) >= 300.0
    )
    if has_valid_fico:
        features["fico_avg"] = (float(fico_low) + float(fico_high)) / 2.0
    else:
        # Default to median reference baseline for tree scoring (unbanked / cashflow mode)
        features["fico_avg"] = float(_feature_medians.get("fico_avg", 692.0))
        features["fico_range_low"] = float(_feature_medians.get("fico_range_low", 690.0))
        features["fico_range_high"] = float(_feature_medians.get("fico_range_high", 694.0))

    features["loan_income_ratio"] = loan_amnt / annual_inc if annual_inc > 0 else 0.0
    features["installment_income_ratio"] = (installment * 12.0) / annual_inc if annual_inc > 0 else 0.0
    features["revol_balance_income_ratio"] = revol_bal / annual_inc if annual_inc > 0 else 0.0

    return features


def check_ood_and_anomalies(features: Dict[str, Any]) -> List[PredictionWarning]:
    """Detects LendingClub out-of-distribution inputs and anomalies."""
    warnings: List[PredictionWarning] = []
    
    annual_inc = float(features.get("annual_inc", 0.0) or 0.0)
    loan_amnt = float(features.get("loan_amnt", 0.0) or 0.0)
    dti = float(features.get("dti", 0.0) or 0.0)
    fico_low_raw = features.get("fico_range_low")
    fico_high_raw = features.get("fico_range_high")
    term = float(features.get("term", 0.0) or 0.0)

    if annual_inc > 1_000_000:
        warnings.append(PredictionWarning(
            code="OUT_OF_DISTRIBUTION",
            field="annual_inc",
            message=f"Annual income (${annual_inc:,.0f}) is unusually high for the LendingClub training population."
        ))

    if annual_inc > 500_000 and loan_amnt > 0:
        ratio = loan_amnt / annual_inc
        if ratio < 0.10:
            warnings.append(PredictionWarning(
                code="UNUSUAL_RATIO",
                field="loan_income_ratio",
                message=f"Loan-to-income ratio ({ratio:.2%}) is unusually small for a personal loan."
            ))

    if dti < 0 or dti > 100:
        warnings.append(PredictionWarning(
            code="ANOMALY",
            field="dti",
            message=f"Debt-to-income ratio ({dti:.1f}) falls outside the expected LendingClub range."
        ))

    if fico_low_raw is not None and fico_high_raw is not None:
        try:
            fl = float(fico_low_raw)
            fh = float(fico_high_raw)
            if 0 < fl < 300 or fh > 850 or (fl > 0 and fl > fh):
                warnings.append(PredictionWarning(
                    code="ANOMALY",
                    field="fico_range_low",
                    message="FICO range should fall between 300 and 850, with the low value not exceeding the high value."
                ))
        except (ValueError, TypeError):
            pass

    if term and term not in {36, 60}:
        warnings.append(PredictionWarning(
            code="ANOMALY",
            field="term",
            message="LendingClub loan term is usually 36 or 60 months."
        ))

    return warnings


def encode_categorical_overrides(features: Dict[str, Any]) -> Dict[str, Any]:
    """Encode request-time categorical strings using mappings saved during feature building."""
    for col, mapping in _category_mappings.items():
        if col in features and features[col] is not None and not pd.isna(features[col]):
            value = str(features[col])
            features[col] = mapping.get(value, len(mapping))
    return features


def align_to_model_schema(features: Dict[str, Any]) -> tuple[pd.DataFrame, Dict[str, float]]:
    """Aligns feature dictionary with exact model column order and imputes remaining unknown medians."""
    df_data = {}
    for col in _feature_names:
        val = features.get(col)
        if val is None or pd.isna(val):
            val = _feature_medians.get(col, 0.0)
        df_data[col] = [val]
    return pd.DataFrame(df_data), {col: df_data[col][0] for col in _feature_names}


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Returns FastAPI container health status."""
    try:
        model = get_model()
        version = get_model_version()
        name = get_model_name()
        m_type = get_model_type()
        stage = get_model_stage()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=version,
            model_name=name,
            model_type=m_type,
            model_stage=stage
        )
    except Exception:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            model_version="",
            model_name="",
            model_type="",
            model_stage=""
        )


@router.post("/predict", response_model=PredictResponse)
async def predict(request_data: Dict[str, Any]) -> PredictResponse:
    """
    Scores a credit application through the Underwriting Adapter pipeline:
    Customer Lookup -> Overrides -> Smart Defaults -> Recompute Derived -> OOD Checks -> Model Inference -> Calibration.
    """
    if not _feature_names:
        _init_metadata()

    try:
        model = get_model()
        model_version = get_model_version()
    except Exception:
        raise HTTPException(status_code=503, detail="Model not available")

    for req in REQUIRED_LENDINGCLUB_INPUTS:
        if req not in request_data:
            raise HTTPException(status_code=422, detail=f"Missing required field: {req}")

    request_id = str(uuid.uuid4())

    # 1. Base features from Feature Store or empty
    base_features, lookup_meta = build_base_feature_vector(request_data)

    # 2. Apply overrides from request_data
    features = apply_request_overrides(base_features, request_data)

    # 3. Apply business defaults
    is_synthetic = not lookup_meta.found
    features = apply_business_defaults(features, is_synthetic=is_synthetic)

    # 4. Recompute derived features & auto-amortization
    features = recompute_derived_features(features)

    # 5. Encode request-time categorical inputs with training mappings
    features = encode_categorical_overrides(features)

    # 6. Evaluate Tier-1 Financial Sanity & Solvency Guardrails
    loan_raw = float(request_data.get("loan_amnt", features.get("loan_amnt", 0.0)) or 0.0)
    annual_inc = float(features.get("annual_inc", 0.0) or 0.0)
    term = int(float(features.get("term", 36) or 36))
    raw_inst = float(request_data.get("installment", 0.0) or 0.0)
    eval_inst = raw_inst if raw_inst > 0 else float(features.get("installment", 0.0) or 0.0)
    dti_val = float(features.get("dti", 0.0) or 0.0)

    solvency_dict, solvency_warnings_raw, is_knockout, knockout_reasons = evaluate_solvency_and_guardrails(
        loan_amnt=loan_raw,
        annual_inc=annual_inc,
        term=term,
        installment=eval_inst,
        dti=dti_val
    )
    solvency_obj = SolvencyDiagnostics(**solvency_dict)
    solvency_warnings = [
        PredictionWarning(code=w["code"], field=w.get("field"), message=w["message"])
        for w in solvency_warnings_raw
    ]

    # 7. Check OOD and anomalies
    ood_warnings = check_ood_and_anomalies(features) if _ood_enabled else []
    all_warnings = solvency_warnings + ood_warnings

    # 8. Clamp loan amount to platform ceiling for tree inference (prevents sparse-leaf extrapolation)
    features["loan_amnt"] = min(loan_raw, PLATFORM_MAX_LOAN)

    # 9. Align to model schema
    df, window_record = align_to_model_schema(features)

    # 10. Model inference
    try:
        raw_prob = float(model.predict_proba(df)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference prediction failed: {e}")

    # 11. Probability calibration
    if _cal_enabled:
        calibrated_prob = calibrate_probability(
            raw_prob=raw_prob,
            method=_cal_method,
            scale_pos_weight=_cal_weight
        )
    else:
        calibrated_prob = raw_prob

    # 12. Two-Tier Decision label logic (Tier 1 Policy Knockouts take precedence)
    if is_knockout:
        decision_label = "High Attention" if calibrated_prob >= _low_max else "Needs Review"
    else:
        if calibrated_prob < _low_max:
            decision_label = "Low Risk"
        elif calibrated_prob <= _review_max:
            decision_label = "Needs Review"
        else:
            decision_label = "High Attention"

    # 13. Record in drift rolling window
    add_request(window_record, calibrated_prob)

    # 14. Return response
    return PredictResponse(
        default_probability=calibrated_prob,
        raw_model_probability=raw_prob,
        calibration_method=_cal_method if _cal_enabled else "none",
        decision_label=decision_label,
        model_version=model_version,
        model_name=get_model_name(),
        request_id=request_id,
        feature_lookup=lookup_meta,
        solvency=solvency_obj,
        warnings=all_warnings
    )


@router.get("/drift", response_model=DriftResponse)
async def drift(min_records: int = Query(default=50, ge=1)) -> DriftResponse:
    """Monitor feature distribution shift between training and recent traffic."""
    window_size = get_window_size()
    if window_size < min_records:
        raise HTTPException(
            status_code=503,
            detail=f"Insufficient data: {window_size} records in window, need {min_records}"
        )

    # Lazily ensure reference stats are loaded
    if not _reference_stats:
        _init_metadata()

    # Build DataFrame from rolling window
    window_records = get_window_records()
    window_df = pd.DataFrame(window_records)

    # Compute PSI for all features
    try:
        psi_values = compute_all_psi(_reference_stats, window_df, bins=10)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift computation failed: {e}")

    # Check alert threshold
    alert = any(v >= _drift_threshold for v in psi_values.values())

    if alert:
        from src.monitoring.alert import check_and_alert
        # Read SNS_ARN from environment variables or serving config
        sns_arn = os.environ.get("SNS_ARN", None)
        check_and_alert(psi_values, threshold=_drift_threshold, sns_arn=sns_arn)

    return DriftResponse(
        psi_values=psi_values,
        n_records_evaluated=window_size,
        alert=alert,
        threshold=_drift_threshold
    )


@router.get("/model-comparison", response_model=ModelComparisonResponse)
async def model_comparison() -> ModelComparisonResponse:
    """Returns benchmark comparison metrics across all 5 evaluated model architectures, 
    selection rationale, and statistical significance test outcomes."""
    models = [
        ModelBenchmarkItem(
            model_id="xgboost",
            name="LendingClub XGBoost Default Risk Classifier",
            architecture="Extreme Gradient Boosted Trees (GPU baseline)",
            roc_auc=0.70487,
            ks_stat=0.29824,
            brier_score=0.24073,
            calibration_slope=0.67559,
            inference_latency_ms=1.2,
            status="GPU Baseline",
            is_champion=False
        ),
        ModelBenchmarkItem(
            model_id="lgbm",
            name="LendingClub LightGBM Default Risk Classifier",
            architecture="Leaf-wise GBDT (GPU baseline)",
            roc_auc=0.70616,
            ks_stat=0.29888,
            brier_score=0.24802,
            calibration_slope=0.66812,
            inference_latency_ms=1.0,
            status="Current Production Champion",
            is_champion=True
        ),
        ModelBenchmarkItem(
            model_id="entity_embedding",
            name="LendingClub Entity Embedding Neural Network",
            architecture="Learned Categorical Embeddings + Dense Residual MLP",
            roc_auc=0.70679,
            ks_stat=0.29967,
            brier_score=0.26739,
            calibration_slope=0.67625,
            inference_latency_ms=5.0,
            status="Neural Challenger",
            is_champion=False
        ),
        ModelBenchmarkItem(
            model_id="mlp",
            name="LendingClub Multi-Layer Perceptron",
            architecture="Fully Connected Dense Network (512-256-128 + BatchNorm)",
            roc_auc=0.69755,
            ks_stat=0.28861,
            brier_score=0.23261,
            calibration_slope=0.71692,
            inference_latency_ms=4.0,
            status="Neural Baseline",
            is_champion=False
        ),
        ModelBenchmarkItem(
            model_id="tab_transformer",
            name="LendingClub TabTransformer Attention Network",
            architecture="Multi-Head Column Self-Attention + Continuous MLP",
            roc_auc=0.65670,
            ks_stat=0.22281,
            brier_score=0.21660,
            calibration_slope=0.66458,
            inference_latency_ms=12.0,
            status="Experimental Neural Model",
            is_champion=False
        ),
    ]

    selection_rationales = [
        SelectionRationaleItem(
            title="LightGBM Selected for Production",
            summary="LightGBM is the selected production model for the LendingClub application.",
            details="LightGBM provides nearly identical ranking performance to Entity Embedding while offering substantially better Brier score and lower inference latency. It is selected as the practical production champion; Entity Embedding remains a neural challenger."
        ),
        SelectionRationaleItem(
            title="Initial Model Family",
            summary="Tree ensembles remain the best first pass for tabular credit-risk data.",
            details="All five model families have now been retrained on the LendingClub feature matrix. LightGBM is selected for production because it offers the best operational balance of ranking, calibration, and latency."
        ),
        SelectionRationaleItem(
            title="Leakage Control",
            summary="Post-origination repayment fields are excluded from the new feature contract.",
            details="The migration keeps application/origination fields such as income, DTI, loan amount, term, employment length, and FICO range while excluding repayment and recovery fields."
        ),
        SelectionRationaleItem(
            title="Temporal Evaluation",
            summary="LendingClub loans should be evaluated with newer loans held out.",
            details="The feature pipeline sorts by issue date before writing processed features so the training split approximates a realistic forward-looking validation setup."
        ),
        SelectionRationaleItem(
            title="Calibration Pending",
            summary="Decision thresholds and class weights must be recomputed after training.",
            details="The serving layer now uses the LightGBM artifact and a LendingClub class-weight correction from the first temporal training split. Risk thresholds should be revisited after calibration analysis."
        )
    ]

    significance_tests = [
        WilcoxonResultItem(
        comparison="LightGBM vs Entity Embedding",
            statistic=1.0,
            p_value=1.0,
            significant=False,
            interpretation="Pending repeated validation after hyperparameter tuning."
        ),
        WilcoxonResultItem(
            comparison="XGBoost vs LightGBM",
            statistic=8.0,
            p_value=1.0,
            significant=False,
            interpretation="First-pass holdout metrics are nearly tied; repeat after tuning."
        ),
        WilcoxonResultItem(
            comparison="MLP vs Entity Embedding",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            interpretation="Entity Embedding remains the neural challenger while LightGBM is used for production serving."
        ),
    ]

    evaluation_dataset = {
        "train_samples": 970631,
        "val_samples": 121328,
        "test_samples": 121330,
        "features_evaluated": 30,
        "evaluation_protocol": "Temporal holdout using LendingClub issue-date ordering before feature export"
    }

    return ModelComparisonResponse(
        champion_model_id="lgbm",
        champion_model_name="LendingClub LightGBM Default Risk Classifier",
        models=models,
        selection_rationales=selection_rationales,
        significance_tests=significance_tests,
        evaluation_dataset=evaluation_dataset
    )
