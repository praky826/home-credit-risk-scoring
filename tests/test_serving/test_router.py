import pytest
import os
import json
from fastapi.testclient import TestClient
from src.serving.app import app
from src.serving.drift import clear_window

@pytest.fixture
def client():
    # Make sure model is loaded before testing or health status check
    # We can trigger startup event manually or let TestClient handle it (TestClient(app) calls startup hooks inside context manager)
    with TestClient(app) as c:
        yield c
    clear_window()

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_version" in data

def test_predict_endpoint(client):
    # Single record prediction body matching required columns
    request_body = {
        "loan_amnt": 12000.0,
        "annual_inc": 72000.0,
        "term": 36,
        "installment": 390.5,
        "dti": 18.4,
        "fico_range_low": 690,
        "fico_range_high": 694,
        "loan_id": 999999,
    }
    
    response = client.post("/predict", json=request_body)
    assert response.status_code == 200
    data = response.json()
    assert "default_probability" in data
    assert "model_version" in data
    assert "request_id" in data
    assert 0.0 <= data["default_probability"] <= 1.0

def test_predict_missing_required(client):
    # Missing required loan_amnt
    request_body = {
        "annual_inc": 72000.0,
        "dti": 18.4,
        "fico_range_low": 690,
        "fico_range_high": 694,
    }
    response = client.post("/predict", json=request_body)
    assert response.status_code == 422

def test_drift_endpoint(client):
    clear_window()
    
    # 1. Check Insufficient data error (under 50 records)
    response = client.get("/drift?min_records=10")
    assert response.status_code == 503
    assert "Insufficient data" in response.json()["detail"]
    
    # 2. Add 10 records to pass min_records threshold
    for i in range(10):
        client.post("/predict", json={
            "loan_amnt": 12000.0 + i * 100,
            "annual_inc": 72000.0,
            "term": 36,
            "installment": 390.5,
            "dti": 18.4,
            "fico_range_low": 690,
            "fico_range_high": 694,
        })
        
    response = client.get("/drift?min_records=10")
    assert response.status_code == 200
    data = response.json()
    assert "psi_values" in data
    assert "n_records_evaluated" in data
    assert "alert" in data
    assert data["n_records_evaluated"] == 10

def test_model_comparison_endpoint(client):
    response = client.get("/model-comparison")
    assert response.status_code == 200
    data = response.json()
    assert "champion_model_id" in data
    assert data["champion_model_id"] == "xgboost"
    assert "champion_model_name" in data
    assert "models" in data
    assert len(data["models"]) == 5
    assert "selection_rationales" in data
    assert len(data["selection_rationales"]) >= 4
    assert "significance_tests" in data
    assert len(data["significance_tests"]) >= 3
    # Check champion model in models list
    champion = next(m for m in data["models"] if m["is_champion"])
    champion = next(m for m in data["models"] if m["is_champion"])
    assert champion["model_id"] == "xgboost"
    assert champion["status"] == "Current Local Champion"
    assert champion["roc_auc"] == pytest.approx(0.70647)


def test_probability_calibration_math():
    from src.serving.calibration import unweight_probability
    # 0.33 raw probability with scale_pos_weight 11.38 should be ~0.0415
    calibrated = unweight_probability(0.33, scale_pos_weight=11.38)
    assert 0.039 < calibrated < 0.044
    assert unweight_probability(0.0) == 0.0
    assert unweight_probability(1.0) == 1.0


def test_predict_with_feature_store_lookup(client):
    # loan_id 70686 is in the processed LendingClub train_features.parquet
    request_body = {
        "loan_id": 70686,
        "loan_amnt": 10000.0,
        "annual_inc": 70000.0,
        "term": 36,
        "installment": 320.0,
        "dti": 16.0,
        "fico_range_low": 680,
        "fico_range_high": 684,
    }
    response = client.post("/predict", json=request_body)
    assert response.status_code == 200
    data = response.json()
    assert "feature_lookup" in data
    assert data["feature_lookup"]["found"] is True
    assert data["feature_lookup"]["customer_id"] == 70686
    assert data["feature_lookup"]["source"] == "train_features"
    assert "raw_model_probability" in data
    assert "default_probability" in data
    assert "decision_label" in data


def test_predict_ood_income_warning(client):
    # 2M income should trigger OOD and low loan-income ratio warning
    request_body = {
        "loan_amnt": 10000.0,
        "annual_inc": 2000000.0,
        "term": 36,
        "installment": 320.0,
        "dti": 3.0,
        "fico_range_low": 760,
        "fico_range_high": 764,
    }
    response = client.post("/predict", json=request_body)
    assert response.status_code == 200
    data = response.json()
    assert "warnings" in data
    warning_codes = [w["code"] for w in data["warnings"]]
    assert "OUT_OF_DISTRIBUTION" in warning_codes
    assert "UNUSUAL_RATIO" in warning_codes
    # Calibrated probability should be low (< low_max => Low Risk)
    assert data["default_probability"] < 0.15
    assert data["decision_label"] == "Low Risk"


def test_clean_defaults_for_synthetic_applicant(client):
    # Synthetic applicant without customer lookup
    request_body = {
        "loan_id": 888888,
        "loan_amnt": 12000.0,
        "annual_inc": 120000.0,
        "term": 36,
        "installment": 390.5,
        "dti": 10.0,
        "fico_range_low": 760,
        "fico_range_high": 764,
        "delinq_2yrs": 0,
        "inq_last_6mths": 0,
        "pub_rec": 0,
    }
    response = client.post("/predict", json=request_body)
    assert response.status_code == 200
    data = response.json()
    assert data["feature_lookup"]["found"] is False
    assert data["default_probability"] < 0.15
    assert data["decision_label"] == "Low Risk"


def test_amortization_math():
    from src.serving.underwriting_rules import calculate_amortized_installment
    # $12,000 for 36 months at 11.5% APR should be ~395.79/month
    inst_12k = calculate_amortized_installment(12000.0, 36)
    assert 390.0 < inst_12k < 405.0

    # $120,000 for 36 months should be ~3,957.90/month
    inst_120k = calculate_amortized_installment(120000.0, 36)
    assert 3900.0 < inst_120k < 4050.0


def test_impossible_amortization_knockout(client):
    # $120k loan with $390.50 payment -> Impossible loan
    request_body = {
        "loan_amnt": 120000.0,
        "annual_inc": 72000.0,
        "term": 36,
        "installment": 390.5,
        "dti": 18.4,
    }
    response = client.post("/predict", json=request_body)
    assert response.status_code == 200
    data = response.json()
    assert "solvency" in data
    assert data["solvency"]["policy_knockout"] is True
    assert "IMPOSSIBLE_AMORTIZATION" in [w["code"] for w in data["warnings"]]
    assert "EXCEEDS_PLATFORM_CAP" in [w["code"] for w in data["warnings"]]
    # The impossible loan should NEVER be marked "Low Risk"
    assert data["decision_label"] in ["Needs Review", "High Attention"]


def test_cashflow_scoring_without_fico(client):
    # Pure cashflow underwriting: no FICO scores provided
    request_body = {
        "loan_amnt": 12000.0,
        "annual_inc": 120000.0,
        "term": 36,
        "dti": 12.0,
    }
    response = client.post("/predict", json=request_body)
    assert response.status_code == 200
    data = response.json()
    assert "default_probability" in data
    assert 0.0 <= data["default_probability"] <= 1.0
    assert "solvency" in data
    assert data["solvency"]["disposable_monthly_cashflow"] > 0
    assert data["solvency"]["policy_knockout"] is False


def test_severe_debt_burden_knockout(client):
    # Borrower earning only $20,000/yr asking for $35,000 loan
    request_body = {
        "loan_amnt": 35000.0,
        "annual_inc": 20000.0,
        "term": 36,
        "dti": 35.0,
    }
    response = client.post("/predict", json=request_body)
    assert response.status_code == 200
    data = response.json()
    assert data["solvency"]["policy_knockout"] is True
    warning_codes = [w["code"] for w in data["warnings"]]
    assert "SEVERE_DEBT_BURDEN" in warning_codes or "NEGATIVE_DISPOSABLE_CASHFLOW" in warning_codes
    assert data["decision_label"] == "High Attention"
