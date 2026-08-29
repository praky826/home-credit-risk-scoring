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
        "AMT_CREDIT": 500000.0,
        "AMT_INCOME_TOTAL": 150000.0,
        "DAYS_BIRTH": -15000,
        "DAYS_EMPLOYED": -2000,
        "EXT_SOURCE_2": 0.5,
        "SK_ID_CURR": 999999
    }
    
    response = client.post("/predict", json=request_body)
    assert response.status_code == 200
    data = response.json()
    assert "default_probability" in data
    assert "model_version" in data
    assert "request_id" in data
    assert 0.0 <= data["default_probability"] <= 1.0

def test_predict_missing_required(client):
    # Missing required AMT_CREDIT
    request_body = {
        "AMT_INCOME_TOTAL": 150000.0,
        "DAYS_BIRTH": -15000,
        "DAYS_EMPLOYED": -2000
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
            "AMT_CREDIT": 500000.0 + i * 1000,
            "AMT_INCOME_TOTAL": 150000.0,
            "DAYS_BIRTH": -15000,
            "DAYS_EMPLOYED": -2000,
            "EXT_SOURCE_2": 0.5
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
    assert champion["model_id"] == "xgboost"
    assert champion["roc_auc"] > 0.78

