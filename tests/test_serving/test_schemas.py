import pytest
from pydantic import ValidationError
from src.serving.schemas import PredictRequest, PredictResponse, HealthResponse, DriftResponse

def test_schemas_validation():
    # 1. HealthResponse
    hr = HealthResponse(status="ok", model_loaded=True, model_version="1.0.0")
    assert hr.status == "ok"
    assert hr.model_loaded is True

    # 2. PredictResponse
    pr = PredictResponse(default_probability=0.0723, model_version="v1", request_id="uuid-string")
    assert pr.default_probability == 0.0723
    with pytest.raises(ValidationError):
        # probability out of range
        PredictResponse(default_probability=1.5, model_version="v1", request_id="uuid")

    # 3. PredictRequest (dynamically created)
    # Correct body
    req_body = {
        "AMT_CREDIT": 500000.0,
        "AMT_INCOME_TOTAL": 150000.0,
        "DAYS_BIRTH": -15000,
        "DAYS_EMPLOYED": -2000,
        "EXT_SOURCE_2": 0.5
    }
    req = PredictRequest(**req_body)
    assert req.AMT_CREDIT == 500000.0
    
    # Extra fields should be allowed (config extra='allow')
    req_extra = PredictRequest(**{**req_body, "EXTRA_UNKNOWN_FIELD": "test-val"})
    assert getattr(req_extra, "EXTRA_UNKNOWN_FIELD", None) == "test-val"

    # Missing required field
    with pytest.raises(ValidationError):
        PredictRequest(
            AMT_INCOME_TOTAL=150000.0,
            DAYS_BIRTH=-15000,
            DAYS_EMPLOYED=-2000
        )
