import os
import pytest
import numpy as np
import pandas as pd
from src.models.xgboost_model import XGBoostModel
from src.models.lgbm_model import LGBMModel

@pytest.fixture
def toy_data():
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f"feat_{i}" for i in range(5)])
    y_train = pd.Series(np.random.choice([0, 1], 100))
    X_val = pd.DataFrame(np.random.randn(20, 5), columns=[f"feat_{i}" for i in range(5)])
    y_val = pd.Series(np.random.choice([0, 1], 20))
    return X_train, y_train, X_val, y_val

def test_xgboost_model(toy_data, tmp_path):
    X_train, y_train, X_val, y_val = toy_data
    
    config = {
        "params": {
            "n_estimators": 5,
            "max_depth": 3,
            "learning_rate": 0.1,
            "random_state": 42
        }
    }
    
    model = XGBoostModel(config)
    model.fit(X_train, y_train, X_val, y_val, tune=False)
    
    # Predict check
    probs = model.predict_proba(X_val)
    assert isinstance(probs, np.ndarray)
    assert len(probs) == len(X_val)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    
    # Save & Load check
    save_path = os.path.join(tmp_path, "xgb_test.pkl")
    model.save(save_path)
    assert os.path.exists(save_path)
    
    loaded_model = XGBoostModel.load(save_path)
    loaded_probs = loaded_model.predict_proba(X_val)
    np.testing.assert_array_almost_equal(probs, loaded_probs)

def test_lgbm_model(toy_data, tmp_path):
    X_train, y_train, X_val, y_val = toy_data
    
    config = {
        "params": {
            "n_estimators": 5,
            "num_leaves": 7,
            "learning_rate": 0.1,
            "random_state": 42,
            "verbose": -1
        }
    }
    
    model = LGBMModel(config)
    model.fit(X_train, y_train, X_val, y_val, tune=False)
    
    # Predict check
    probs = model.predict_proba(X_val)
    assert isinstance(probs, np.ndarray)
    assert len(probs) == len(X_val)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    
    # Save & Load check
    save_path = os.path.join(tmp_path, "lgb_test.pkl")
    model.save(save_path)
    assert os.path.exists(save_path)
    
    loaded_model = LGBMModel.load(save_path)
    loaded_probs = loaded_model.predict_proba(X_val)
    np.testing.assert_array_almost_equal(probs, loaded_probs)
