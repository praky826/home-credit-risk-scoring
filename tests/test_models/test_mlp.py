import os
import json
import pytest
import numpy as np
import pandas as pd
from src.models.mlp import MLPModel
from src.models.entity_embedding import EntityEmbeddingModel
from src.models.tab_transformer import TabTransformerModel

@pytest.fixture
def toy_data():
    np.random.seed(42)
    # Generate 10 features, first 2 will simulate categorical indices [0, 1]
    X_train = pd.DataFrame(np.random.randn(80, 10), columns=[f"feat_{i}" for i in range(10)])
    # Make first two features integers for category encoding simulation
    X_train["feat_0"] = np.random.randint(0, 3, 80).astype(float)
    X_train["feat_1"] = np.random.randint(0, 5, 80).astype(float)
    
    y_train = pd.Series(np.random.choice([0, 1], 80))
    
    X_val = pd.DataFrame(np.random.randn(20, 10), columns=[f"feat_{i}" for i in range(10)])
    X_val["feat_0"] = np.random.randint(0, 3, 20).astype(float)
    X_val["feat_1"] = np.random.randint(0, 5, 20).astype(float)
    
    y_val = pd.Series(np.random.choice([0, 1], 20))
    return X_train, y_train, X_val, y_val

def test_mlp_model(toy_data, tmp_path):
    X_train, y_train, X_val, y_val = toy_data
    
    config = {
        "params": {
            "hidden_dims": [16, 8],
            "batch_size": 16,
            "epochs": 2,
            "patience": 1,
            "lr": 0.01,
            "dropout": 0.1,
            "weight_decay": 1e-4
        }
    }
    
    model = MLPModel(config)
    model.fit(X_train, y_train, X_val, y_val)
    
    probs = model.predict_proba(X_val)
    assert isinstance(probs, np.ndarray)
    assert len(probs) == len(X_val)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    
    save_path = os.path.join(tmp_path, "mlp_test.pt")
    model.save(save_path)
    assert os.path.exists(save_path)
    
    loaded = MLPModel.load(save_path)
    loaded_probs = loaded.predict_proba(X_val)
    np.testing.assert_array_almost_equal(probs, loaded_probs, decimal=5)

def test_entity_embedding_model(toy_data, tmp_path):
    X_train, y_train, X_val, y_val = toy_data
    
    config = {
        "params": {
            "hidden_dims": [16, 8],
            "batch_size": 16,
            "epochs": 2,
            "patience": 1,
            "lr": 0.01,
            "dropout": 0.1,
            "weight_decay": 1e-4
        }
    }
    
    # Mock metadata files
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/cat_feature_indices.json", "w") as f:
        json.dump([0, 1], f)
    with open("data/processed/cat_cardinalities.json", "w") as f:
        json.dump({"feat_0": 3, "feat_1": 5}, f)
        
    model = EntityEmbeddingModel(config)
    model.fit(X_train, y_train, X_val, y_val)
    
    probs = model.predict_proba(X_val)
    assert isinstance(probs, np.ndarray)
    assert len(probs) == len(X_val)
    
    save_path = os.path.join(tmp_path, "een_test.pt")
    model.save(save_path)
    assert os.path.exists(save_path)
    
    loaded = EntityEmbeddingModel.load(save_path)
    loaded_probs = loaded.predict_proba(X_val)
    np.testing.assert_array_almost_equal(probs, loaded_probs, decimal=5)

def test_tab_transformer_model(toy_data, tmp_path):
    X_train, y_train, X_val, y_val = toy_data
    
    config = {
        "params": {
            "d_model": 8,
            "n_heads": 2,
            "n_layers": 1,
            "ff_dim": 16,
            "dropout": 0.1,
            "mlp_hidden": [16, 8],
            "batch_size": 16,
            "epochs": 2,
            "patience": 1,
            "lr": 0.01,
            "weight_decay": 1e-4
        }
    }
    
    # Mock metadata files
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/cat_feature_indices.json", "w") as f:
        json.dump([0, 1], f)
    with open("data/processed/cat_cardinalities.json", "w") as f:
        json.dump({"feat_0": 3, "feat_1": 5}, f)
        
    model = TabTransformerModel(config)
    model.fit(X_train, y_train, X_val, y_val)
    
    probs = model.predict_proba(X_val)
    assert isinstance(probs, np.ndarray)
    assert len(probs) == len(X_val)
    
    save_path = os.path.join(tmp_path, "tabtrans_test.pt")
    model.save(save_path)
    assert os.path.exists(save_path)
    
    loaded = TabTransformerModel.load(save_path)
    loaded_probs = loaded.predict_proba(X_val)
    np.testing.assert_array_almost_equal(probs, loaded_probs, decimal=5)
