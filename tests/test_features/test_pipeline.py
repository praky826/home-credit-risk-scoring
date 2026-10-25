import os
import json
import pytest
import pandas as pd
from src.features.pipeline import build_all_features

def test_pipeline_execution():
    config_path = "configs/features.yaml"
    output_dir = "data/processed"
    
    # Run pipeline (should execute without error using existing mock datasets)
    build_all_features(config_path=config_path, output_dir=output_dir)

    # Check output files exist
    train_path = os.path.join(output_dir, "train_features.parquet")
    test_path = os.path.join(output_dir, "test_features.parquet")
    feat_names_path = os.path.join(output_dir, "feature_names.json")
    cat_indices_path = os.path.join(output_dir, "cat_feature_indices.json")
    cat_cards_path = os.path.join(output_dir, "cat_cardinalities.json")
    ref_stats_path = "data/drift/reference_stats.json"

    assert os.path.exists(train_path)
    assert os.path.exists(test_path)
    assert os.path.exists(feat_names_path)
    assert os.path.exists(cat_indices_path)
    assert os.path.exists(cat_cards_path)
    assert os.path.exists(ref_stats_path)

    # Load outputs to verify structure
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    assert "loan_id" in train_df.columns
    assert "TARGET" in train_df.columns
    assert "loan_id" in test_df.columns
    assert "TARGET" not in test_df.columns
    assert train_df.shape[0] > 1000
    assert test_df.shape[0] > 100

    # Load JSON metadata
    with open(feat_names_path, "r") as f:
        feature_names = json.load(f)
    with open(cat_indices_path, "r") as f:
        cat_indices = json.load(f)
    with open(cat_cards_path, "r") as f:
        cat_cards = json.load(f)

    assert isinstance(feature_names, list)
    assert isinstance(cat_indices, list)
    assert isinstance(cat_cards, dict)
    
    # Feature names should not contain identifier or target columns
    assert "loan_id" not in feature_names
    assert "TARGET" not in feature_names
    assert "loan_amnt" in feature_names
    assert "annual_inc" in feature_names
    assert "fico_avg" in feature_names
    
    # All categorical index positions should map to elements in feature_names
    for idx in cat_indices:
        assert idx < len(feature_names)
