import pytest
import pandas as pd
import numpy as np
from src.features.application import build_application_features

def test_build_application_features():
    train_df, test_df = build_application_features(
        train_path="data/raw/application_train.csv",
        test_path="data/raw/application_test.csv"
    )

    # Check types and indices
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert train_df.index.name == "SK_ID_CURR"
    assert test_df.index.name == "SK_ID_CURR"

    # Check TARGET is present in train, but not in test
    assert "TARGET" in train_df.columns
    assert "TARGET" not in test_df.columns

    # Check engineered features
    assert "CREDIT_INCOME_RATIO" in train_df.columns
    assert "ANNUITY_INCOME_RATIO" in train_df.columns
    assert "CREDIT_TERM" in train_df.columns
    assert "DAYS_EMPLOYED_PERCENT" in train_df.columns
    assert "EMPLOYED_ANOM" in train_df.columns

    # Verify no missing values in numeric features
    feature_cols = [c for c in train_df.columns if c != "TARGET"]
    num_cols = train_df[feature_cols].select_dtypes(include=[np.number]).columns
    assert train_df[num_cols].isnull().sum().sum() == 0
    assert test_df[num_cols].isnull().sum().sum() == 0
