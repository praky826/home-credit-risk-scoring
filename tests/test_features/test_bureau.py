import pytest
import pandas as pd
from src.features.bureau import build_bureau_features

def test_build_bureau_features():
    agg_df = build_bureau_features(
        bureau_path="data/raw/bureau.csv",
        balance_path="data/raw/bureau_balance.csv"
    )

    # Verify types and index
    assert isinstance(agg_df, pd.DataFrame)
    assert agg_df.index.name == "SK_ID_CURR"

    # Verify key columns
    assert "BUREAU_LOAN_COUNT" in agg_df.columns
    assert "BUREAU_ACTIVE_LOAN_COUNT" in agg_df.columns
    assert "BUREAU_AMT_CREDIT_SUM_mean" in agg_df.columns
    assert "BBAL_MONTHS_BALANCE_count" in agg_df.columns
    assert "BBAL_STATUS_C_rate" in agg_df.columns
    assert "BBAL_STATUS_X_rate" in agg_df.columns
    assert "BBAL_DPD_MAX" in agg_df.columns
