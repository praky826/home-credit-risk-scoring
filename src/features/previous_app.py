import pandas as pd
import numpy as np
from src.features.utils import reduce_memory_usage, safe_divide

def build_previous_app_features(
    path: str = "data/raw/previous_application.csv"
) -> pd.DataFrame:
    """
    Returns DataFrame indexed by SK_ID_CURR with aggregated previous application features.
    """
    df = pd.read_csv(path)

    # Derived columns (computed before grouping)
    df["CREDIT_TO_APP_RATIO"] = safe_divide(df["AMT_CREDIT"], df["AMT_APPLICATION"])

    # Columns for status counting
    df["PREV_APPROVED"] = (df["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    df["PREV_REFUSED"] = (df["NAME_CONTRACT_STATUS"] == "Refused").astype(int)

    # Aggregations grouped by SK_ID_CURR
    prev_agg = df.groupby("SK_ID_CURR").agg(
        PREV_APP_COUNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_COUNT=("PREV_APPROVED", "sum"),
        PREV_REFUSED_COUNT=("PREV_REFUSED", "sum"),
        PREV_AMT_APPLICATION_mean=("AMT_APPLICATION", "mean"),
        PREV_AMT_APPLICATION_max=("AMT_APPLICATION", "max"),
        PREV_AMT_CREDIT_mean=("AMT_CREDIT", "mean"),
        PREV_AMT_CREDIT_max=("AMT_CREDIT", "max"),
        PREV_AMT_ANNUITY_mean=("AMT_ANNUITY", "mean"),
        PREV_DAYS_DECISION_mean=("DAYS_DECISION", "mean"),
        PREV_DAYS_DECISION_min=("DAYS_DECISION", "min"),
        PREV_CREDIT_TO_APP_RATIO_mean=("CREDIT_TO_APP_RATIO", "mean"),
        PREV_PRODUCT_TYPE_nunique=("NAME_PRODUCT_TYPE", "nunique")
    )

    # Calculate approval rate
    prev_agg["PREV_APPROVAL_RATE"] = safe_divide(
        prev_agg["PREV_APPROVED_COUNT"], prev_agg["PREV_APP_COUNT"]
    )

    # Downcast memory footprint
    prev_agg = reduce_memory_usage(prev_agg, verbose=False)

    return prev_agg
