import pandas as pd
import numpy as np
from src.features.utils import reduce_memory_usage, safe_divide

def build_credit_card_features(
    path: str = "data/raw/credit_card_balance.csv"
) -> pd.DataFrame:
    """
    Returns DataFrame indexed by SK_ID_CURR.
    """
    df = pd.read_csv(path)

    # Derived columns (before grouping)
    df["UTILIZATION"] = safe_divide(df["AMT_BALANCE"], df["AMT_CREDIT_LIMIT_ACTUAL"]).clip(0.0, 1.0)

    cc_agg = df.groupby("SK_ID_CURR").agg(
        CC_COUNT=("SK_ID_PREV", "count"),
        CC_AMT_BALANCE_mean=("AMT_BALANCE", "mean"),
        CC_AMT_BALANCE_max=("AMT_BALANCE", "max"),
        CC_UTILIZATION_mean=("UTILIZATION", "mean"),
        CC_UTILIZATION_max=("UTILIZATION", "max"),
        CC_AMT_PAYMENT_TOTAL_CURRENT_mean=("AMT_PAYMENT_TOTAL_CURRENT", "mean"),
        CC_AMT_DRAWINGS_TOTAL_CURRENT_mean=("AMT_DRAWINGS_TOTAL_CURRENT", "mean"),
        CC_CNT_DRAWINGS_TOTAL_CURRENT_mean=("CNT_DRAWINGS_TOTAL_CURRENT", "mean"),
        CC_SK_DPD_MAX=("SK_DPD", "max")
    )

    # Downcast memory footprint
    cc_agg = reduce_memory_usage(cc_agg, verbose=False)

    return cc_agg
