import pandas as pd
import numpy as np
from src.features.utils import reduce_memory_usage, safe_divide

def build_installment_features(
    path: str = "data/raw/installments_payments.csv"
) -> pd.DataFrame:
    """
    Returns DataFrame indexed by SK_ID_CURR.
    """
    df = pd.read_csv(path)

    # Derived columns (computed before grouping)
    df["PAYMENT_DIFF"] = df["AMT_INSTALMENT"] - df["AMT_PAYMENT"]
    
    # Raw diff for past due & before due calculations
    raw_delay = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    df["DAYS_PAST_DUE"] = raw_delay.clip(lower=0)
    df["DAYS_BEFORE_DUE"] = (-raw_delay).clip(lower=0)

    # Indicator for late payment (raw delay > 0)
    df["IS_LATE_PAYMENT"] = (raw_delay > 0).astype(int)

    inst_agg = df.groupby("SK_ID_CURR").agg(
        INS_COUNT=("SK_ID_PREV", "count"),
        INS_PAYMENT_DIFF_mean=("PAYMENT_DIFF", "mean"),
        INS_PAYMENT_DIFF_max=("PAYMENT_DIFF", "max"),
        INS_DAYS_PAST_DUE_mean=("DAYS_PAST_DUE", "mean"),
        INS_DAYS_PAST_DUE_max=("DAYS_PAST_DUE", "max"),
        INS_DAYS_BEFORE_DUE_mean=("DAYS_BEFORE_DUE", "mean"),
        INS_LATE_PAYMENT_RATE=("IS_LATE_PAYMENT", "mean"),
        INS_AMT_PAYMENT_mean=("AMT_PAYMENT", "mean"),
        INS_AMT_PAYMENT_sum=("AMT_PAYMENT", "sum")
    )

    # Downcast memory footprint
    inst_agg = reduce_memory_usage(inst_agg, verbose=False)

    return inst_agg
