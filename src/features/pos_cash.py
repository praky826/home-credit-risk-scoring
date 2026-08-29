import pandas as pd
import numpy as np
from src.features.utils import reduce_memory_usage

def build_pos_cash_features(
    path: str = "data/raw/POS_CASH_balance.csv"
) -> pd.DataFrame:
    """
    Returns DataFrame indexed by SK_ID_CURR.
    """
    df = pd.read_csv(path)

    # Completed count indicator
    df["POS_COMPLETED"] = (df["NAME_CONTRACT_STATUS"] == "Completed").astype(int)

    pos_agg = df.groupby("SK_ID_CURR").agg(
        POS_MONTHS_BALANCE_count=("MONTHS_BALANCE", "count"),
        POS_SK_DPD_MAX=("SK_DPD", "max"),
        POS_SK_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
        POS_CNT_INSTALMENT_FUTURE_mean=("CNT_INSTALMENT_FUTURE", "mean"),
        POS_COMPLETED_COUNT=("POS_COMPLETED", "sum")
    )

    # Downcast memory footprint
    pos_agg = reduce_memory_usage(pos_agg, verbose=False)

    return pos_agg
