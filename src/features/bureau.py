import os
import pandas as pd
import numpy as np
from src.features.utils import reduce_memory_usage

def build_bureau_features(
    bureau_path: str = "data/raw/bureau.csv",
    balance_path: str = "data/raw/bureau_balance.csv"
) -> pd.DataFrame:
    """
    Returns DataFrame indexed by SK_ID_CURR with aggregated bureau features.
    No TARGET column. Rows = one per unique SK_ID_CURR in bureau.csv.
    """
    # Load data
    bureau = pd.read_csv(bureau_path)
    
    # 1. Process bureau_balance.csv if exists
    if os.path.exists(balance_path):
        bureau_balance = pd.read_csv(balance_path)
        
        # Map STATUS to numeric for DPD calculation
        status_map = {
            'C': 0, '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'X': np.nan
        }
        bureau_balance['STATUS_NUM'] = bureau_balance['STATUS'].map(status_map)
        
        # Group by SK_ID_BUREAU
        bb_grp = bureau_balance.groupby('SK_ID_BUREAU').agg(
            BBAL_MONTHS_BALANCE_count=('MONTHS_BALANCE', 'count'),
            BBAL_STATUS_C_count=('STATUS', lambda x: (x == 'C').sum()),
            BBAL_STATUS_X_count=('STATUS', lambda x: (x == 'X').sum()),
            BBAL_DPD_MAX=('STATUS_NUM', 'max')
        )
        
        # Calculate rates
        bb_grp['BBAL_STATUS_C_rate'] = bb_grp['BBAL_STATUS_C_count'] / bb_grp['BBAL_MONTHS_BALANCE_count']
        bb_grp['BBAL_STATUS_X_rate'] = bb_grp['BBAL_STATUS_X_count'] / bb_grp['BBAL_MONTHS_BALANCE_count']
        bb_grp = bb_grp.drop(columns=['BBAL_STATUS_C_count', 'BBAL_STATUS_X_count'])
        
        # Merge onto bureau
        bureau = bureau.merge(bb_grp, on='SK_ID_BUREAU', how='left')
    else:
        # Create empty columns if balance doesn't exist
        bureau['BBAL_MONTHS_BALANCE_count'] = np.nan
        bureau['BBAL_STATUS_C_rate'] = np.nan
        bureau['BBAL_STATUS_X_rate'] = np.nan
        bureau['BBAL_DPD_MAX'] = np.nan

    # 2. Aggregations grouped by SK_ID_CURR
    # Count active loans
    bureau['BUREAU_ACTIVE_LOAN_COUNT'] = (bureau['CREDIT_ACTIVE'] == 'Active').astype(int)

    # Let's group
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(
        BUREAU_LOAN_COUNT=('SK_ID_BUREAU', 'count'),
        BUREAU_ACTIVE_LOAN_COUNT=('BUREAU_ACTIVE_LOAN_COUNT', 'sum'),
        BUREAU_AMT_CREDIT_SUM_mean=('AMT_CREDIT_SUM', 'mean'),
        BUREAU_AMT_CREDIT_SUM_sum=('AMT_CREDIT_SUM', 'sum'),
        BUREAU_AMT_CREDIT_SUM_max=('AMT_CREDIT_SUM', 'max'),
        BUREAU_AMT_CREDIT_DEBT_mean=('AMT_CREDIT_SUM_DEBT', 'mean'),
        BUREAU_AMT_CREDIT_DEBT_sum=('AMT_CREDIT_SUM_DEBT', 'sum'),
        BUREAU_AMT_OVERDUE_mean=('AMT_CREDIT_SUM_OVERDUE', 'mean'),
        BUREAU_DAYS_CREDIT_mean=('DAYS_CREDIT', 'mean'),
        BUREAU_DAYS_CREDIT_min=('DAYS_CREDIT', 'min'),
        BUREAU_DAYS_CREDIT_ENDDATE_mean=('DAYS_CREDIT_ENDDATE', 'mean'),
        BUREAU_CREDIT_TYPE_nunique=('CREDIT_TYPE', 'nunique'),
        # Bureau balance features aggregated to client level
        BBAL_MONTHS_BALANCE_count=('BBAL_MONTHS_BALANCE_count', 'sum'),
        BBAL_STATUS_C_rate=('BBAL_STATUS_C_rate', 'mean'),
        BBAL_STATUS_X_rate=('BBAL_STATUS_X_rate', 'mean'),
        BBAL_DPD_MAX=('BBAL_DPD_MAX', 'max')
    )

    # Downcast memory footprint
    bureau_agg = reduce_memory_usage(bureau_agg, verbose=False)

    return bureau_agg
