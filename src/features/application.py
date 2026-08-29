import pandas as pd
import numpy as np
from typing import Tuple
from src.features.utils import reduce_memory_usage, SafeLabelEncoder, MedianImputer, safe_divide

def build_application_features(
    train_path: str = "data/raw/application_train.csv",
    test_path: str = "data/raw/application_test.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads application_train.csv and application_test.csv.
    Returns (train_df, test_df) each with:
      - SK_ID_CURR (index, not a feature column)
      - TARGET (train only)
      - All engineered application features (see DATA_DICTIONARY.md)
    """
    # Load raw csvs
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 1. Drop columns with > 60% missingness (computed on train, applied to both).
    # Ignore SK_ID_CURR and TARGET for missingness threshold calculation
    check_cols = [c for c in train_df.columns if c not in ["SK_ID_CURR", "TARGET"]]
    missing_pct = train_df[check_cols].isnull().mean()
    cols_to_drop = missing_pct[missing_pct > 0.60].index.tolist()
    
    train_df = train_df.drop(columns=cols_to_drop)
    test_df = test_df.drop(columns=cols_to_drop)

    # Helper to apply features on both
    for df in [train_df, test_df]:
        # 2. Create CREDIT_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL
        df["CREDIT_INCOME_RATIO"] = safe_divide(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])

        # 3. Create ANNUITY_INCOME_RATIO = AMT_ANNUITY / AMT_INCOME_TOTAL
        df["ANNUITY_INCOME_RATIO"] = safe_divide(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])

        # 4. Create CREDIT_TERM = AMT_CREDIT / AMT_ANNUITY (months to repay)
        df["CREDIT_TERM"] = safe_divide(df["AMT_CREDIT"], df["AMT_ANNUITY"])

        # 5. Create DAYS_EMPLOYED_PERCENT = DAYS_EMPLOYED / DAYS_BIRTH
        df["DAYS_EMPLOYED_PERCENT"] = safe_divide(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])

        # 6. Flag EMPLOYED_ANOM = (DAYS_EMPLOYED == 365243).astype(int); set DAYS_EMPLOYED=NaN where anomalous
        df["EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
        df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
        # Recompute DAYS_EMPLOYED_PERCENT for anomalies to be NaN (since employment is unknown/anomalous)
        df.loc[df["EMPLOYED_ANOM"] == 1, "DAYS_EMPLOYED_PERCENT"] = np.nan

    # 7. Binary encode binary object columns (Y/N -> 1/0, M/F -> 1/0).
    # Keep CODE_GENDER separate so XNA can be handled explicitly.
    obj_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
    binary_cols = []
    if "CODE_GENDER" in obj_cols:
        gender_map = {"F": 0, "M": 1, "XNA": np.nan}
        train_df["CODE_GENDER"] = train_df["CODE_GENDER"].map(gender_map)
        test_df["CODE_GENDER"] = test_df["CODE_GENDER"].map(gender_map)
        gender_fill = train_df["CODE_GENDER"].dropna().mode()
        if gender_fill.empty:
            gender_fill = 0
        else:
            gender_fill = int(gender_fill.iloc[0])
        train_df["CODE_GENDER"] = train_df["CODE_GENDER"].fillna(gender_fill).astype(int)
        test_df["CODE_GENDER"] = test_df["CODE_GENDER"].fillna(gender_fill).astype(int)
        binary_cols.append("CODE_GENDER")

    for col in obj_cols:
        if col == "CODE_GENDER":
            continue
        unique_vals = train_df[col].dropna().unique()
        if len(unique_vals) <= 2:
            binary_cols.append(col)

    for col in binary_cols:
        # Fit on train mapping
        unique_vals = sorted(train_df[col].dropna().unique())
        val_map = {val: idx for idx, val in enumerate(unique_vals)}
        train_mapped = train_df[col].map(val_map)
        test_mapped = test_df[col].map(val_map)

        # Fill missing/unknown values with the most frequent training class so
        # binary columns remain valid integer codes.
        mode_series = train_mapped.dropna().mode()
        if mode_series.empty:
            fill_val = 0
        else:
            fill_val = int(mode_series.iloc[0])

        train_df[col] = train_mapped.fillna(fill_val).astype(int)
        test_df[col] = test_mapped.fillna(fill_val).astype(int)

    # 8. Label-encode remaining object columns (fit on train, transform both).
    remaining_obj_cols = [c for c in obj_cols if c not in binary_cols]
    if remaining_obj_cols:
        encoder = SafeLabelEncoder()
        encoder.fit(train_df[remaining_obj_cols])
        train_df[remaining_obj_cols] = encoder.transform(train_df[remaining_obj_cols])
        test_df[remaining_obj_cols] = encoder.transform(test_df[remaining_obj_cols])

    # 9. Impute medians on continuous numeric columns only.
    # Categorical codes are already handled above and should stay integer-like.
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_code_cols = binary_cols + remaining_obj_cols
    num_cols = [c for c in num_cols if c not in ["SK_ID_CURR", "TARGET"] + cat_code_cols]

    imputer = MedianImputer()
    imputer.fit(train_df[num_cols])
    train_df[num_cols] = imputer.transform(train_df[num_cols])
    test_df[num_cols] = imputer.transform(test_df[num_cols])

    # Sets SK_ID_CURR as index
    train_df = train_df.set_index("SK_ID_CURR")
    test_df = test_df.set_index("SK_ID_CURR")

    # Downcast memory footprint
    train_df = reduce_memory_usage(train_df, verbose=False)
    test_df = reduce_memory_usage(test_df, verbose=False)

    return train_df, test_df
