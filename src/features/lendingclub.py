from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.features.utils import MedianImputer, SafeLabelEncoder, reduce_memory_usage, safe_divide


GOOD_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}

BAD_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}

LEAKAGE_COLUMNS = {
    "loan_status",
    "funded_amnt",
    "funded_amnt_inv",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_credit_pull_d",
    "last_fico_range_high",
    "last_fico_range_low",
    "out_prncp",
    "out_prncp_inv",
    "next_pymnt_d",
    "debt_settlement_flag",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
}

BASE_FEATURES = [
    "loan_amnt",
    "term",
    "installment",
    "annual_inc",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "initial_list_status",
    "application_type",
    "mort_acc",
    "pub_rec_bankruptcies",
    "earliest_cr_line",
    "issue_d",
]

OPTIONAL_UNDERWRITING_FEATURES = ["grade", "sub_grade", "int_rate"]


def parse_term(value) -> float:
    """Parse LendingClub term strings like ' 36 months' into month counts."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"\d+", str(value))
    return float(match.group(0)) if match else np.nan


def parse_emp_length(value) -> float:
    """Parse employment length strings into numeric years on a 0-10 scale."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"", "n/a", "nan", "none"}:
        return np.nan
    if text.startswith("<"):
        return 0.0
    if text.startswith("10+"):
        return 10.0
    match = re.search(r"\d+", text)
    return float(match.group(0)) if match else np.nan


def parse_percent(value) -> float:
    """Parse percentage fields that may arrive as strings with percent signs."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip().replace("%", "")
    return pd.to_numeric(value, errors="coerce")


def parse_lendingclub_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%b-%Y", errors="coerce")


def _find_id_column(df: pd.DataFrame) -> str | None:
    for candidate in ["id", "loan_id", "member_id"]:
        if candidate in df.columns:
            return candidate
    return None


def _make_loan_ids(df: pd.DataFrame, id_column: str | None) -> pd.Series:
    if id_column is None:
        return pd.Series(np.arange(1, len(df) + 1), index=df.index, name="loan_id")

    ids = pd.to_numeric(df[id_column], errors="coerce")
    if ids.isna().any() or ids.duplicated().any():
        return pd.Series(np.arange(1, len(df) + 1), index=df.index, name="loan_id")
    return ids.astype("int64").rename("loan_id")


def _select_available_features(
    df: pd.DataFrame,
    include_grade_features: bool,
    include_interest_rate: bool,
) -> list[str]:
    requested = list(BASE_FEATURES)
    if include_grade_features:
        requested.extend(["grade", "sub_grade"])
    if include_interest_rate:
        requested.append("int_rate")
    return [col for col in requested if col in df.columns]


def build_lendingclub_features(
    raw_path: str = "data/raw/lendingclub/accepted_2007_to_2018Q4.csv",
    include_grade_features: bool = False,
    include_interest_rate: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a one-table LendingClub default-risk feature matrix.

    Returns train_df and test_df with loan_id as the index. The test frame is a
    holdout copy without TARGET so the serving feature-store behavior remains
    compatible with the existing project structure.
    """
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(
            f"LendingClub raw file not found at {raw_path}. Download the accepted loans CSV "
            "and update configs/features.yaml if the filename differs."
        )

    raw = pd.read_csv(path, low_memory=False)
    if "loan_status" not in raw.columns:
        raise ValueError("LendingClub raw data must contain a loan_status column.")

    raw = raw[raw["loan_status"].isin(GOOD_STATUSES | BAD_STATUSES)].copy()
    if raw.empty:
        raise ValueError("No rows matched final LendingClub loan_status values.")

    raw["TARGET"] = raw["loan_status"].isin(BAD_STATUSES).astype(int)
    raw["loan_id"] = _make_loan_ids(raw, _find_id_column(raw))

    feature_cols = _select_available_features(
        raw,
        include_grade_features=include_grade_features,
        include_interest_rate=include_interest_rate,
    )
    if not feature_cols:
        raise ValueError("No supported LendingClub feature columns were found in the raw file.")

    df = raw[["loan_id", "TARGET"] + feature_cols].copy()

    if "term" in df.columns:
        df["term"] = df["term"].map(parse_term)
    if "emp_length" in df.columns:
        df["emp_length"] = df["emp_length"].map(parse_emp_length)
    if "revol_util" in df.columns:
        df["revol_util"] = df["revol_util"].map(parse_percent)
    if "int_rate" in df.columns:
        df["int_rate"] = df["int_rate"].map(parse_percent)

    issue_dt = parse_lendingclub_date(df["issue_d"]) if "issue_d" in df.columns else pd.Series(pd.NaT, index=df.index)
    earliest_dt = (
        parse_lendingclub_date(df["earliest_cr_line"])
        if "earliest_cr_line" in df.columns
        else pd.Series(pd.NaT, index=df.index)
    )

    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2.0
    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        df["loan_income_ratio"] = safe_divide(df["loan_amnt"], df["annual_inc"])
    if "installment" in df.columns and "annual_inc" in df.columns:
        df["installment_income_ratio"] = safe_divide(df["installment"] * 12.0, df["annual_inc"])
    if "revol_bal" in df.columns and "annual_inc" in df.columns:
        df["revol_balance_income_ratio"] = safe_divide(df["revol_bal"], df["annual_inc"])

    credit_history_days = (issue_dt - earliest_dt).dt.days
    df["credit_history_years"] = credit_history_days / 365.25
    df["issue_year"] = issue_dt.dt.year
    df["issue_month"] = issue_dt.dt.month

    # Keep issue_d only long enough for temporal sorting/splitting; models train on numeric issue_year/month.
    if "issue_d" in df.columns:
        df["issue_d"] = issue_dt
    if "earliest_cr_line" in df.columns:
        df = df.drop(columns=["earliest_cr_line"])

    for col in LEAKAGE_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in ["TARGET"]]
    category_mappings = {}
    if categorical_cols:
        encoder = SafeLabelEncoder()
        encoder.fit(df[categorical_cols])
        category_mappings = encoder.mapping
        df[categorical_cols] = encoder.transform(df[categorical_cols])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ["loan_id", "TARGET"]]
    imputer = MedianImputer()
    imputer.fit(df[numeric_cols])
    df[numeric_cols] = imputer.transform(df[numeric_cols])

    sort_cols = ["issue_d", "loan_id"] if "issue_d" in df.columns else ["loan_id"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    if "issue_d" in df.columns:
        df = df.drop(columns=["issue_d"])

    split_at = int(len(df) * 0.90)
    train_df = df.iloc[:split_at].copy().set_index("loan_id")
    test_df = df.iloc[split_at:].copy().drop(columns=["TARGET"]).set_index("loan_id")

    train_df = reduce_memory_usage(train_df, verbose=False)
    test_df = reduce_memory_usage(test_df, verbose=False)
    train_df.attrs["categorical_features"] = categorical_cols
    train_df.attrs["category_mappings"] = category_mappings

    return train_df, test_df
