import os
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from src.features.application import build_application_features
from src.features.bureau import build_bureau_features
from src.features.previous_app import build_previous_app_features
from src.features.pos_cash import build_pos_cash_features
from src.features.installments import build_installment_features
from src.features.credit_card import build_credit_card_features
from src.features.utils import reduce_memory_usage, MedianImputer

def generate_reference_stats(
    train_df: pd.DataFrame,
    output_path: str = "data/drift/reference_stats.json",
    bins: int = 10
) -> None:
    """
    For each numeric feature column:
    - Compute mean, std
    - Compute bin_edges = np.percentile(col.dropna(), np.linspace(0, 100, bins+1))
    Save to output_path as JSON. Skips TARGET and SK_ID_CURR.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    ref_stats = {}
    for col in train_df.columns:
        if col in ["TARGET", "SK_ID_CURR"]:
            continue
        if pd.api.types.is_numeric_dtype(train_df[col]):
            col_data = train_df[col].dropna()
            if len(col_data) > 0:
                mean_val = float(col_data.mean())
                std_val = float(col_data.std())
                # If std is 0 or NaN, set to a small eps
                if pd.isna(std_val) or std_val == 0:
                    std_val = 1e-5
                
                # Compute percentiles (0 to 100 percentiles)
                p_edges = np.percentile(col_data, np.linspace(0, 100, bins + 1))
                # Ensure edges are strictly increasing to avoid empty bins/errors
                # Add a tiny noise to duplicate edges
                p_edges = list(p_edges)
                for idx in range(1, len(p_edges)):
                    if p_edges[idx] <= p_edges[idx-1]:
                        p_edges[idx] = p_edges[idx-1] + 1e-5
                
                ref_stats[col] = {
                    "mean": mean_val,
                    "std": std_val,
                    "percentiles": [float(x) for x in p_edges]
                }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ref_stats, f, indent=2)
    print(f"Saved reference stats for {len(ref_stats)} features to {output_path}")

def build_all_features(
    config_path: str = "configs/features.yaml",
    output_dir: str = "data/processed"
) -> None:
    """
    Orchestrator. Reads config, calls all feature builders, merges on SK_ID_CURR, saves.
    """
    print(f"Loading feature configuration from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Call build_application_features() -> (train_df, test_df)
    print("Building application features...")
    train_df, test_df = build_application_features()
    print(f"Initial application train shape: {train_df.shape}, test shape: {test_df.shape}")

    # Helper to load and join auxiliary tables
    # 2. For each auxiliary table (controlled by config flags)
    auxiliary_builders = [
        ("use_bureau", build_bureau_features, {"bureau_path": "data/raw/bureau.csv", "balance_path": "data/raw/bureau_balance.csv"}),
        ("use_previous_app", build_previous_app_features, {"path": "data/raw/previous_application.csv"}),
        ("use_pos_cash", build_pos_cash_features, {"path": "data/raw/POS_CASH_balance.csv"}),
        ("use_installments", build_installment_features, {"path": "data/raw/installments_payments.csv"}),
        ("use_credit_card", build_credit_card_features, {"path": "data/raw/credit_card_balance.csv"}),
    ]

    for config_key, builder_func, kwargs in auxiliary_builders:
        if config.get(config_key, False):
            print(f"Building auxiliary features for: {config_key}...")
            try:
                # Call builder
                agg_df = builder_func(**kwargs)
                print(f" -> Aggregated shape: {agg_df.shape}")
                
                # 3. Left-join onto train_df and test_df
                train_df = train_df.join(agg_df, how="left")
                test_df = test_df.join(agg_df, how="left")
                print(f" -> Joined train shape: {train_df.shape}, test shape: {test_df.shape}")
            except Exception as e:
                print(f"Error building features for {config_key}: {e}")

    # 4. After all joins, impute remaining NaNs in numeric columns with median (fit on train, apply to test)
    # Identify numeric columns excluding TARGET
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "TARGET"]

    print("Imputing remaining missing values with training medians...")
    imputer = MedianImputer()
    imputer.fit(train_df[num_cols])
    train_df[num_cols] = imputer.transform(train_df[num_cols])
    test_df[num_cols] = imputer.transform(test_df[num_cols])

    # 5. Metadata extraction
    # Find categorical columns
    # We identify original object columns from raw application_train.csv
    try:
        raw_app = pd.read_csv("data/raw/application_train.csv", nrows=5)
        raw_obj_cols = raw_app.select_dtypes(include=["object", "str"]).columns.tolist()
    except Exception as e:
        print(f"Could not load raw application columns: {e}")
        raw_obj_cols = []

    # Map current feature names to categoricals
    feature_names = [c for c in train_df.columns if c != "TARGET"]
    
    # Categorical features are those in raw_obj_cols that are present in the final features
    cat_features = [c for c in raw_obj_cols if c in feature_names]

    cat_feature_indices = [feature_names.index(c) for c in cat_features]
    cat_cardinalities = {}
    for col in cat_features:
        # Cardinality calculation: max valid code + 1.
        # Categorical columns should already be integer-like at this point.
        col_values = pd.to_numeric(train_df[col], errors="coerce")
        if col_values.isna().any():
            raise ValueError(f"Categorical feature {col} contains NaN after cleaning.")
        if not np.allclose(col_values, np.round(col_values)):
            raise ValueError(f"Categorical feature {col} contains non-integer codes after cleaning.")
        max_idx = int(col_values.max())
        cat_cardinalities[col] = max_idx + 1

    # Save to output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Saving processed data as Parquet...")
    # Reset index to include SK_ID_CURR in the parquets
    train_save = train_df.reset_index()
    test_save = test_df.reset_index()

    # Apply memory downcasting once more on final datasets
    train_save = reduce_memory_usage(train_save, verbose=False)
    test_save = reduce_memory_usage(test_save, verbose=False)

    train_save.to_parquet(os.path.join(output_dir, "train_features.parquet"), index=False)
    test_save.to_parquet(os.path.join(output_dir, "test_features.parquet"), index=False)

    # Save metadata jsons
    with open(os.path.join(output_dir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    with open(os.path.join(output_dir, "cat_feature_indices.json"), "w", encoding="utf-8") as f:
        json.dump(cat_feature_indices, f, indent=2)
    with open(os.path.join(output_dir, "cat_cardinalities.json"), "w", encoding="utf-8") as f:
        json.dump(cat_cardinalities, f, indent=2)

    print("Generating reference stats...")
    generate_reference_stats(train_df, output_path="data/drift/reference_stats.json")

    print("\n=== Feature Pipeline Summary ===")
    print(f"Final Train Shape:   {train_df.shape}")
    print(f"Final Test Shape:    {test_df.shape}")
    print(f"Total Features:      {len(feature_names)}")
    print(f"Categorical Features: {len(cat_features)}")
    print(f"Saved artifacts to {output_dir}")

if __name__ == "__main__":
    build_all_features()
