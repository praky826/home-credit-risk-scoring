import argparse
import mlflow
import yaml
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from src.evaluation.metrics import compute_all_metrics

def load_data(
    features_path: str = "data/processed/train_features.parquet",
    splits_dir: str = "data/splits"
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Loads train_features.parquet.
    Loads (or creates) index arrays from splits_dir.
    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    See EXPERIMENT_DESIGN.md for split strategy.
    """
    df = pd.read_parquet(features_path)
    # Ensure sorted by SK_ID_CURR ascending
    df = df.sort_values("SK_ID_CURR").reset_index(drop=True)

    train_idx_path = os.path.join(splits_dir, "train_idx.npy")
    val_idx_path = os.path.join(splits_dir, "val_idx.npy")
    test_idx_path = os.path.join(splits_dir, "test_idx.npy")

    if os.path.exists(train_idx_path) and os.path.exists(val_idx_path) and os.path.exists(test_idx_path):
        print(f"Loading existing split indices from {splits_dir}...")
        train_idx = np.load(train_idx_path)
        val_idx = np.load(val_idx_path)
        test_idx = np.load(test_idx_path)
    else:
        print(f"Creating new time-based split indices in {splits_dir}...")
        os.makedirs(splits_dir, exist_ok=True)
        sk_ids = df["SK_ID_CURR"].values
        n = len(sk_ids)
        
        n_train = int(0.80 * n)
        n_val = int(0.10 * n)
        
        train_idx = sk_ids[:n_train]
        val_idx = sk_ids[n_train:n_train + n_val]
        test_idx = sk_ids[n_train + n_val:]
        
        np.save(train_idx_path, train_idx)
        np.save(val_idx_path, val_idx)
        np.save(test_idx_path, test_idx)

    # Split dataset based on SK_ID_CURR indices
    train_df = df[df["SK_ID_CURR"].isin(train_idx)].set_index("SK_ID_CURR")
    val_df = df[df["SK_ID_CURR"].isin(val_idx)].set_index("SK_ID_CURR")
    test_df = df[df["SK_ID_CURR"].isin(test_idx)].set_index("SK_ID_CURR")

    y_train = train_df["TARGET"]
    X_train = train_df.drop(columns=["TARGET"])
    
    y_val = val_df["TARGET"]
    X_val = val_df.drop(columns=["TARGET"])
    
    y_test = test_df["TARGET"]
    X_test = test_df.drop(columns=["TARGET"])

    print(f"Dataset splits loaded successfully:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}, positive_rate={y_train.mean()*100:.2f}%")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}, positive_rate={y_val.mean()*100:.2f}%")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}, positive_rate={y_test.mean()*100:.2f}%")

    return X_train, y_train, X_val, y_val, X_test, y_test

def train_and_log(
    model_type: str,            # one of: xgboost, lgbm, mlp, entity_embedding, tab_transformer
    config_path: str | None = None,  # if None, uses configs/{model_type}.yaml
    tune: bool = False,
    experiment_name: str | None = None  # if None, inferred from model_type
) -> str:
    """
    Main training entry point.
    """
    if config_path is None:
        config_path = f"configs/{model_type}.yaml"

    print(f"Loading configuration from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Set up MLflow Tracking
    mlflow.set_tracking_uri("mlruns")
    if experiment_name is None:
        if model_type in ["xgboost", "lgbm"]:
            experiment_name = "home_credit_trees"
        else:
            experiment_name = "home_credit_nn"
    
    mlflow.set_experiment(experiment_name)

    # Load splits
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Instantiate model wrapper
    if model_type == "xgboost":
        from src.models.xgboost_model import XGBoostModel
        model = XGBoostModel(config)
    elif model_type == "lgbm":
        from src.models.lgbm_model import LGBMModel
        model = LGBMModel(config)
    elif model_type == "mlp":
        from src.models.mlp import MLPModel
        model = MLPModel(config)
    elif model_type == "entity_embedding":
        from src.models.entity_embedding import EntityEmbeddingModel
        model = EntityEmbeddingModel(config)
    elif model_type == "tab_transformer":
        from src.models.tab_transformer import TabTransformerModel
        model = TabTransformerModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train model
    print(f"Fitting {model_type} model (tuning={tune})...")
    # For quick testing, we can limit trials in tuning if needed (defaults to 50)
    n_trials = 50
    # If we are in a testing or mock environment, let's limit tuning trials to 5 to run quickly
    # (Checking if dataset size is small as a proxy for mock environment)
    if len(X_train) < 1000:
        n_trials = 3
        print(f"Small dataset detected. Limiting tuning to {n_trials} trials.")

    model.fit(X_train, y_train, X_val, y_val, tune=tune, n_trials=n_trials)

    # Predict probas
    val_probs = model.predict_proba(X_val)
    test_probs = model.predict_proba(X_test)

    # Compute metrics
    val_metrics = compute_all_metrics(y_val, val_probs)
    test_metrics = compute_all_metrics(y_test, test_probs)

    print("\nValidation Metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.5f}")

    print("\nTest Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.5f}")

    # Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Logging model run to MLflow (run_id: {run_id})...")
        
        # Log config params
        if "params" in config:
            mlflow.log_params(config["params"])
        mlflow.log_param("tune", tune)
        
        # Log metrics (prefix to distinguish validation vs test)
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # Save model locally and log artifact
        os.makedirs("mlflow_artifacts", exist_ok=True)
        ext = ".pt" if model_type in ["mlp", "entity_embedding", "tab_transformer"] else ".pkl"
        model_filename = f"model_{model_type}_{run_id}{ext}"
        model_save_path = os.path.join("mlflow_artifacts", model_filename)
        model.save(model_save_path)
        
        mlflow.log_artifact(model_save_path)
        mlflow.set_tag("model_type", model_type)
        
        # Also save metadata features list
        feature_names = X_train.columns.tolist()
        feat_list_path = os.path.join("mlflow_artifacts", f"features_{run_id}.json")
        with open(feat_list_path, "w") as f:
            json.dump(feature_names, f, indent=2)
        mlflow.log_artifact(feat_list_path)

        print("MLflow logging complete.")
        return run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
        choices=["xgboost","lgbm","mlp","entity_embedding","tab_transformer"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    train_and_log(args.model, args.config, args.tune)
