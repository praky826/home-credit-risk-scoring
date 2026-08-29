import os
import json
import pandas as pd
import numpy as np
import mlflow
from typing import Dict, Any
from mlflow.tracking import MlflowClient

from src.evaluation.metrics import compute_all_metrics, wilcoxon_test
from src.evaluation.plots import (
    plot_calibration_curves,
    plot_feature_importances,
    plot_pr_curves,
    plot_roc_curves,
)
from src.models.train import load_data


def _model_artifact_path(models_dir: str, model_type: str, run_id: str) -> str:
    ext = ".pt" if model_type in ["mlp", "entity_embedding", "tab_transformer"] else ".pkl"
    return os.path.join(models_dir, f"model_{model_type}_{run_id}{ext}")


def _is_plausible_full_data_run(run, models_dir: str, model_type: str) -> bool:
    metrics = run.data.metrics
    val_auc = metrics.get("val_roc_auc")
    test_auc = metrics.get("test_roc_auc")
    if val_auc is None or test_auc is None:
        return False
    if not (0.5 < val_auc < 0.99 and 0.5 < test_auc < 0.99):
        return False
    return os.path.exists(_model_artifact_path(models_dir, model_type, run.info.run_id))

def run_comparison(
    models_dir: str = "mlflow_artifacts",
    splits_dir: str = "data/splits",
    features_path: str = "data/processed/train_features.parquet"
) -> Dict[str, Dict[str, float]]:
    """
    Loads best MLflow run for each of the 5 model types.
    Computes all metrics on the test split.
    Returns nested dict: {model_name: {metric_name: value}}.
    Also prints a formatted markdown table to stdout.
    Also saves comparison_table.csv to mlflow_artifacts/.
    """
    mlflow.set_tracking_uri("mlruns")
    client = MlflowClient()

    # Load splits
    _, _, _, _, X_test, y_test = load_data(features_path=features_path, splits_dir=splits_dir)

    model_types = ["xgboost", "lgbm", "mlp", "entity_embedding", "tab_transformer"]
    best_runs = {}

    print("\nQuerying MLflow for best runs...")
    try:
        experiments = client.search_experiments()
        exp_ids = [e.experiment_id for e in experiments]
        
        for m_type in model_types:
            # Search runs for this model type, then filter out stale/sample or
            # implausible runs before choosing the strongest validation result.
            runs = client.search_runs(
                experiment_ids=exp_ids,
                filter_string=f"tags.model_type = '{m_type}'",
                order_by=["metrics.val_roc_auc DESC"],
                max_results=50
            )
            valid_runs = [r for r in runs if _is_plausible_full_data_run(r, models_dir, m_type)]
            if valid_runs:
                best_runs[m_type] = valid_runs[0]
                print(
                    f"  Found best run for {m_type}: "
                    f"run_id={valid_runs[0].info.run_id}, "
                    f"Val AUC={valid_runs[0].data.metrics.get('val_roc_auc'):.5f}, "
                    f"Test AUC={valid_runs[0].data.metrics.get('test_roc_auc'):.5f}"
                )
            else:
                print(f"  No runs found for {m_type}")
    except Exception as e:
        print(f"Error querying MLflow: {e}")

    results = {}
    predictions = {}

    for m_type, run in best_runs.items():
        run_id = run.info.run_id
        model_path = _model_artifact_path(models_dir, m_type, run_id)

        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}. Skipping.")
            continue

        print(f"Loading {m_type} model from {model_path}...")
        try:
            if m_type == "xgboost":
                from src.models.xgboost_model import XGBoostModel
                model = XGBoostModel.load(model_path)
            elif m_type == "lgbm":
                from src.models.lgbm_model import LGBMModel
                model = LGBMModel.load(model_path)
            elif m_type == "mlp":
                from src.models.mlp import MLPModel
                model = MLPModel.load(model_path)
            elif m_type == "entity_embedding":
                from src.models.entity_embedding import EntityEmbeddingModel
                model = EntityEmbeddingModel.load(model_path)
            elif m_type == "tab_transformer":
                from src.models.tab_transformer import TabTransformerModel
                model = TabTransformerModel.load(model_path)
            else:
                continue

            # Predict probabilities
            probs = model.predict_proba(X_test)
            predictions[m_type] = probs

            # Calculate metrics
            metrics = compute_all_metrics(y_test.values, probs)
            results[m_type] = metrics

            if m_type in ["xgboost", "lgbm"] and hasattr(model.model, "feature_importances_"):
                importance = dict(zip(X_test.columns, model.model.feature_importances_))
                plot_feature_importances(
                    importance,
                    title=m_type,
                    save_path=os.path.join("reports", f"{m_type}_feature_importance.png")
                )
        except Exception as e:
            print(f"Error evaluating model {m_type}: {e}")

    # Compute statistical significance tests
    print("\nConducting pairwise Wilcoxon significance tests (p-value threshold = 0.05):")
    comparison_pairs = [
        ("lgbm", "entity_embedding"),
        ("xgboost", "lgbm"),
        ("mlp", "entity_embedding")
    ]
    for m_a, m_b in comparison_pairs:
        if m_a in predictions and m_b in predictions:
            test_res = wilcoxon_test(predictions[m_a], predictions[m_b], y_test.values)
            p_val = test_res["p_value"]
            sig = "Statistically Significant" if p_val < 0.05 else "Not Statistically Significant"
            print(f"  {m_a} vs {m_b}: p-value = {p_val:.5f} ({sig})")
        else:
            print(f"  {m_a} vs {m_b}: One or both models missing for significance test.")

    if results:
        print_comparison_table(results)

        plot_roc_curves(y_test.values, predictions, os.path.join("reports", "roc_curves.png"))
        plot_pr_curves(y_test.values, predictions, os.path.join("reports", "pr_curves.png"))
        plot_calibration_curves(
            y_test.values,
            predictions,
            os.path.join("reports", "calibration_curves.png")
        )
        
        # Save to csv
        os.makedirs(models_dir, exist_ok=True)
        csv_path = os.path.join(models_dir, "comparison_table.csv")
        df_res = pd.DataFrame(results).T
        df_res.index.name = "Model"
        df_res.to_csv(csv_path)
        print(f"Saved comparison table to {csv_path}")
    else:
        print("No models were successfully evaluated.")

    return results

def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """
    Prints markdown-formatted table:
    | Model | ROC-AUC | KS Stat | Brier Score | Calibration Slope |
    """
    print("\n=== Model Comparison Table ===")
    print("| Model | ROC-AUC | KS Stat | Brier Score | Calibration Slope |")
    print("|---|---|---|---|---|")
    for name, metrics in results.items():
        print(f"| {name} | {metrics['roc_auc']:.5f} | {metrics['ks_stat']:.5f} | {metrics['brier_score']:.5f} | {metrics['calibration_slope']:.5f} |")
    print("==============================\n")

if __name__ == "__main__":
    run_comparison()
