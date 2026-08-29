import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.calibration import calibration_curve
from typing import Dict

def plot_roc_curves(
    y_true: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    save_path: str
) -> None:
    """Plots ROC curves for all models on one plot and saves it."""
    plt.figure(figsize=(8, 6))
    
    for name, probs in model_probs.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")
        
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curves")
    plt.legend(loc="lower right")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved ROC curves to {save_path}")

def plot_pr_curves(
    y_true: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    save_path: str
) -> None:
    """Plots Precision-Recall curves for all models and saves it."""
    plt.figure(figsize=(8, 6))
    
    for name, probs in model_probs.items():
        precision, recall, _ = precision_recall_curve(y_true, probs)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.4f})")
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (PR) Curves")
    plt.legend(loc="lower left")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved PR curves to {save_path}")

def plot_calibration_curves(
    y_true: np.ndarray,
    model_probs: Dict[str, np.ndarray],
    save_path: str
) -> None:
    """Plots calibration reliability curves for all models and saves it."""
    plt.figure(figsize=(8, 6))
    
    for name, probs in model_probs.items():
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label=name)
        
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curves (Reliability Diagram)")
    plt.legend(loc="upper left")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved calibration curves to {save_path}")

def plot_feature_importances(
    importance_dict: Dict[str, float],
    title: str,
    save_path: str,
    top_n: int = 30
) -> None:
    """Plots top feature importances from a dict and saves it."""
    if not importance_dict:
        print("Feature importance dictionary is empty. Skipping plot.")
        return
        
    df = pd.DataFrame(list(importance_dict.items()), columns=["Feature", "Importance"])
    df = df.sort_values(by="Importance", ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    positions = np.arange(len(df))
    plt.barh(positions, df["Importance"].values, color="#2878b5")
    plt.yticks(positions, df["Feature"].values)
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Features by Importance - {title}")
    plt.xlabel("Importance Value")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved feature importance plot to {save_path}")
