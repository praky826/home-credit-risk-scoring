import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from typing import Dict

def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Computes Kolmogorov-Smirnov statistic for binary classification.
    = max separation between CDFs of predicted probabilities for class 0 vs class 1.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    
    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return 0.0
        
    return float(stats.ks_2samp(pos_probs, neg_probs).statistic)

def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Returns dict with keys:
      roc_auc: float - sklearn roc_auc_score
      ks_stat: float - max(TPR - FPR) across all thresholds
      brier_score: float - sklearn brier_score_loss
      calibration_slope: float - slope of linear regression (y_true ~ y_prob)
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    auc = float(roc_auc_score(y_true, y_prob))
    ks = ks_statistic(y_true, y_prob)
    brier = float(brier_score_loss(y_true, y_prob))
    
    # Calibration slope
    try:
        slope, _, _, _, _ = stats.linregress(y_prob, y_true)
        if pd.isna(slope):
            slope = 1.0
    except Exception:
        slope = 1.0
        
    return {
        "roc_auc": auc,
        "ks_stat": ks,
        "brier_score": brier,
        "calibration_slope": float(slope)
    }

def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    y_true: np.ndarray
) -> Dict[str, float]:
    """
    Computes Wilcoxon signed-rank test between fold-level AUCs
    (5-fold stratified CV on test set, pairing scores by fold).
    """
    y_true = np.array(y_true)
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_a = []
    auc_b = []
    
    for _, fold_idx in skf.split(np.arange(len(y_true)), y_true):
        auc_a.append(roc_auc_score(y_true[fold_idx], scores_a[fold_idx]))
        auc_b.append(roc_auc_score(y_true[fold_idx], scores_b[fold_idx]))
        
    # Wilcoxon signed-rank test requires differences to be non-zero
    diffs = np.array(auc_a) - np.array(auc_b)
    if np.all(diffs == 0):
        return {"statistic": 0.0, "p_value": 1.0}
        
    try:
        stat, p_val = stats.wilcoxon(auc_a, auc_b, alternative="two-sided")
        return {"statistic": float(stat), "p_value": float(p_val)}
    except Exception:
        return {"statistic": 0.0, "p_value": 1.0}
