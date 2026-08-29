import pytest
import numpy as np
from src.evaluation.metrics import compute_all_metrics, wilcoxon_test

def test_compute_all_metrics():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 0, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.05, 0.6, 0.15, 0.3, 0.4, 0.7])
    
    metrics = compute_all_metrics(y_true, y_pred)
    assert "roc_auc" in metrics
    assert "ks_stat" in metrics
    assert "brier_score" in metrics
    assert "calibration_slope" in metrics
    
    assert metrics["roc_auc"] > 0.5
    assert 0.0 <= metrics["brier_score"] <= 1.0

def test_wilcoxon_test():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    # Model A predicts slightly better than Model B
    probs_a = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8, 0.1, 0.9])
    probs_b = np.array([0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.3, 0.7, 0.2, 0.8])
    
    res = wilcoxon_test(probs_a, probs_b, y_true)
    assert "statistic" in res
    assert "p_value" in res
    assert 0.0 <= res["p_value"] <= 1.0
