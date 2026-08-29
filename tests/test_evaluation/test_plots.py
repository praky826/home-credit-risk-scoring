import os
import pytest
import numpy as np
from src.evaluation.plots import (
    plot_roc_curves,
    plot_pr_curves,
    plot_calibration_curves,
    plot_feature_importances
)

def test_plots(tmp_path):
    y_true = np.array([0, 1, 0, 1])
    probs = {"model_a": np.array([0.1, 0.9, 0.2, 0.8])}
    
    # 1. ROC Curve
    roc_path = os.path.join(tmp_path, "roc.png")
    plot_roc_curves(y_true, probs, roc_path)
    assert os.path.exists(roc_path)
    
    # 2. PR Curve
    pr_path = os.path.join(tmp_path, "pr.png")
    plot_pr_curves(y_true, probs, pr_path)
    assert os.path.exists(pr_path)
    
    # 3. Calibration Curve
    cal_path = os.path.join(tmp_path, "cal.png")
    plot_calibration_curves(y_true, probs, cal_path)
    assert os.path.exists(cal_path)
    
    # 4. Feature Importance Plot
    imp_dict = {"feat_1": 0.5, "feat_2": 0.3}
    imp_path = os.path.join(tmp_path, "imp.png")
    plot_feature_importances(imp_dict, "Model A", imp_path)
    assert os.path.exists(imp_path)
