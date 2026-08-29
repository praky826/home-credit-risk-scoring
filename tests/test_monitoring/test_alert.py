import pytest
from src.monitoring.alert import check_and_alert

def test_check_and_alert_no_drift():
    psi_values = {"feat_1": 0.05, "feat_2": 0.02}
    fired = check_and_alert(psi_values, threshold=0.2)
    assert not fired

def test_check_and_alert_with_drift():
    psi_values = {"feat_1": 0.25, "feat_2": 0.02}
    fired = check_and_alert(psi_values, threshold=0.2)
    assert fired
