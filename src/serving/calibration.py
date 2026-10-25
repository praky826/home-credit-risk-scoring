import numpy as np

def unweight_probability(weighted_prob: float, scale_pos_weight: float = 11.38) -> float:
    """
    Inverts the probability shift induced by training with `scale_pos_weight = w`.
    
    When training with class weight w > 1, the raw odds are multiplied by w:
        odds_raw = p_raw / (1 - p_raw) = w * odds_true
    Therefore:
        odds_true = odds_raw / w
        p_true = odds_true / (1 + odds_true)
    """
    if weighted_prob <= 0.0:
        return 0.0
    if weighted_prob >= 1.0:
        return 1.0
    if scale_pos_weight <= 0.0:
        return float(weighted_prob)

    # Safe logit / odds calculation
    # Clamp weighted_prob to avoid division by zero
    p_clamped = np.clip(weighted_prob, 1e-12, 1.0 - 1e-12)
    odds = p_clamped / (1.0 - p_clamped)
    unweighted_odds = odds / scale_pos_weight
    calibrated = unweighted_odds / (1.0 + unweighted_odds)
    
    return float(np.clip(calibrated, 0.0, 1.0))


def calibrate_probability(
    raw_prob: float,
    method: str = "scale_pos_weight_unbias",
    scale_pos_weight: float = 11.38
) -> float:
    """
    Master entrypoint for probability calibration.
    Supports scale_pos_weight_unbias, identity, and future platt/isotonic methods.
    """
    if method == "scale_pos_weight_unbias":
        return unweight_probability(raw_prob, scale_pos_weight=scale_pos_weight)
    elif method == "identity" or not method:
        return float(np.clip(raw_prob, 0.0, 1.0))
    else:
        # Fallback to unweighting if method is unknown
        return unweight_probability(raw_prob, scale_pos_weight=scale_pos_weight)
