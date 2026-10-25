import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

def compute_psi(
    reference: Union[np.ndarray, List[float]],
    current: np.ndarray,
    bins: int = 10,
    eps: float = 1e-6
) -> float:
    """
    Population Stability Index.
    reference: 1D array of reference distribution values OR list of pre-computed bin edges (length bins+1)
    current:   1D array of current distribution values
    bins: number of equal-frequency bins.
    eps: small constant added to all bucket proportions before log (avoids log(0)).
    """
    current = np.array(current)
    if len(current) == 0:
        return 0.0

    # Determine edges
    if isinstance(reference, list) or (isinstance(reference, np.ndarray) and len(reference) == bins + 1):
        edges = np.array(reference)
    else:
        reference = np.array(reference)
        edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
        # Ensure edges are strictly increasing
        for idx in range(1, len(edges)):
            if edges[idx] <= edges[idx-1]:
                edges[idx] = edges[idx-1] + 1e-5

    # Clip current to edges min/max
    current_clipped = np.clip(current, edges[0], edges[-1])
    
    # Compute proportions
    counts_cur, _ = np.histogram(current_clipped, bins=edges)
    p_cur = counts_cur / len(current)
    
    # Proportions for reference
    if isinstance(reference, list) or (isinstance(reference, np.ndarray) and len(reference) == bins + 1):
        p_ref = np.ones(bins) / bins
    else:
        ref_clipped = np.clip(reference, edges[0], edges[-1])
        counts_ref, _ = np.histogram(ref_clipped, bins=edges)
        p_ref = counts_ref / len(reference)
        
    # Calculate PSI
    psi_value = np.sum((p_ref - p_cur) * np.log((p_ref + eps) / (p_cur + eps)))
    return float(psi_value)

def compute_all_psi(
    reference_stats: Dict[str, Any],
    current_df: pd.DataFrame,
    bins: int = 10
) -> Dict[str, float]:
    """
    Computes PSI for every numeric feature in reference_stats.
    Skips features not present in current_df.
    Returns {feature_name: psi_value}.
    """
    psi_values = {}
    for col, stats in reference_stats.items():
        if col in current_df.columns:
            # Skip non-numeric or targets if present
            if col in ["TARGET", "SK_ID_CURR", "loan_id"]:
                continue
            current_vals = current_df[col].dropna().values
            if len(current_vals) > 0 and "percentiles" in stats:
                edges = stats["percentiles"]
                # Must be length bins+1
                if len(edges) == bins + 1:
                    psi_val = compute_psi(edges, current_vals, bins=bins)
                    psi_values[col] = psi_val
    return psi_values
