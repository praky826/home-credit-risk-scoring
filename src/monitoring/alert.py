import json
import logging
import time
import boto3
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Module-level tracking for alert cooldown (1 hour cooldown)
_last_alert_time = 0.0
COOLDOWN_SECONDS = 3600

def check_and_alert(
    psi_values: Dict[str, float],
    threshold: float = 0.2,
    sns_arn: Optional[str] = None
) -> bool:
    """
    Checks if any feature PSI exceeds threshold.
    If yes:
      - Logs WARNING with JSON payload
      - If sns_arn provided: publishes to AWS SNS via boto3 (respecting cooldown).
    Returns True if alert fired, False otherwise.
    """
    global _last_alert_time

    # Find all features exceeding threshold
    drifted_features = {col: val for col, val in psi_values.items() if val >= threshold}
    
    if not drifted_features:
        return False

    max_psi = float(max(drifted_features.values()))
    
    alert_payload = {
        "alert_type": "FEATURE_DRIFT",
        "features_above_threshold": drifted_features,
        "max_psi": max_psi,
        "timestamp": time.time()
    }
    
    # Log warning with JSON payload
    logger.warning(f"Drift Alert Triggered: {json.dumps(alert_payload)}")
    print(f"WARNING: Feature drift detected! {len(drifted_features)} features exceed threshold of {threshold}. Max PSI: {max_psi:.4f}")

    # Fire SNS alert if arn is provided and cooldown has passed
    if sns_arn:
        current_time = time.time()
        if current_time - _last_alert_time < COOLDOWN_SECONDS:
            logger.info("Drift Alert: SNS notification skipped due to active cooldown.")
            print("SNS Alert skipped due to cooldown.")
            return False
            
        try:
            print(f"Publishing drift alert to AWS SNS topic: {sns_arn}...")
            sns_client = boto3.client("sns")
            
            message_body = (
                f"Home Credit Default Risk - Feature Drift Alert\n"
                f"-------------------------------------------------\n"
                f"Alert Type: FEATURE_DRIFT\n"
                f"Threshold:  {threshold}\n"
                f"Max PSI:    {max_psi:.4f}\n"
                f"Drifted features list:\n"
            )
            for k, v in drifted_features.items():
                message_body += f"  - {k}: {v:.4f}\n"
                
            sns_client.publish(
                TopicArn=sns_arn,
                Subject="Home Credit Alert: Feature Drift Detected",
                Message=message_body
            )
            
            _last_alert_time = current_time
            print("SNS Alert published successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to publish SNS alert: {e}")
            print(f"Error publishing SNS alert: {e}")
            return False

    return True
