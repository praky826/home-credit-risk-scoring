import os
import yaml
import logging
import pandas as pd
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class FeatureStore:
    """
    Simulates an online feature store for credit risk scoring.
    Looks up precomputed borrower/loan features by the configured ID column
    from processed train and test Parquet tables.
    """

    def __init__(
        self,
        train_path: str = "data/processed/train_features.parquet",
        test_path: str = "data/processed/test_features.parquet",
        id_column: str = "loan_id",
        enabled: bool = True
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.id_column = id_column
        self.enabled = enabled
        self._cache: Dict[int, Tuple[Dict[str, Any], str]] = {}

    def get_customer_features(self, customer_id: int) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Looks up a customer/loan profile by the configured ID column.
        Returns (features_dict, source_name). If not found, returns (None, "").
        """
        if not self.enabled or customer_id is None:
            return None, ""

        # Check in-memory cache
        if customer_id in self._cache:
            return self._cache[customer_id]

        # 1. Search train_features
        if os.path.exists(self.train_path):
            try:
                df = pd.read_parquet(
                    self.train_path,
                    filters=[(self.id_column, "==", customer_id)]
                )
                if not df.empty:
                    row_dict = df.iloc[0].to_dict()
                    self._cache[customer_id] = (row_dict, "train_features")
                    return row_dict, "train_features"
            except Exception as e:
                logger.warning(f"Error reading train_features for {self.id_column}={customer_id}: {e}")

        # 2. Search test_features
        if os.path.exists(self.test_path):
            try:
                df = pd.read_parquet(
                    self.test_path,
                    filters=[(self.id_column, "==", customer_id)]
                )
                if not df.empty:
                    row_dict = df.iloc[0].to_dict()
                    self._cache[customer_id] = (row_dict, "test_features")
                    return row_dict, "test_features"
            except Exception as e:
                logger.warning(f"Error reading test_features for {self.id_column}={customer_id}: {e}")

        return None, ""


_feature_store_instance: Optional[FeatureStore] = None

def get_feature_store(config_path: str = "configs/serving.yaml") -> FeatureStore:
    global _feature_store_instance
    if _feature_store_instance is None:
        fs_config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
                fs_config = cfg.get("feature_store", {})

        enabled = fs_config.get("enabled", True)
        train_path = fs_config.get("train_features_path", "data/processed/train_features.parquet")
        test_path = fs_config.get("test_features_path", "data/processed/test_features.parquet")
        id_column = fs_config.get("id_column", "loan_id")

        _feature_store_instance = FeatureStore(
            train_path=train_path,
            test_path=test_path,
            id_column=id_column,
            enabled=enabled
        )
    return _feature_store_instance
