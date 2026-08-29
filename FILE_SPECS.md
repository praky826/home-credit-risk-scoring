# FILE_SPECS.md
## Home Credit Default Risk — Exact File Specifications

Every non-trivial file: exact signatures, types, and behavior.
Precise enough to hand to a code generator with zero clarifying questions.

---

## `src/features/utils.py`

```python
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcasts numeric columns to smallest fitting dtype.
    float64 → float32, int64 → int32/int16/int8 based on value range.
    Returns modified df in place (also returns it).
    Does NOT touch object/category columns.
    """

def label_encode_categoricals(
    df: pd.DataFrame,
    cat_cols: List[str],
    fit: bool = True,
    encoders: Dict[str, dict] | None = None
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """
    Applies integer label encoding to cat_cols.
    If fit=True: builds mapping {col: {value: int_code}} from df, unseen → -1.
    If fit=False: uses provided encoders dict; values not in mapping → -1.
    Returns (df_encoded, encoders).
    Modifies df in place and also returns it.
    NaN in cat column → encoded as -1 before any other mapping.
    """

def one_hot_encode(
    df: pd.DataFrame,
    cat_cols: List[str],
    drop_first: bool = False
) -> pd.DataFrame:
    """
    pd.get_dummies wrapper. Adds prefix=col_name, prefix_sep='_'.
    Returns new df with cat_cols replaced by dummy columns.
    """

def impute_medians(
    df: pd.DataFrame,
    cols: List[str],
    medians: Dict[str, float] | None = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Fills NaN in cols with median value.
    If fit=True: computes medians from df.
    If fit=False: uses provided medians dict.
    Returns (df, medians_dict).
    """

def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    fill_value: float = 0.0
) -> pd.Series:
    """
    Returns numerator / denominator, replacing inf and NaN with fill_value.
    """
```

---

## `src/features/application.py`

```python
import pandas as pd
from typing import Tuple

def build_application_features(
    train_path: str = "data/raw/application_train.csv",
    test_path: str = "data/raw/application_test.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads application_train.csv and application_test.csv.
    Returns (train_df, test_df) each with:
      - SK_ID_CURR (index, not a feature column)
      - TARGET (train only)
      - All engineered application features (see DATA_DICTIONARY.md §Application)
    
    Engineering steps (in order):
    1. Drop columns with > 60% missingness (computed on train, applied to both).
    2. Create CREDIT_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL
    3. Create ANNUITY_INCOME_RATIO = AMT_ANNUITY / AMT_INCOME_TOTAL
    4. Create CREDIT_TERM = AMT_CREDIT / AMT_ANNUITY  (months to repay)
    5. Create DAYS_EMPLOYED_PERCENT = DAYS_EMPLOYED / DAYS_BIRTH
    6. Flag EMPLOYED_ANOM = (DAYS_EMPLOYED == 365243).astype(int); set DAYS_EMPLOYED=NaN where anomalous
    7. Binary encode binary object columns (Y/N → 1/0, M/F → 1/0).
    8. Label-encode remaining object columns (fit on train, transform both).
    9. impute_medians on all numeric columns (fit on train, transform both).
    
    Does NOT one-hot encode — callers decide encoding strategy.
    Sets SK_ID_CURR as index on both returned DataFrames.
    """
```

---

## `src/features/bureau.py`

```python
import pandas as pd

def build_bureau_features(
    bureau_path: str = "data/raw/bureau.csv",
    balance_path: str = "data/raw/bureau_balance.csv"
) -> pd.DataFrame:
    """
    Returns DataFrame indexed by SK_ID_CURR with aggregated bureau features.
    No TARGET column. Rows = one per unique SK_ID_CURR in bureau.csv.
    
    bureau.csv aggregations (grouped by SK_ID_CURR):
      - BUREAU_LOAN_COUNT: count of rows
      - BUREAU_ACTIVE_LOAN_COUNT: count where CREDIT_ACTIVE == 'Active'
      - BUREAU_AMT_CREDIT_SUM_mean, _sum, _max
      - BUREAU_AMT_CREDIT_DEBT_mean, _sum
      - BUREAU_AMT_OVERDUE_mean (from AMT_CREDIT_SUM_OVERDUE)
      - BUREAU_DAYS_CREDIT_mean, _min (most recent = least negative)
      - BUREAU_DAYS_CREDIT_ENDDATE_mean
      - BUREAU_CREDIT_TYPE_nunique

    bureau_balance.csv aggregations (grouped by SK_ID_BUREAU → then SK_ID_CURR):
      - BBAL_MONTHS_BALANCE_count: total month-snapshots per applicant
      - BBAL_STATUS_C_rate: fraction of months with STATUS == 'C' (closed/paid)
      - BBAL_STATUS_X_rate: fraction of months with STATUS == 'X' (unknown)
      - BBAL_DPD_MAX: max months past due (STATUS numeric values 1-5)
    
    Merge: bureau_balance aggs joined to bureau on SK_ID_BUREAU before grouping to SK_ID_CURR.
    Missing values (applicants with no bureau record) NOT filled here — pipeline.py handles that.
    """
```

---

## `src/features/previous_app.py`

```python
import pandas as pd

def build_previous_app_features(
    path: str = "data/raw/previous_application.csv"
) -> pd.DataFrame:
    """
    Returns DataFrame indexed by SK_ID_CURR with aggregated previous application features.
    
    Aggregations grouped by SK_ID_CURR:
      - PREV_APP_COUNT: total previous applications
      - PREV_APPROVED_COUNT: count where NAME_CONTRACT_STATUS == 'Approved'
      - PREV_REFUSED_COUNT: count where NAME_CONTRACT_STATUS == 'Refused'
      - PREV_APPROVAL_RATE: PREV_APPROVED_COUNT / PREV_APP_COUNT
      - PREV_AMT_APPLICATION_mean, _max
      - PREV_AMT_CREDIT_mean, _max
      - PREV_AMT_ANNUITY_mean
      - PREV_DAYS_DECISION_mean, _min  (min = most recent)
      - PREV_CREDIT_TO_APP_RATIO_mean: mean(AMT_CREDIT / AMT_APPLICATION)
      - PREV_PRODUCT_TYPE_nunique
    """
```

---

## `src/features/pos_cash.py`

```python
import pandas as pd

def build_pos_cash_features(
    path: str = "data/raw/POS_CASH_balance.csv"
) -> pd.DataFrame:
    """
    Returns DataFrame indexed by SK_ID_CURR.
    
    Aggregations grouped by SK_ID_CURR:
      - POS_MONTHS_BALANCE_count: total snapshots
      - POS_SK_DPD_MAX: max days past due across all months
      - POS_SK_DPD_DEF_MAX: max days past due (defined)
      - POS_CNT_INSTALMENT_FUTURE_mean: mean remaining instalments
      - POS_COMPLETED_COUNT: count of months where NAME_CONTRACT_STATUS == 'Completed'
    """
```

---

## `src/features/installments.py`

```python
import pandas as pd

def build_installment_features(
    path: str = "data/raw/installments_payments.csv"
) -> pd.DataFrame:
    """
    Returns DataFrame indexed by SK_ID_CURR.
    
    Derived columns (computed before grouping):
      - PAYMENT_DIFF = AMT_INSTALMENT - AMT_PAYMENT  (positive → underpaid)
      - DAYS_PAST_DUE = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT  (positive → late)
      - DAYS_BEFORE_DUE = -DAYS_PAST_DUE clipped at 0  (positive → early)
    
    Aggregations grouped by SK_ID_CURR:
      - INS_COUNT: total installment records
      - INS_PAYMENT_DIFF_mean, _max
      - INS_DAYS_PAST_DUE_mean, _max
      - INS_DAYS_BEFORE_DUE_mean
      - INS_LATE_PAYMENT_RATE: fraction of records where DAYS_PAST_DUE > 0
      - INS_AMT_PAYMENT_mean, _sum
    """
```

---

## `src/features/credit_card.py`

```python
import pandas as pd

def build_credit_card_features(
    path: str = "data/raw/credit_card_balance.csv"
) -> pd.DataFrame:
    """
    Returns DataFrame indexed by SK_ID_CURR.
    
    Derived columns (before grouping):
      - UTILIZATION = AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL  (clipped 0–1)
    
    Aggregations grouped by SK_ID_CURR:
      - CC_COUNT: total monthly snapshots
      - CC_AMT_BALANCE_mean, _max
      - CC_UTILIZATION_mean, _max
      - CC_AMT_PAYMENT_TOTAL_CURRENT_mean
      - CC_AMT_DRAWINGS_TOTAL_CURRENT_mean
      - CC_CNT_DRAWINGS_TOTAL_CURRENT_mean
      - CC_SK_DPD_MAX
    """
```

---

## `src/features/pipeline.py`

```python
import pandas as pd
import numpy as np
import json
from pathlib import Path

def build_all_features(
    config_path: str = "configs/features.yaml",
    output_dir: str = "data/processed"
) -> None:
    """
    Orchestrator. Reads config, calls all feature builders, merges on SK_ID_CURR, saves.
    
    Steps:
    1. Call build_application_features() → (train_app, test_app)
    2. For each auxiliary table (controlled by config flags):
       Call the builder → returns agg_df indexed by SK_ID_CURR
    3. Left-join each agg_df onto train_app and test_app on SK_ID_CURR
       (left join: applicants with no history get NaN → handled by downstream imputation)
    4. After all joins, impute remaining NaNs in numeric columns with median
       (fit on train, apply to test).
    5. Save to output_dir:
       - train_features.parquet
       - test_features.parquet
       - feature_names.json  (list of column names, excluding SK_ID_CURR and TARGET)
       - cat_feature_indices.json  (list of integer indices into feature_names for cat cols)
       - cat_cardinalities.json  ({col_name: n_unique+1} for embedding size)
    6. Print summary: shape, null count, feature count.
    
    Config keys (features.yaml):
      use_bureau: bool
      use_previous_app: bool
      use_pos_cash: bool
      use_installments: bool
      use_credit_card: bool
    """

if __name__ == "__main__":
    build_all_features()
```

---

## `src/models/base.py`

```python
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any

class BaseModel(ABC):
    """
    Abstract base. All model wrappers must implement this interface.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        config: dict loaded from the model's yaml config file.
        """
        self.config = config
        self.model = None  # underlying model object set during fit()

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> None:
        """
        Trains the model. Sets self.model.
        Must print/log val AUC at end of training.
        """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns 1D np.ndarray of shape (n_samples,) with probability of class 1.
        Values in [0, 1].
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Persists model to path. Extension convention:
          Tree models → .pkl (joblib)
          PyTorch models → .pt (torch.save state_dict + config)
        """

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel":
        """
        Class method. Loads and returns a fitted BaseModel instance from path.
        """
```

---

## `src/models/xgboost_model.py`

```python
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import joblib
from typing import Dict, Any, Optional
from src.models.base import BaseModel

class XGBoostModel(BaseModel):
    """XGBoost wrapper with optional Optuna hyperparameter tuning."""

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        tune: bool = False,
        n_trials: int = 50
    ) -> None:
        """
        If tune=False: trains with self.config["params"].
        If tune=True: runs Optuna study for n_trials, uses best params, then
                      trains final model on full train+val with best params.
        Sets self.model = trained xgb.XGBClassifier.
        Logs best val AUC to stdout.
        Early stopping: rounds=100, eval_set=[(X_val, y_val)].
        scale_pos_weight = (y_train==0).sum() / (y_train==1).sum()
        """

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns self.model.predict_proba(X)[:, 1]."""

    def save(self, path: str) -> None:
        """joblib.dump(self.model, path)"""

    @classmethod
    def load(cls, path: str) -> "XGBoostModel":
        """
        Returns XGBoostModel instance with self.model = joblib.load(path).
        config={} (not needed for inference).
        """

def xgb_optuna_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> float:
    """
    Optuna objective function for XGBoost tuning.
    Search space (all trial.suggest_* calls):
      n_estimators: int [300, 2000]
      max_depth: int [3, 9]
      learning_rate: float log-uniform [1e-3, 0.3]
      subsample: float [0.5, 1.0]
      colsample_bytree: float [0.3, 1.0]
      min_child_weight: int [1, 10]
      reg_alpha: float log-uniform [1e-8, 10.0]
      reg_lambda: float log-uniform [1e-8, 10.0]
    Returns val ROC-AUC (higher = better, Optuna direction="maximize").
    """
```

---

## `src/models/lgbm_model.py`

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import joblib
from src.models.base import BaseModel

class LGBMModel(BaseModel):
    """LightGBM wrapper with optional Optuna tuning."""

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        tune: bool = False,
        n_trials: int = 50
    ) -> None:
        """
        Same contract as XGBoostModel.fit.
        Uses lgb.LGBMClassifier.
        Early stopping: callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)].
        is_unbalance=True (instead of scale_pos_weight).
        """

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...
    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "LGBMModel": ...

def lgbm_optuna_objective(trial, X_train, y_train, X_val, y_val) -> float:
    """
    Search space:
      n_estimators: int [300, 3000]
      num_leaves: int [20, 300]
      learning_rate: float log-uniform [1e-3, 0.3]
      subsample: float [0.5, 1.0]
      colsample_bytree: float [0.3, 1.0]
      min_child_samples: int [5, 100]
      reg_alpha: float log-uniform [1e-8, 10.0]
      reg_lambda: float log-uniform [1e-8, 10.0]
    """
```

---

## `src/models/mlp.py`

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.models.base import BaseModel

class MLPNet(nn.Module):
    """
    Pure PyTorch module. No sklearn interface.
    Architecture defined in ARCHITECTURE.md §MLP.
    
    __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
      Builds Sequential of [Linear → BatchNorm1d → ReLU → Dropout] × len(hidden_dims)
      followed by Linear(hidden_dims[-1], 1) output head.
      No activation on output (raw logit for BCEWithLogitsLoss).
    
    forward(self, x: torch.Tensor) -> torch.Tensor:
      x shape: (batch_size, input_dim)
      Returns shape: (batch_size, 1) — raw logits.
    """

class MLPModel(BaseModel):
    """sklearn-compatible wrapper around MLPNet."""

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> None:
        """
        Training loop spec:
        - Converts DataFrames to float32 tensors.
        - DataLoader: batch_size from config, shuffle=True for train.
        - Loss: nn.BCEWithLogitsLoss(pos_weight=neg_count/pos_count tensor).
        - Optimizer: AdamW(lr=config["lr"], weight_decay=config["weight_decay"]).
        - Scheduler: OneCycleLR(max_lr=config["lr"], epochs=config["epochs"],
                                steps_per_epoch=len(train_loader)).
        - Training loop: config["epochs"] epochs; after each epoch compute val AUC
          using sigmoid(logits) → sklearn roc_auc_score.
        - Early stopping: patience=config["patience"] epochs on val AUC.
        - Saves best weights to self._best_state_dict (in-memory).
        - After training: loads best weights into self.net.
        - Sets self.model = self.net (for save/load compatibility).
        """

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        self.net.eval(); torch.no_grad().
        Returns torch.sigmoid(logits).cpu().numpy().flatten()
        """

    def save(self, path: str) -> None:
        """
        torch.save({
            "state_dict": self.net.state_dict(),
            "config": self.config,
            "input_dim": self._input_dim
        }, path)
        """

    @classmethod
    def load(cls, path: str) -> "MLPModel":
        """
        checkpoint = torch.load(path, map_location="cpu")
        Reconstructs MLPNet from checkpoint["config"] and checkpoint["input_dim"].
        Loads state dict. Returns fitted MLPModel instance.
        """
```

---

## `src/models/entity_embedding.py`

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from typing import List, Dict
from src.models.base import BaseModel

class EntityEmbeddingNet(nn.Module):
    """
    Architecture defined in ARCHITECTURE.md §EntityEmbeddingNetwork.
    
    __init__(
        self,
        cat_cardinalities: List[int],   # [n_unique+1 for each cat col]
        n_continuous: int,              # count of continuous features
        embedding_dims: List[int],      # [min(50, (card+1)//2) for each cat col]
        hidden_dims: List[int],
        dropout: float
    ):
      - One nn.Embedding per categorical column.
      - Embedding lookup outputs concatenated → shape (batch, sum(embedding_dims)).
      - Continuous features passed through BatchNorm1d(n_continuous).
      - Concatenate [embedded_cats, normed_continuous] → MLP head.
      - MLP head: same [Linear → BN → ReLU → Dropout] × len(hidden_dims) pattern.
      - Output: Linear(hidden_dims[-1], 1), no activation.
    
    forward(
        self,
        x_cat: torch.Tensor,    # (batch, n_cat_cols) — long tensor
        x_cont: torch.Tensor    # (batch, n_continuous) — float32
    ) -> torch.Tensor:          # (batch, 1) — raw logit
    """

class EntityEmbeddingModel(BaseModel):
    """
    Wrapper. Same fit/predict_proba/save/load contract as MLPModel.
    
    Additional responsibility in fit():
      - Reads data/processed/cat_feature_indices.json to split X into x_cat, x_cont.
      - Reads data/processed/cat_cardinalities.json to build embedding dims.
      - x_cat columns cast to torch.long; x_cont to torch.float32.
    
    save(): torch.save({state_dict, config, cat_feature_indices, cat_cardinalities, input_dim})
    load(): reconstructs from checkpoint, restores cat split info.
    """
```

---

## `src/models/tab_transformer.py`

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from src.models.base import BaseModel

class TabTransformerNet(nn.Module):
    """
    Architecture defined in ARCHITECTURE.md §TabTransformer.
    
    __init__(
        self,
        cat_cardinalities: List[int],
        n_continuous: int,
        d_model: int,               # embedding dim for transformer (default 32)
        n_heads: int,               # attention heads (default 8)
        n_layers: int,              # transformer encoder layers (default 6)
        ff_dim: int,                # feed-forward dim = 4 * d_model
        dropout: float,
        mlp_hidden: List[int]
    ):
      - Each categorical column: nn.Embedding(cardinality, d_model) + positional encoding.
      - Stack of n_layers TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout).
      - Transformer output (n_cat_cols × d_model) flattened → concat with
        LayerNorm(n_continuous) continuous features.
      - MLP head over concatenated vector → scalar logit.
    
    forward(x_cat, x_cont) → (batch, 1) logit
    """

class TabTransformerModel(BaseModel):
    """Same interface as EntityEmbeddingModel."""
```

---

## `src/models/train.py`

```python
import argparse
import mlflow
import yaml
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pathlib import Path

def load_data(
    features_path: str = "data/processed/train_features.parquet",
    splits_dir: str = "data/splits"
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Loads train_features.parquet.
    Loads (or creates) index arrays from splits_dir.
    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    See EXPERIMENT_DESIGN.md for split strategy.
    """

def train_and_log(
    model_type: str,            # one of: xgboost, lgbm, mlp, entity_embedding, tab_transformer
    config_path: str | None = None,  # if None, uses configs/{model_type}.yaml
    tune: bool = False,
    experiment_name: str | None = None  # if None, inferred from model_type
) -> str:
    """
    Main training entry point.
    1. Loads config from yaml.
    2. Loads data via load_data().
    3. Instantiates model class based on model_type.
    4. Calls model.fit(X_train, y_train, X_val, y_val, tune=tune).
    5. Calls model.predict_proba(X_val) → computes val_roc_auc.
    6. Calls model.predict_proba(X_test) → computes test_roc_auc.
    7. mlflow.start_run() context:
       - mlflow.log_params(config)
       - mlflow.log_metric("val_roc_auc", val_roc_auc)
       - mlflow.log_metric("test_roc_auc", test_roc_auc)
       - mlflow.log_artifact(model_save_path)
       - mlflow.set_tag("model_type", model_type)
    8. Returns mlflow run_id (str).
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
        choices=["xgboost","lgbm","mlp","entity_embedding","tab_transformer"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    train_and_log(args.model, args.config, args.tune)
```

---

## `src/evaluation/metrics.py`

```python
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy import stats
from typing import Dict

def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Returns dict with keys:
      roc_auc: float — sklearn roc_auc_score
      ks_stat: float — max(TPR - FPR) across all thresholds
      brier_score: float — sklearn brier_score_loss
      calibration_slope: float — slope of linear regression (y_true ~ y_prob)
                                  1.0 = perfectly calibrated
    """

def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Computes Kolmogorov-Smirnov statistic for binary classification.
    = max separation between CDFs of predicted probabilities for class 0 vs class 1.
    """

def wilcoxon_test(
    scores_a: np.ndarray,    # per-sample probabilities from model A (on test set)
    scores_b: np.ndarray,    # per-sample probabilities from model B
    y_true: np.ndarray
) -> Dict[str, float]:
    """
    Computes Wilcoxon signed-rank test between fold-level AUCs
    (5-fold stratified CV on test set, pairing scores by fold).
    Returns {"statistic": float, "p_value": float}.
    """
```

---

## `src/evaluation/comparison.py`

```python
import pandas as pd
from typing import Dict, Any

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

def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """
    Prints markdown-formatted table:
    | Model | ROC-AUC | KS Stat | Brier Score | Calibration Slope |
    """
```

---

## `src/serving/schemas.py`

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict

class PredictRequest(BaseModel):
    """See SERVING_SPEC.md for full field list."""
    SK_ID_CURR: Optional[int] = None          # for logging only
    EXT_SOURCE_1: Optional[float] = None
    EXT_SOURCE_2: Optional[float] = None
    EXT_SOURCE_3: Optional[float] = None
    AMT_CREDIT: float
    AMT_INCOME_TOTAL: float
    AMT_ANNUITY: Optional[float] = None
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    # ... all other features from feature_names.json
    # All optional except AMT_CREDIT, AMT_INCOME_TOTAL, DAYS_BIRTH, DAYS_EMPLOYED
    # Missing optionals filled with feature median at serving time

class PredictResponse(BaseModel):
    default_probability: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    request_id: str

class HealthResponse(BaseModel):
    status: str          # "ok" or "degraded"
    model_loaded: bool
    model_version: str

class DriftResponse(BaseModel):
    psi_values: Dict[str, float]   # {feature_name: psi_score}
    n_records_evaluated: int
    alert: bool                    # True if any PSI > threshold
    threshold: float
```

---

## `src/serving/model_loader.py`

```python
from typing import Any
import yaml

_model = None          # module-level singleton
_model_version = ""

def load_model(config_path: str = "configs/serving.yaml") -> None:
    """
    Called once at FastAPI startup.
    Reads serving.yaml for artifact_path and model_type.
    If artifact_path starts with "s3://": downloads to /tmp/ first via boto3.
    If model_type in ["xgboost","lgbm"]: loads with joblib.
    If model_type in ["mlp","entity_embedding","tab_transformer"]: loads with torch.
    Sets module-level _model and _model_version.
    Raises RuntimeError if artifact not found.
    """

def get_model() -> Any:
    """Returns _model. Raises RuntimeError if not loaded."""

def get_model_version() -> str:
    """Returns _model_version string."""
```

---

## `src/serving/router.py`

```python
from fastapi import APIRouter, HTTPException
from src.serving.schemas import PredictRequest, PredictResponse, HealthResponse, DriftResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Returns {"status": "ok", "model_loaded": True, "model_version": "..."}.
    Returns {"status": "degraded", "model_loaded": False, ...} if model not loaded.
    Always HTTP 200 (health endpoint never 5xx).
    """

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    1. Convert request to DataFrame (1 row).
    2. Reindex to feature_names.json order, filling missing with median values.
    3. Call model.predict_proba(df)[0].
    4. Append raw features + probability to in-memory rolling window (last 1000 records).
    5. Return PredictResponse.
    HTTP 422: Pydantic validation failure (automatic).
    HTTP 503: model not loaded.
    """

@router.get("/drift", response_model=DriftResponse)
async def drift() -> DriftResponse:
    """
    Runs PSI computation on rolling window vs reference_stats.json.
    Returns PSI per feature + alert flag.
    HTTP 503 if fewer than 50 records in window (not enough data).
    """
```

---

## `src/monitoring/psi.py`

```python
import numpy as np
from typing import Union

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    eps: float = 1e-6
) -> float:
    """
    Population Stability Index.
    reference: 1D array of reference distribution values (e.g., training data)
    current:   1D array of current distribution values (e.g., recent predictions)
    bins: number of equal-frequency bins (computed on reference).
    eps: small constant added to all bucket proportions before log (avoids log(0)).
    
    Algorithm:
    1. Compute bin edges from np.percentile(reference, np.linspace(0,100,bins+1)).
    2. Clip both arrays to [min_edge, max_edge].
    3. Compute proportion of values in each bin for reference and current.
    4. PSI = sum((p_ref - p_cur) * log((p_ref+eps) / (p_cur+eps))).
    
    Returns float. Interpretation:
      PSI < 0.1: no significant change
      0.1 ≤ PSI < 0.2: moderate change, monitor
      PSI ≥ 0.2: significant shift, trigger alert
    """

def compute_all_psi(
    reference_stats: dict,      # loaded from data/drift/reference_stats.json
    current_df: "pd.DataFrame", # recent scored records
    bins: int = 10
) -> dict[str, float]:
    """
    Computes PSI for every numeric feature in reference_stats.
    Skips features not present in current_df.
    Returns {feature_name: psi_value}.
    """
```

---

## `src/monitoring/alert.py`

```python
import json
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def check_and_alert(
    psi_values: Dict[str, float],
    threshold: float = 0.2,
    sns_arn: str | None = None
) -> bool:
    """
    Checks if any feature PSI exceeds threshold.
    If yes:
      - Logs WARNING with JSON payload: {alert_type, features_above_threshold, max_psi}
      - If sns_arn provided: publishes to AWS SNS via boto3.
    Returns True if alert fired, False otherwise.
    """
```
