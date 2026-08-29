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
          Tree models -> .pkl (joblib)
          PyTorch models -> .pt (torch.save state_dict + config)
        """

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel":
        """
        Class method. Loads and returns a fitted BaseModel instance from path.
        """

import torch
from typing import List

class TabularDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
        cat_indices: List[int],
        has_labels: bool = True
    ):
        # Cast inputs properly
        X_vals = X.values
        self.x_cat = torch.tensor(X_vals[:, cat_indices], dtype=torch.long) if len(cat_indices) > 0 else torch.zeros((len(X), 0), dtype=torch.long)
        
        cont_indices = [i for i in range(X.shape[1]) if i not in cat_indices]
        self.x_cont = torch.tensor(X_vals[:, cont_indices], dtype=torch.float32)
        
        self.y = torch.tensor(y.values, dtype=torch.float32) if has_labels and y is not None else None
        self.has_labels = has_labels

    def __len__(self):
        return len(self.x_cont)

    def __getitem__(self, idx):
        if self.has_labels:
            return self.x_cat[idx], self.x_cont[idx], self.y[idx]
        return self.x_cat[idx], self.x_cont[idx]

