import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.models.base import BaseModel, TabularDataset
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

class MLPNet(nn.Module):
    """
    Pure PyTorch module for MLP.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            # Use BatchNorm1d if batch size > 1
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MLPModel(BaseModel):
    """sklearn-compatible wrapper around MLPNet."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.net = None
        self._input_dim = None
        self._best_state_dict = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        tune: bool = False,
        n_trials: int = 50
    ) -> None:
        """Trains the model with early stopping."""
        params = self.config.get("params", {})
        batch_size = params.get("batch_size", 1024)
        epochs = params.get("epochs", 50)
        patience = params.get("patience", 10)
        lr = params.get("lr", 0.001)
        dropout = params.get("dropout", 0.3)
        weight_decay = params.get("weight_decay", 0.0001)
        hidden_dims = params.get("hidden_dims", [512, 256, 128])

        self._input_dim = X_train.shape[1]
        
        # Instantiate network
        self.net = MLPNet(
            input_dim=self._input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(self.device)

        # Prepare datasets (MLP uses empty cat_indices)
        train_ds = TabularDataset(X_train, y_train, cat_indices=[], has_labels=True)
        val_ds = TabularDataset(X_val, y_val, cat_indices=[], has_labels=True)
        
        # For batch norm to work on small batches in test runs, set drop_last=False
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=(len(train_ds) > batch_size))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Calculate pos_weight for class imbalance
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        pos_weight_val = neg_count / pos_count if pos_count > 0 else 1.0
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(self.device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch
        )

        best_val_auc = 0.0
        epochs_no_improve = 0

        print(f"Training MLP on {self.device}...")
        for epoch in range(epochs):
            # Training phase
            self.net.train()
            train_loss = 0.0
            for _, x_cont, y in train_loader:
                x_cont, y = x_cont.to(self.device), y.to(self.device).unsqueeze(1)
                
                optimizer.zero_grad()
                logits = self.net(x_cont)
                loss = criterion(logits, y)
                loss.backward()
                # Clip gradients
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item() * len(y)
            
            train_loss /= len(train_ds)

            # Validation phase
            self.net.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for _, x_cont, y in val_loader:
                    x_cont = x_cont.to(self.device)
                    logits = self.net(x_cont)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds.extend(probs)
                    val_targets.extend(y.numpy())
            
            val_auc = roc_auc_score(val_targets, val_preds) if len(np.unique(val_targets)) > 1 else 0.5
            
            # Print progress every 10 epochs or if it's the last one
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.5f}")

            # Check early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                self._best_state_dict = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}!")
                    break

        # Load best weights
        if self._best_state_dict is not None:
            self.net.load_state_dict({k: v.to(self.device) for k, v in self._best_state_dict().items() if hasattr(self._best_state_dict, 'items')} if not isinstance(self._best_state_dict, dict) else self._best_state_dict)
        self.model = self.net

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.net is None:
            raise ValueError("Model is not trained yet. Call fit() first.")
        self.net.eval()
        ds = TabularDataset(X, y=None, cat_indices=[], has_labels=False)
        loader = DataLoader(ds, batch_size=1024, shuffle=False)
        
        all_probs = []
        with torch.no_grad():
            for _, x_cont in loader:
                x_cont = x_cont.to(self.device)
                logits = self.net(x_cont)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.append(probs)
        return np.concatenate(all_probs)

    def save(self, path: str) -> None:
        if self.net is None:
            raise ValueError("Model is not trained yet. Cannot save.")
        torch.save({
            "state_dict": self.net.state_dict(),
            "config": self.config,
            "input_dim": self._input_dim
        }, path)
        print(f"Saved MLP model to {path}")

    @classmethod
    def load(cls, path: str) -> "MLPModel":
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]
        input_dim = checkpoint["input_dim"]
        
        inst = cls(config)
        inst._input_dim = input_dim
        inst.net = MLPNet(
            input_dim=input_dim,
            hidden_dims=config.get("params", {}).get("hidden_dims", [512, 256, 128]),
            dropout=config.get("params", {}).get("dropout", 0.3)
        )
        inst.net.load_state_dict(checkpoint["state_dict"])
        inst.net.to(inst.device)
        inst.model = inst.net
        return inst
