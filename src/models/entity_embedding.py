import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.models.base import BaseModel, TabularDataset
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

class EntityEmbeddingNet(nn.Module):
    """
    Entity Embedding Network architecture.
    """
    def __init__(
        self,
        cat_cardinalities: List[int],
        n_continuous: int,
        embedding_dims: List[int],
        hidden_dims: List[int],
        dropout: float
    ):
        super().__init__()
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, dim) for card, dim in zip(cat_cardinalities, embedding_dims)
        ])
        
        self.cont_bn = nn.BatchNorm1d(n_continuous) if n_continuous > 0 else nn.Identity()
        
        total_emb_dim = sum(embedding_dims)
        input_dim = total_emb_dim + n_continuous
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        embedded = []
        for i, emb in enumerate(self.embeddings):
            # Embed each column
            embedded.append(emb(x_cat[:, i]))
            
        if len(embedded) > 0:
            x_cat_emb = torch.cat(embedded, dim=1)
        else:
            x_cat_emb = torch.zeros((x_cont.size(0), 0), device=x_cont.device)
            
        x_cont_norm = self.cont_bn(x_cont)
        
        x = torch.cat([x_cat_emb, x_cont_norm], dim=1)
        return self.mlp(x)

class EntityEmbeddingModel(BaseModel):
    """sklearn-compatible wrapper around EntityEmbeddingNet."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.net = None
        self._best_state_dict = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cat_indices = []
        self.cat_cardinalities = []
        self.embedding_dims = []
        self.n_continuous = 0

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
        hidden_dims = params.get("hidden_dims", [256, 128])

        # Load categorical features metadata
        try:
            with open("data/processed/cat_feature_indices.json", "r") as f:
                self.cat_indices = json.load(f)
            with open("data/processed/cat_cardinalities.json", "r") as f:
                cat_cards = json.load(f)
        except Exception as e:
            print(f"Error loading categorical metadata: {e}. Using empty.")
            self.cat_indices = []
            cat_cards = {}

        feature_names = X_train.columns.tolist()
        self.n_continuous = len(feature_names) - len(self.cat_indices)
        
        # Reconstruct list of cardinalities in correct order
        cat_cols = [feature_names[i] for i in self.cat_indices]
        self.cat_cardinalities = [cat_cards.get(col, 1) for col in cat_cols]
        self.embedding_dims = [min(50, (card + 1) // 2) for card in self.cat_cardinalities]

        # Instantiate network
        self.net = EntityEmbeddingNet(
            cat_cardinalities=self.cat_cardinalities,
            n_continuous=self.n_continuous,
            embedding_dims=self.embedding_dims,
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(self.device)

        # Prepare datasets
        train_ds = TabularDataset(X_train, y_train, cat_indices=self.cat_indices, has_labels=True)
        val_ds = TabularDataset(X_val, y_val, cat_indices=self.cat_indices, has_labels=True)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=(len(train_ds) > batch_size))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Imbalance correction
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

        print(f"Training EntityEmbeddingNetwork on {self.device}...")
        for epoch in range(epochs):
            # Training
            self.net.train()
            train_loss = 0.0
            for x_cat, x_cont, y in train_loader:
                x_cat, x_cont, y = x_cat.to(self.device), x_cont.to(self.device), y.to(self.device).unsqueeze(1)
                
                optimizer.zero_grad()
                logits = self.net(x_cat, x_cont)
                loss = criterion(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item() * len(y)
            
            train_loss /= len(train_ds)

            # Validation
            self.net.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for x_cat, x_cont, y in val_loader:
                    x_cat, x_cont = x_cat.to(self.device), x_cont.to(self.device)
                    logits = self.net(x_cat, x_cont)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds.extend(probs)
                    val_targets.extend(y.numpy())
            
            val_auc = roc_auc_score(val_targets, val_preds) if len(np.unique(val_targets)) > 1 else 0.5
            
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.5f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                self._best_state_dict = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}!")
                    break

        if self._best_state_dict is not None:
            self.net.load_state_dict({k: v.to(self.device) for k, v in self._best_state_dict().items() if hasattr(self._best_state_dict, 'items')} if not isinstance(self._best_state_dict, dict) else self._best_state_dict)
        self.model = self.net

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.net is None:
            raise ValueError("Model is not trained yet. Call fit() first.")
        self.net.eval()
        ds = TabularDataset(X, y=None, cat_indices=self.cat_indices, has_labels=False)
        loader = DataLoader(ds, batch_size=1024, shuffle=False)
        
        all_probs = []
        with torch.no_grad():
            for x_cat, x_cont in loader:
                x_cat, x_cont = x_cat.to(self.device), x_cont.to(self.device)
                logits = self.net(x_cat, x_cont)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.append(probs)
        return np.concatenate(all_probs)

    def save(self, path: str) -> None:
        if self.net is None:
            raise ValueError("Model is not trained yet. Cannot save.")
        torch.save({
            "state_dict": self.net.state_dict(),
            "config": self.config,
            "cat_indices": self.cat_indices,
            "cat_cardinalities": self.cat_cardinalities,
            "embedding_dims": self.embedding_dims,
            "n_continuous": self.n_continuous
        }, path)
        print(f"Saved EEN model to {path}")

    @classmethod
    def load(cls, path: str) -> "EntityEmbeddingModel":
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]
        
        inst = cls(config)
        inst.cat_indices = checkpoint["cat_indices"]
        inst.cat_cardinalities = checkpoint["cat_cardinalities"]
        inst.embedding_dims = checkpoint["embedding_dims"]
        inst.n_continuous = checkpoint["n_continuous"]
        
        inst.net = EntityEmbeddingNet(
            cat_cardinalities=inst.cat_cardinalities,
            n_continuous=inst.n_continuous,
            embedding_dims=inst.embedding_dims,
            hidden_dims=config.get("params", {}).get("hidden_dims", [256, 128]),
            dropout=config.get("params", {}).get("dropout", 0.3)
        )
        inst.net.load_state_dict(checkpoint["state_dict"])
        inst.net.to(inst.device)
        inst.model = inst.net
        return inst
