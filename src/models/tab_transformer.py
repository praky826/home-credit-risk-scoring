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

class TabTransformerNet(nn.Module):
    """
    TabTransformer architecture.
    """
    def __init__(
        self,
        cat_cardinalities: List[int],
        n_continuous: int,
        d_model: int = 32,
        n_heads: int = 8,
        n_layers: int = 6,
        ff_dim: int = 128,
        dropout: float = 0.1,
        mlp_hidden: List[int] = [128, 64]
    ):
        super().__init__()
        self.n_cat = len(cat_cardinalities)
        
        # All categoricals embedded to same d_model
        self.col_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, d_model) for card in cat_cardinalities
        ])
        
        # Positional embedding (one per column)
        self.col_pos_embedding = nn.Parameter(torch.randn(1, self.n_cat, d_model)) if self.n_cat > 0 else None
        
        # Transformer encoder
        if self.n_cat > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        else:
            self.transformer = None

        # Layer norm for continuous features
        self.cont_norm = nn.LayerNorm(n_continuous) if n_continuous > 0 else nn.Identity()
        
        # MLP input dimension
        mlp_input = self.n_cat * d_model + n_continuous
        
        layers = []
        in_dim = mlp_input
        for h in mlp_hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        batch_size = x_cont.size(0)
        
        if self.n_cat > 0:
            # Embed each categorical column -> (batch, n_cat, d_model)
            embs = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.col_embeddings)], dim=1)
            
            # Add positional encoding
            embs = embs + self.col_pos_embedding
            
            # Apply Transformer
            ctx_embs = self.transformer(embs)
            
            # Flatten transformer output -> (batch, n_cat * d_model)
            ctx_flat = ctx_embs.reshape(batch_size, -1)
        else:
            ctx_flat = torch.zeros((batch_size, 0), device=x_cont.device)
            
        # Normalize continuous
        x_cont_norm = self.cont_norm(x_cont)
        
        # Concatenate and MLP
        x = torch.cat([ctx_flat, x_cont_norm], dim=1)
        return self.mlp(x)

class TabTransformerModel(BaseModel):
    """sklearn-compatible wrapper around TabTransformerNet."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.net = None
        self._best_state_dict = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cat_indices = []
        self.cat_cardinalities = []
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
        batch_size = params.get("batch_size", 512) # Default 512 for TabTransformer
        epochs = params.get("epochs", 30) # Default 30
        patience = params.get("patience", 5) # Default 5
        lr = params.get("lr", 0.001)
        dropout = params.get("dropout", 0.1)
        weight_decay = params.get("weight_decay", 0.0001)
        
        d_model = params.get("d_model", 32)
        n_heads = params.get("n_heads", 8)
        n_layers = params.get("n_layers", 6)
        ff_dim = params.get("ff_dim", 128)
        mlp_hidden = params.get("mlp_hidden", [128, 64])

        # Load categorical metadata
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
        
        cat_cols = [feature_names[i] for i in self.cat_indices]
        self.cat_cardinalities = [cat_cards.get(col, 1) for col in cat_cols]

        # Instantiate network
        self.net = TabTransformerNet(
            cat_cardinalities=self.cat_cardinalities,
            n_continuous=self.n_continuous,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            mlp_hidden=mlp_hidden
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

        print(f"Training TabTransformer on {self.device}...")
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
            "n_continuous": self.n_continuous
        }, path)
        print(f"Saved TabTransformer model to {path}")

    @classmethod
    def load(cls, path: str) -> "TabTransformerModel":
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]
        
        inst = cls(config)
        inst.cat_indices = checkpoint["cat_indices"]
        inst.cat_cardinalities = checkpoint["cat_cardinalities"]
        inst.n_continuous = checkpoint["n_continuous"]
        
        params = config.get("params", {})
        inst.net = TabTransformerNet(
            cat_cardinalities=inst.cat_cardinalities,
            n_continuous=inst.n_continuous,
            d_model=params.get("d_model", 32),
            n_heads=params.get("n_heads", 8),
            n_layers=params.get("n_layers", 6),
            ff_dim=params.get("ff_dim", 128),
            dropout=params.get("dropout", 0.1),
            mlp_hidden=params.get("mlp_hidden", [128, 64])
        )
        inst.net.load_state_dict(checkpoint["state_dict"])
        inst.net.to(inst.device)
        inst.model = inst.net
        return inst
