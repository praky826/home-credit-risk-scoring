# ARCHITECTURE.md
## Home Credit Default Risk — PyTorch Architecture Specifications

All three neural network architectures fully specified. No ambiguity.

---

## Architecture 1: MLP (Multilayer Perceptron)

### Purpose
Baseline neural network. All features treated as continuous floats. Categorical features are label-encoded integers used as float inputs (not embedded). Establishes NN baseline before introducing embeddings.

### Input
- `x: torch.Tensor` — shape `(batch_size, n_features)`, dtype `float32`
- Features are label-encoded and median-imputed (same matrix as tree models)
- **No separate categorical handling**

### Architecture Definition
```python
class MLPNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # output head
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # shape: (batch_size, 1) — raw logit
```

### Layer Sizes (default config)
```
Input:           n_features (≈ 250 after feature engineering)
Hidden Layer 1:  512 units → BatchNorm1d(512) → ReLU → Dropout(p)
Hidden Layer 2:  256 units → BatchNorm1d(256) → ReLU → Dropout(p)
Hidden Layer 3:  128 units → BatchNorm1d(128) → ReLU → Dropout(p)
Output:          1 unit (raw logit)
```

### Training Configuration
| Parameter | Value | Notes |
|---|---|---|
| Loss | `BCEWithLogitsLoss(pos_weight=tensor)` | pos_weight = n_neg/n_pos per batch composition |
| Optimizer | `AdamW(lr, weight_decay)` | weight_decay from grid search |
| Scheduler | `OneCycleLR(max_lr=lr, epochs=E, steps_per_epoch=S)` | Warm up + cosine anneal |
| Batch size | 1024 | Fits in GPU memory for this dataset |
| Epochs max | 50 | Early stopping kicks in earlier |
| Early stopping | patience=10, monitor=val_roc_auc, mode=max | Save best state_dict |
| Gradient clip | `clip_grad_norm_(params, max_norm=1.0)` | Prevents gradient explosion |
| Mixed precision | `torch.cuda.amp.GradScaler()` if CUDA available | Halves memory, 2x speed |

### Normalization Placement
BatchNorm is placed **after Linear, before activation** (pre-activation norm):
```
Linear → BatchNorm1d → ReLU → Dropout
```
This is the standard convention for classification MLPs. Do NOT put BN after Dropout.

### Output Interpretation
- Raw logit from the model → apply `torch.sigmoid()` to get probability
- Training uses `BCEWithLogitsLoss` which applies sigmoid internally (numerically stable)
- Inference: `prob = torch.sigmoid(model(x)).item()`

---

## Architecture 2: Entity Embedding Network (EEN)

### Purpose
Learns dense vector representations for each categorical feature. Each categorical value maps to a learned embedding vector. These are concatenated with continuous features and passed through an MLP. This architecture outperforms plain MLP on datasets with high-cardinality categoricals.

### Reference
Guo & Berkhahn (2016): *Entity Embeddings of Categorical Variables*

### Input
- `x_cat: torch.Tensor` — shape `(batch_size, n_cat_cols)`, dtype `torch.long`
  - Values are label-encoded category indices (0 to cardinality-1)
  - Unknown categories encoded as 0 (reserved index)
- `x_cont: torch.Tensor` — shape `(batch_size, n_continuous)`, dtype `float32`

### Embedding Dimension Formula
```python
def embedding_dim(cardinality: int) -> int:
    """Standard rule from Fast.ai / Guo & Berkhahn."""
    return min(50, (cardinality + 1) // 2)
```

Example embedding dims for Home Credit categoricals:
| Column | n_unique | Embedding Dim |
|---|---|---|
| NAME_EDUCATION_TYPE | 5 | 3 |
| OCCUPATION_TYPE | 18 | 9 |
| ORGANIZATION_TYPE | 58 | 29 |
| NAME_INCOME_TYPE | 8 | 4 |
| NAME_HOUSING_TYPE | 6 | 3 |
| CODE_GENDER (binary) | 2 | 1 |

### Architecture Definition
```python
class EntityEmbeddingNet(nn.Module):
    def __init__(
        self,
        cat_cardinalities: List[int],   # one per categorical column
        n_continuous: int,
        embedding_dims: List[int],      # precomputed via embedding_dim()
        hidden_dims: List[int],
        dropout: float
    ):
        super().__init__()

        # One embedding table per categorical column
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, emb_dim)   # +1 for unknown/OOV index 0
            for card, emb_dim in zip(cat_cardinalities, embedding_dims)
        ])

        # Batch norm for continuous features
        self.cont_bn = nn.BatchNorm1d(n_continuous)

        # MLP head
        total_emb_dim = sum(embedding_dims)
        mlp_input_dim = total_emb_dim + n_continuous
        layers = []
        in_dim = mlp_input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        # Lookup embeddings for each categorical column
        emb_outputs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(emb_outputs, dim=1)   # (batch, sum_emb_dims)

        # Normalize continuous features
        x_cont = self.cont_bn(x_cont)            # (batch, n_continuous)

        # Concatenate and pass through MLP
        x = torch.cat([x_emb, x_cont], dim=1)   # (batch, total_input_dim)
        return self.mlp(x)                        # (batch, 1) — raw logit
```

### Layer Sizes (default config)
```
Embedding lookup: sum of all embedding dims ≈ 80-100 dims total
Continuous BN:    n_continuous ≈ 180-200 features
After concat:     ≈ 280 dims
Hidden Layer 1:   256 units → BN → ReLU → Dropout
Hidden Layer 2:   128 units → BN → ReLU → Dropout
Output:           1 unit (raw logit)
```

### Training Configuration
Same as MLP except:
| Parameter | Value |
|---|---|
| Batch size | 1024 |
| Epochs max | 50 |
| Hidden dims | [256, 128] |
| DataLoader | must return (x_cat, x_cont, y) tuples |

### Data Split for Forward Pass
```python
# In MLPModel's DataLoader, all features in one tensor.
# In EntityEmbeddingModel's DataLoader, split by column type:
cat_cols_idx = json.load(open("data/processed/cat_feature_indices.json"))
# cat_feature_indices.json contains list of integer indices into feature_names
# x_cat = X_tensor[:, cat_cols_idx].long()
# x_cont = X_tensor[:, cont_cols_idx].float()
```

### Embedding Initialization
Default PyTorch random initialization (uniform). Do NOT use pretrained embeddings — we're learning from scratch.

---

## Architecture 3: TabTransformer

### Purpose
Applies Transformer attention to categorical embeddings. Each categorical feature is embedded and then contextualised by attention over all other categoricals. Continuous features bypass the transformer and are merged later. Based on Huang et al. (2020).

### Reference
Huang et al. (2020): *TabTransformer: Tabular Data Modeling Using Contextual Embeddings*

### Input
Same as EEN:
- `x_cat: torch.Tensor` — `(batch, n_cat_cols)`, `torch.long`
- `x_cont: torch.Tensor` — `(batch, n_continuous)`, `float32`

### Architecture Definition
```python
class TabTransformerNet(nn.Module):
    def __init__(
        self,
        cat_cardinalities: List[int],
        n_continuous: int,
        d_model: int = 32,          # embedding dimension for all cat cols
        n_heads: int = 8,           # attention heads (d_model must be divisible by n_heads)
        n_layers: int = 6,          # transformer encoder layers
        ff_dim: int = 128,          # feed-forward dim inside transformer (= 4 * d_model)
        dropout: float = 0.1,
        mlp_hidden: List[int] = [128, 64]
    ):
        super().__init__()
        n_cat = len(cat_cardinalities)

        # All categoricals embedded to same d_model (required for attention)
        self.col_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, d_model) for card in cat_cardinalities
        ])

        # Column-specific positional embedding (learned, one per column)
        self.col_pos_embedding = nn.Parameter(torch.randn(1, n_cat, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True   # (batch, seq_len, d_model) — requires PyTorch >= 1.9
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Layer norm for continuous features
        self.cont_norm = nn.LayerNorm(n_continuous)

        # MLP head
        mlp_input = n_cat * d_model + n_continuous
        layers = []
        in_dim = mlp_input
        for h in mlp_hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        batch_size = x_cat.size(0)
        n_cat = x_cat.size(1)

        # Embed each categorical column → (batch, n_cat, d_model)
        embs = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.col_embeddings)], dim=1)

        # Add positional encoding
        embs = embs + self.col_pos_embedding    # (batch, n_cat, d_model)

        # Apply Transformer
        ctx_embs = self.transformer(embs)        # (batch, n_cat, d_model)

        # Flatten transformer output
        ctx_flat = ctx_embs.reshape(batch_size, -1)  # (batch, n_cat * d_model)

        # Normalize continuous
        x_cont = self.cont_norm(x_cont)           # (batch, n_continuous)

        # Concatenate and MLP
        x = torch.cat([ctx_flat, x_cont], dim=1)  # (batch, n_cat*d_model + n_continuous)
        return self.mlp(x)                         # (batch, 1) logit
```

### Configuration (default)
| Parameter | Value | Constraint |
|---|---|---|
| `d_model` | 32 | Must be divisible by `n_heads` |
| `n_heads` | 8 | d_model / n_heads = 4 (head dim) |
| `n_layers` | 6 | From paper ablation |
| `ff_dim` | 128 | = 4 × d_model |
| `dropout` | 0.1 | Inside transformer layers |
| `mlp_hidden` | [128, 64] | Shallower than EEN (transformer already rich) |
| `batch_size` | 512 | Smaller than MLP/EEN due to transformer memory |
| `epochs` | 30 | Transformer converges faster |
| `patience` | 5 | Stricter early stopping |
| `learning_rate` | 1e-3 or 3e-4 | Grid search |

### Memory Note
TabTransformer is significantly more memory-hungry than MLP/EEN due to attention matrices.
On 8GB GPU: batch_size=512 is safe. If OOM, reduce to 256.
n_cat_cols ≈ 20–25 for this dataset → attention matrix size manageable.

---

## Shared Training Utilities

### Dataset Class
```python
class TabularDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
        cat_indices: List[int],
        has_labels: bool = True
    ):
        self.x_cat = torch.tensor(X.iloc[:, cat_indices].values, dtype=torch.long)
        cont_indices = [i for i in range(X.shape[1]) if i not in cat_indices]
        self.x_cont = torch.tensor(X.iloc[:, cont_indices].values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32) if has_labels else None
        self.has_labels = has_labels

    def __len__(self): return len(self.x_cat)

    def __getitem__(self, idx):
        if self.has_labels:
            return self.x_cat[idx], self.x_cont[idx], self.y[idx]
        return self.x_cat[idx], self.x_cont[idx]
```

For MLP: `cat_indices=[]` so all features go to `x_cont`; `x_cat` is empty.

### Loss Function
```python
pos_weight = torch.tensor(
    [(y_train == 0).sum() / (y_train == 1).sum()],
    dtype=torch.float32
).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### Inference
```python
model.eval()
all_probs = []
with torch.no_grad():
    for batch in loader:
        x_cat, x_cont = batch[0].to(device), batch[1].to(device)
        logits = model(x_cat, x_cont)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
return np.concatenate(all_probs).flatten()
```
