# EXPERIMENT_DESIGN.md
## Home Credit Default Risk — Experiment Protocol

The exact protocol for the model comparison. Nothing left to interpretation.

---

## 1. Research Question

**Do neural networks (MLP, Entity Embedding Network, TabTransformer) outperform gradient boosted trees (XGBoost, LightGBM) on the Home Credit Default Risk dataset, and if so, under what conditions?**

Secondary questions:
- Does entity embedding over categorical features provide meaningful lift over raw label encoding?
- Is any performance difference statistically significant?
- Are the models' errors correlated (same applicants misclassified) or complementary?

---

## 2. Data Split Strategy

### Why Not Random Split
The Kaggle competition explicitly evaluates temporal stability (test set comes from a later time window). A random split would overestimate generalization. We use a time-based approach.

### Split Construction
```python
# DAYS_DECISION does not exist in application_train — proxy: use SK_ID_CURR ordering
# Higher SK_ID_CURR correlates with later application date (confirmed in EDA)

def create_splits(
    df: pd.DataFrame,          # application_train with SK_ID_CURR + TARGET
    val_frac: float = 0.10,    # 10% validation (~30,751 rows)
    test_frac: float = 0.10,   # 10% holdout test (~30,751 rows)
    splits_dir: str = "data/splits",
    random_seed: int = 42
) -> None:
    """
    Sort df by SK_ID_CURR ascending (proxy for time).
    Train: first 80% of sorted rows.
    Val:   next 10%.
    Test:  last 10% (most recent applications).
    
    Saves train_idx.npy, val_idx.npy, test_idx.npy as arrays of SK_ID_CURR values.
    Prints target rates for each split (should be similar; large divergence → recheck).
    """
```

### Split Sizes (approximate)
| Split | Rows | Positive Rate |
|---|---|---|
| Train | 246,009 | ~8.1% |
| Val | 30,751 | ~8.1% |
| Test | 30,751 | ~8.1% |

### Reproducibility
- Splits saved as `.npy` files in `data/splits/` and committed to git.
- `random_seed=42` used for any stochastic elements (Optuna, PyTorch DataLoader).
- If `data/splits/` already exists, `load_data()` uses existing splits without recreation.

---

## 3. Primary Evaluation Metric

### ROC-AUC (Area Under ROC Curve)
- **Why:** Competition metric; threshold-independent; appropriate for imbalanced classes.
- **How computed:** `sklearn.metrics.roc_auc_score(y_true, y_prob)`
- **Reported on:** Validation set (for model selection during training), test set (for final comparison).
- **Never use training AUC for comparison** — only val and test.

### Secondary Metrics (reported in comparison table, not used for model selection)

| Metric | Function | Interpretation |
|---|---|---|
| KS Statistic | `ks_statistic(y_true, y_prob)` | Max TPR-FPR separation; used in credit scoring |
| Brier Score | `sklearn.metrics.brier_score_loss` | Calibration quality; lower = better |
| Calibration Slope | `linregress(y_prob, y_true).slope` | 1.0 = perfectly calibrated |
| Log Loss | `sklearn.metrics.log_loss` | Information-theoretic; aligns with training objective |

---

## 4. Hyperparameter Search Protocol

### Tree Models (XGBoost and LightGBM)

**Framework:** Optuna 3.x, `TPESampler`, `direction="maximize"`  
**Objective:** Val ROC-AUC  
**Trials:** 50 per model  
**Pruner:** MedianPruner (prunes unpromising trials at 20 rounds)  
**Parallelism:** n_jobs=1 (single GPU machine; Optuna handles trial parallelism)

#### XGBoost Search Space
```yaml
n_estimators:     int   [300, 2000]      # early stopping finds actual best
max_depth:        int   [3, 9]
learning_rate:    float log-uniform [0.001, 0.3]
subsample:        float uniform [0.5, 1.0]
colsample_bytree: float uniform [0.3, 1.0]
min_child_weight: int   [1, 10]
reg_alpha:        float log-uniform [1e-8, 10.0]
reg_lambda:       float log-uniform [1e-8, 10.0]
# Fixed across all trials:
tree_method:      gpu_hist  (or hist if no GPU)
eval_metric:      auc
early_stopping_rounds: 100
scale_pos_weight: (n_neg / n_pos)  # class imbalance correction
```

#### LightGBM Search Space
```yaml
n_estimators:       int   [300, 3000]
num_leaves:         int   [20, 300]
learning_rate:      float log-uniform [0.001, 0.3]
subsample:          float uniform [0.5, 1.0]       # bagging_fraction
colsample_bytree:   float uniform [0.3, 1.0]       # feature_fraction
min_child_samples:  int   [5, 100]
reg_alpha:          float log-uniform [1e-8, 10.0]
reg_lambda:         float log-uniform [1e-8, 10.0]
# Fixed:
is_unbalance:       true
metric:             auc
early_stopping_rounds: 100
verbose:            -1
```

#### Post-tuning Training
After 50 Optuna trials, take the best params and retrain on **train + val combined** for final model. Test set remains untouched throughout.

---

### Neural Network Models (MLP, EEN, TabTransformer)

**Approach:** Fixed architecture from `ARCHITECTURE.md` + small grid search on learning rate and dropout. No Optuna for NNs (too slow per trial).

#### MLP Grid Search
```yaml
learning_rate:   [1e-3, 3e-4, 1e-4]
dropout:         [0.0, 0.3, 0.5]
weight_decay:    [1e-5, 1e-4]  # AdamW regularization
# Fixed:
hidden_dims:     [512, 256, 128]
batch_size:      1024
epochs:          50
patience:        10
```
Total: 3 × 3 × 2 = 18 combinations. Run each for full epochs with early stopping.

#### Entity Embedding Network Grid Search
```yaml
learning_rate:   [1e-3, 3e-4]
dropout:         [0.3, 0.5]
# Fixed:
hidden_dims:     [256, 128]   # after embedding concat
batch_size:      1024
epochs:          50
patience:        10
```
Total: 4 combinations.

#### TabTransformer Grid Search
```yaml
learning_rate:   [1e-3, 3e-4]
dropout:         [0.1, 0.3]
# Fixed (from paper defaults):
d_model:         32
n_heads:         8
n_layers:        6
ff_dim:          128   # 4 * d_model
batch_size:      512   # smaller due to transformer memory
epochs:          30
patience:        5
```
Total: 4 combinations.

#### Training Infrastructure
- Device: CUDA if available, else CPU.
- Mixed precision: `torch.cuda.amp.autocast()` if CUDA.
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)`.
- Val AUC evaluated every epoch; best checkpoint saved in memory.

---

## 5. MLflow Tracking Schema

See `MLOPS_SPEC.md` for the full schema. Experiment naming convention:
- Trees: `experiment_name = "home_credit_trees"`
- NNs: `experiment_name = "home_credit_nn"`

Every run logs: `model_type`, all hyperparams, `val_roc_auc`, `test_roc_auc`, model artifact path.

---

## 6. Statistical Comparison Protocol

### Wilcoxon Signed-Rank Test
Used to determine if performance differences are statistically significant.

```python
# 5-fold stratified CV on test set only (not retraining — scoring existing models)
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_a = []
auc_b = []
for train_idx, fold_idx in skf.split(X_test, y_test):
    fold_proba_a = model_a.predict_proba(X_test.iloc[fold_idx])
    fold_proba_b = model_b.predict_proba(X_test.iloc[fold_idx])
    auc_a.append(roc_auc_score(y_test.iloc[fold_idx], fold_proba_a))
    auc_b.append(roc_auc_score(y_test.iloc[fold_idx], fold_proba_b))

stat, p_value = wilcoxon(auc_a, auc_b, alternative="two-sided")
```

Threshold: p < 0.05 = statistically significant difference.

### Comparison Pairs to Test
1. Best tree (LightGBM expected) vs. Best NN (EEN expected)
2. XGBoost vs. LightGBM
3. MLP vs. Entity Embedding Network

---

## 7. Post-Training Analysis

After comparison table is complete:

1. **ROC Curve Plot** — all 5 models on one axis; AUC in legend.
2. **Calibration Plot** — reliability diagram (fraction positive vs. mean predicted prob) for all 5 models in 10 bins.
3. **Feature Importance** — top 30 features for XGBoost (gain) and LightGBM (split count).
4. **Error Analysis** — for best tree vs. best NN:
   - False positive rate at 10%, 20%, 30% recall thresholds
   - Venn diagram of misclassified applicants (how much overlap?)
   - Subgroup AUC: by `CODE_GENDER`, `NAME_EDUCATION_TYPE`, `AMT_CREDIT` quartile
5. **Result Writeup** — in `notebooks/03_model_comparison.ipynb`:
   - 1-paragraph summary of findings
   - Hypothesis for why result came out this way (trees better / NNs better / tied)
   - Recommendation for production model

---

## 8. Serving Model Selection

The model promoted to "Production" in MLflow is chosen by:
1. Highest test ROC-AUC (primary)
2. In case of tie (< 0.002 difference): prefer lower Brier score (better calibration)
3. In case of still tied: prefer tree model (faster inference, easier debugging)

`scripts/promote_model.sh` takes `--run-id` and `--model-name` and calls MLflow registry API.
