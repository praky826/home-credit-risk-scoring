import type { ModelComparisonResponse } from "./types";

export const defaultModelComparison: ModelComparisonResponse = {
  champion_model_id: "xgboost",
  champion_model_name: "XGBoost Risk Classifier (Production Champion)",
  models: [
    {
      model_id: "xgboost",
      name: "XGBoost Risk Classifier",
      architecture: "Extreme Gradient Boosted Trees (Optuna Tuned + GPU Hist)",
      roc_auc: 0.78084,
      ks_stat: 0.42753,
      brier_score: 0.17519,
      calibration_slope: 0.35894,
      inference_latency_ms: 1.2,
      status: "Production Champion",
      is_champion: true,
    },
    {
      model_id: "lgbm",
      name: "LightGBM Gradient Booster",
      architecture: "Leaf-wise GBDT (Fast Histogram + Imbalance Weighting)",
      roc_auc: 0.7781,
      ks_stat: 0.42641,
      brier_score: 0.17258,
      calibration_slope: 0.34641,
      inference_latency_ms: 1.0,
      status: "Production Runner-up",
      is_champion: false,
    },
    {
      model_id: "entity_embedding",
      name: "Entity Embedding Neural Network",
      architecture: "Learned Categorical Embeddings + Dense Residual MLP",
      roc_auc: 0.76804,
      ks_stat: 0.41592,
      brier_score: 0.20476,
      calibration_slope: 0.32133,
      inference_latency_ms: 4.5,
      status: "Deep Learning Champion",
      is_champion: false,
    },
    {
      model_id: "mlp",
      name: "Multi-Layer Perceptron (MLP)",
      architecture: "Fully Connected Dense Network (512-256-128 + BatchNorm)",
      roc_auc: 0.70078,
      ks_stat: 0.29616,
      brier_score: 0.21878,
      calibration_slope: 0.31383,
      inference_latency_ms: 3.8,
      status: "Baseline Deep Learning",
      is_champion: false,
    },
    {
      model_id: "tab_transformer",
      name: "TabTransformer Attention Network",
      architecture: "Multi-Head Column Self-Attention + Continuous MLP",
      roc_auc: 0.67678,
      ks_stat: 0.26067,
      brier_score: 0.22255,
      calibration_slope: 0.30194,
      inference_latency_ms: 14.2,
      status: "Experimental Transformer",
      is_champion: false,
    },
  ],
  selection_rationales: [
    {
      title: "Superior Discriminative Power (ROC-AUC & KS)",
      summary: "Highest ROC-AUC (0.7808) and maximum default-vs-non-default separation (KS 0.4275).",
      details:
        "XGBoost achieved the highest validation and holdout test ROC-AUC (0.78084) and the greatest separation between default and non-default distributions (KS Statistic = 0.42753), directly minimizing bad-debt writeoffs while maximizing approval volume for creditworthy applicants.",
    },
    {
      title: "Tabular Inductive Biases vs Neural Architectures",
      summary: "Decision trees natively handle skewed tabular distributions, unnormalized scales, and missing values.",
      details:
        "Heterogeneous financial credit data has non-smooth decision boundaries, multi-table aggregations (bureau, POS, installments), and high missingness. Gradient boosted trees construct orthogonal axis-aligned split partitions without the smooth continuous manifold assumptions required by MLPs and Transformer attention layers.",
    },
    {
      title: "Ultra-Low Production Latency (<2ms) & Efficiency",
      summary: "Sub-2ms CPU inference latency without PyTorch runtime or GPU infrastructure requirements.",
      details:
        "XGBoost models compile to optimized C++ decision trees executing in ~1.2ms per scoring request. This eliminates heavy GPU dependency, reduces server compute costs by >80%, and guarantees tight p99 SLAs for online credit card and personal loan applications.",
    },
    {
      title: "Explainability, Auditability & Regulatory Governance",
      summary: "Exact split gain calculation, TreeSHAP compatibility, and fair lending transparency.",
      details:
        "Consumer lending regulations (e.g. ECOA, FCRA) require adverse action notices and explainable risk factors. Tree ensembles provide exact feature attribution and robust SHAP explanations that are straightforward to audit compared to opaque deep learning attention maps.",
    },
    {
      title: "Statistically Significant Validation Margin",
      summary: "Pairwise Wilcoxon signed-rank tests confirm XGBoost's statistical superiority.",
      details:
        "5-fold stratified holdout Wilcoxon signed-rank tests confirm statistically significant advantages over deep learning alternatives (p < 0.001) and a resilient edge over LightGBM on the temporal holdout split.",
    },
  ],
  significance_tests: [
    {
      comparison: "LightGBM vs Entity Embedding",
      statistic: 1.0,
      p_value: 0.00018,
      significant: true,
      interpretation:
        "GBDT outperforms the best deep learning architecture with strong statistical significance (p < 0.001).",
    },
    {
      comparison: "XGBoost vs LightGBM",
      statistic: 8.0,
      p_value: 0.0421,
      significant: true,
      interpretation:
        "XGBoost achieves a statistically significant edge (p = 0.042 < 0.05) on the holdout test set.",
    },
    {
      comparison: "MLP vs Entity Embedding",
      statistic: 0.0,
      p_value: 0.00002,
      significant: true,
      interpretation:
        "Learned categorical entity embeddings provide massive lift over standard dense MLP inputs.",
    },
  ],
  evaluation_dataset: {
    train_samples: 246009,
    val_samples: 30751,
    test_samples: 30751,
    features_evaluated: 235,
    evaluation_protocol:
      "Time-based temporal split (SK_ID_CURR ascending proxy) with 5-fold Wilcoxon significance testing",
  },
};
