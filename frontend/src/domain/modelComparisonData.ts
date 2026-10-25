import type { ModelComparisonResponse } from "./types";

export const defaultModelComparison: ModelComparisonResponse = {
  champion_model_id: "lgbm",
  champion_model_name: "LendingClub LightGBM Default Risk Classifier",
  models: [
    {
      model_id: "xgboost",
      name: "LendingClub XGBoost Default Risk Classifier",
      architecture: "Extreme Gradient Boosted Trees (GPU baseline)",
      roc_auc: 0.70487,
      ks_stat: 0.29824,
      brier_score: 0.24073,
      calibration_slope: 0.67559,
      inference_latency_ms: 1.2,
      status: "GPU Baseline",
      is_champion: false,
    },
    {
      model_id: "lgbm",
      name: "LendingClub LightGBM Default Risk Classifier",
      architecture: "Leaf-wise GBDT (GPU baseline)",
      roc_auc: 0.70616,
      ks_stat: 0.29888,
      brier_score: 0.24802,
      calibration_slope: 0.66812,
      inference_latency_ms: 1.0,
      status: "Current Production Champion",
      is_champion: true,
    },
    {
      model_id: "entity_embedding",
      name: "LendingClub Entity Embedding Neural Network",
      architecture: "Learned Categorical Embeddings + Dense Residual MLP",
      roc_auc: 0.70679,
      ks_stat: 0.29967,
      brier_score: 0.26739,
      calibration_slope: 0.67625,
      inference_latency_ms: 5.0,
      status: "Neural Challenger",
      is_champion: false,
    },
    {
      model_id: "mlp",
      name: "LendingClub Multi-Layer Perceptron",
      architecture: "Fully Connected Dense Network (512-256-128 + BatchNorm)",
      roc_auc: 0.69755,
      ks_stat: 0.28861,
      brier_score: 0.23261,
      calibration_slope: 0.71692,
      inference_latency_ms: 4.0,
      status: "Neural Baseline",
      is_champion: false,
    },
    {
      model_id: "tab_transformer",
      name: "LendingClub TabTransformer Attention Network",
      architecture: "Multi-Head Column Self-Attention + Continuous MLP",
      roc_auc: 0.65670,
      ks_stat: 0.22281,
      brier_score: 0.21660,
      calibration_slope: 0.66458,
      inference_latency_ms: 12.0,
      status: "Experimental Neural Model",
      is_champion: false,
    },
  ],
  selection_rationales: [
    {
      title: "LightGBM Selected for Production",
      summary: "LightGBM is the selected production model for the LendingClub application.",
      details:
        "LightGBM provides nearly identical ranking performance to Entity Embedding while offering substantially better Brier score and lower inference latency. It is selected as the practical production champion; Entity Embedding remains a neural challenger.",
    },
    {
      title: "Initial Model Family",
      summary: "Tree ensembles remain the best first pass for tabular credit-risk data.",
      details:
        "All five model families have now been retrained on the LendingClub feature matrix. LightGBM is selected for production because it offers the best operational balance of ranking, calibration, and latency.",
    },
    {
      title: "Leakage Control",
      summary: "Post-origination repayment fields are excluded from the new feature contract.",
      details:
        "The migration keeps application/origination fields such as income, DTI, loan amount, term, employment length, and FICO range while excluding repayment and recovery fields.",
    },
    {
      title: "Temporal Evaluation",
      summary: "LendingClub loans should be evaluated with newer loans held out.",
      details:
        "The feature pipeline sorts by issue date before writing processed features so the training split approximates a realistic forward-looking validation setup.",
    },
    {
      title: "Calibration Pending",
      summary: "Decision thresholds and class weights must be recomputed after training.",
      details:
        "The serving layer now uses the LightGBM artifact and a LendingClub class-weight correction from the first temporal training split. Risk thresholds should be revisited after calibration analysis.",
    },
  ],
  significance_tests: [
    {
      comparison: "LightGBM vs Entity Embedding",
      statistic: 1.0,
      p_value: 1,
      significant: false,
      interpretation:
        "Pending repeated validation after hyperparameter tuning.",
    },
    {
      comparison: "XGBoost vs LightGBM",
      statistic: 8.0,
      p_value: 1,
      significant: false,
      interpretation:
        "First-pass holdout metrics are nearly tied; repeat after tuning.",
    },
    {
      comparison: "MLP vs Entity Embedding",
      statistic: 0.0,
      p_value: 1,
      significant: false,
      interpretation:
        "Entity Embedding remains the neural challenger while LightGBM is used for production serving.",
    },
  ],
  evaluation_dataset: {
    train_samples: 970631,
    val_samples: 121328,
    test_samples: 121330,
    features_evaluated: 30,
    evaluation_protocol:
      "Temporal holdout using LendingClub issue-date ordering before feature export",
  },
};
