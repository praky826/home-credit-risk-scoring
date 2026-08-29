export type HealthResponse = {
  status: string;
  model_loaded: boolean;
  model_version: string;
  model_name?: string;
  model_type?: string;
  model_stage?: string;
};

export type PredictResponse = {
  default_probability: number;
  model_version: string;
  model_name?: string;
  request_id: string;
};

export type DriftResponse = {
  psi_values: Record<string, number>;
  n_records_evaluated: number;
  alert: boolean;
  threshold: number;
};

export type ModelBenchmarkItem = {
  model_id: string;
  name: string;
  architecture: string;
  roc_auc: number;
  ks_stat: number;
  brier_score: number;
  calibration_slope: number;
  inference_latency_ms: number;
  status: string;
  is_champion: boolean;
};

export type WilcoxonResultItem = {
  comparison: string;
  statistic: number;
  p_value: number;
  significant: boolean;
  interpretation: string;
};

export type SelectionRationaleItem = {
  title: string;
  summary: string;
  details: string;
};

export type ModelComparisonResponse = {
  champion_model_id: string;
  champion_model_name: string;
  models: ModelBenchmarkItem[];
  selection_rationales: SelectionRationaleItem[];
  significance_tests: WilcoxonResultItem[];
  evaluation_dataset: {
    train_samples?: number;
    val_samples?: number;
    test_samples?: number;
    features_evaluated?: number;
    evaluation_protocol?: string;
  };
};


export type ApplicantFormValues = {
  customerId: string;
  ageYears: string;
  annualIncome: string;
  loanAmount: string;
  yearsEmployed: string;
  externalScore2: string;
  annuityAmount: string;
  goodsPrice: string;
  externalScore1: string;
  externalScore3: string;
  gender: "" | "F" | "M";
  ownsCar: "" | "Y" | "N";
  ownsProperty: "" | "Y" | "N";
  childrenCount: string;
  familyMembers: string;
};

export type OptionalFieldKey =
  | "goodsPrice"
  | "gender"
  | "ownsCar"
  | "ownsProperty"
  | "familyMembers"
  | "childrenCount";

export type FieldOption = {
  value: string;
  label: string;
};

export type FeatureCatalogItem = {
  key: OptionalFieldKey;
  label: string;
  importance: number;
  kind: "number" | "select";
  options?: FieldOption[];
};

export type PredictPayload = {
  SK_ID_CURR?: number;
  AMT_CREDIT: number;
  AMT_INCOME_TOTAL: number;
  DAYS_BIRTH: number;
  DAYS_EMPLOYED: number;
  EXT_SOURCE_2?: number;
  AMT_ANNUITY?: number;
  AMT_GOODS_PRICE?: number;
  EXT_SOURCE_1?: number;
  EXT_SOURCE_3?: number;
  CODE_GENDER?: number;
  FLAG_OWN_CAR?: number;
  FLAG_OWN_REALTY?: number;
  CNT_CHILDREN?: number;
  CNT_FAM_MEMBERS?: number;
};

export type RiskLevel = "low" | "review" | "high";

export type RiskLabel = {
  level: RiskLevel;
  title: string;
  message: string;
};
