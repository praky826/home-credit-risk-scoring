export type HealthResponse = {
  status: string;
  model_loaded: boolean;
  model_version: string;
  model_name?: string;
  model_type?: string;
  model_stage?: string;
};

export type PredictionWarning = {
  code: string;
  field?: string;
  message: string;
};

export type FeatureLookupMetadata = {
  customer_id?: number;
  found: boolean;
  source?: string;
};

export type SolvencyDiagnostics = {
  repayment_coverage_ratio: number;
  disposable_monthly_cashflow: number;
  debt_service_burden_pct: number;
  salary_multiplier: number;
  amortized_installment: number;
  policy_knockout: boolean;
  policy_reasons: string[];
};

export type PredictResponse = {
  default_probability: number;
  raw_model_probability?: number;
  calibration_method?: string;
  decision_label?: string;
  model_version: string;
  model_name?: string;
  request_id: string;
  feature_lookup?: FeatureLookupMetadata;
  solvency?: SolvencyDiagnostics;
  warnings?: PredictionWarning[];
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
  loanId: string;
  annualIncome: string;
  loanAmount: string;
  term: string;
  installment: string;
  dti: string;
  ficoRangeLow: string;
  ficoRangeHigh: string;
  empLength: string;
  purpose: string;
  homeOwnership: string;
  verificationStatus: string;
  delinq2Yrs: string;
  inquiriesLast6Months: string;
  publicRecords: string;
  revolvingBalance: string;
  revolvingUtilization: string;
  openAccounts: string;
  totalAccounts: string;
  mortgageAccounts: string;
  publicRecordBankruptcies: string;
};

export type OptionalFieldKey =
  | "homeOwnership"
  | "verificationStatus"
  | "purpose"
  | "delinq2Yrs"
  | "inquiriesLast6Months"
  | "publicRecords"
  | "revolvingBalance"
  | "revolvingUtilization"
  | "openAccounts"
  | "totalAccounts"
  | "mortgageAccounts"
  | "publicRecordBankruptcies";

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
  loan_id?: number;
  loan_amnt: number;
  annual_inc: number;
  term: number;
  installment: number;
  dti: number;
  fico_range_low?: number;
  fico_range_high?: number;
  emp_length?: number;
  purpose?: string;
  home_ownership?: string;
  verification_status?: string;
  delinq_2yrs?: number;
  inq_last_6mths?: number;
  pub_rec?: number;
  revol_bal?: number;
  revol_util?: number;
  open_acc?: number;
  total_acc?: number;
  mort_acc?: number;
  pub_rec_bankruptcies?: number;
};

export type RiskLevel = "low" | "review" | "high";

export type RiskLabel = {
  level: RiskLevel;
  title: string;
  message: string;
};
