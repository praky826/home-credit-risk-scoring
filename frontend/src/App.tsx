import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Award,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  HeartPulse,
  LayoutGrid,
  Loader2,
  Plus,
  X,
} from "lucide-react";
import { getDrift, getHealth, predictRisk } from "./api/client";
import { ModelComparisonView } from "./components/ModelComparisonView";
import { optionalFeatureCatalog } from "./domain/featureCatalog";
import { mapFormToPredictPayload } from "./domain/inputMapping";
import { formatProbability, getRiskLabel } from "./domain/riskLabels";
import { hasValidationErrors, validateApplicantForm, type ValidationErrors } from "./domain/validation";
import type {
  ApplicantFormValues,
  DriftResponse,
  FeatureCatalogItem,
  FieldOption,
  HealthResponse,
  OptionalFieldKey,
  PredictResponse,
} from "./domain/types";

function calculateAmortizedInstallment(loanAmount: number, termMonths: number, annualApr = 0.115): number {
  if (loanAmount <= 0 || termMonths <= 0) return 0;
  const monthlyRate = annualApr / 12;
  const factor = Math.pow(1 + monthlyRate, termMonths);
  return Math.round((loanAmount * (monthlyRate * factor) / (factor - 1)) * 100) / 100;
}

const initialValues: ApplicantFormValues = {
  loanId: "999999",
  annualIncome: "72000",
  loanAmount: "12000",
  term: "36",
  installment: "395.79",
  dti: "18.4",
  ficoRangeLow: "",
  ficoRangeHigh: "",
  empLength: "5",
  purpose: "debt_consolidation",
  homeOwnership: "RENT",
  verificationStatus: "",
  delinq2Yrs: "0",
  inquiriesLast6Months: "1",
  publicRecords: "0",
  revolvingBalance: "",
  revolvingUtilization: "42.3",
  openAccounts: "",
  totalAccounts: "",
  mortgageAccounts: "",
  publicRecordBankruptcies: "",
};

function App() {
  const [activeTab, setActiveTab] = useState<"checker" | "comparison">("checker");
  const [values, setValues] = useState<ApplicantFormValues>(initialValues);
  const [errors, setErrors] = useState<ValidationErrors>({});
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState("");
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [predictError, setPredictError] = useState("");
  const [drift, setDrift] = useState<DriftResponse | null>(null);
  const [driftMessage, setDriftMessage] = useState("");
  const [isPredicting, setIsPredicting] = useState(false);
  const [selectedOptionalFields, setSelectedOptionalFields] = useState<OptionalFieldKey[]>([]);
  const [showTechnical, setShowTechnical] = useState(false);


  const modelReady = Boolean(health?.model_loaded);
  const missingRecommendedCount = [values.empLength, values.purpose, values.homeOwnership, values.revolvingUtilization].filter(
    (value) => value.trim() === "",
  ).length;
  const estimateStrength = missingRecommendedCount === 0 ? "Stronger estimate" : "Basic estimate";
  const availableOptionalFields = optionalFeatureCatalog.filter(
    (feature) =>
      !selectedOptionalFields.includes(feature.key) &&
      !["purpose", "homeOwnership"].includes(feature.key),
  );
  const riskLabel = useMemo(
    () => (prediction ? getRiskLabel(prediction.default_probability) : null),
    [prediction],
  );

  useEffect(() => {
    refreshHealth();
  }, []);

  async function refreshHealth() {
    setHealthError("");

    try {
      const response = await getHealth();
      setHealth(response);
    } catch {
      setHealth(null);
      setHealthError("Prediction service is offline.");
    }
  }

  async function refreshDrift(minRecords = 1) {
    setDriftMessage("");

    try {
      const response = await getDrift(minRecords);
      setDrift(response);
    } catch (error) {
      setDrift(null);
      const message = error instanceof Error ? error.message : "";

      if (message.toLowerCase().includes("insufficient data")) {
        setDriftMessage("Not enough recent checks yet.");
      } else {
        setDriftMessage("System drift status is unavailable.");
      }
    }
  }

  function updateField<K extends keyof ApplicantFormValues>(field: K, value: ApplicantFormValues[K]) {
    setValues((current) => {
      const next = { ...current, [field]: value };
      if (field === "loanAmount" || field === "term") {
        const loan = Number(field === "loanAmount" ? value : next.loanAmount);
        const term = Number(field === "term" ? value : next.term);
        if (loan > 0 && term > 0) {
          next.installment = String(calculateAmortizedInstallment(loan, term));
        }
      }
      return next;
    });
    setErrors((current) => ({
      ...current,
      [field]: undefined,
      ...(field === "loanAmount" || field === "term" ? { installment: undefined } : {}),
    }));
  }

  function addOptionalField(fieldKey: OptionalFieldKey) {
    if (!fieldKey || selectedOptionalFields.includes(fieldKey)) {
      return;
    }

    setSelectedOptionalFields((current) => [...current, fieldKey]);
  }

  function removeOptionalField(fieldKey: OptionalFieldKey) {
    setSelectedOptionalFields((current) => current.filter((key) => key !== fieldKey));
    updateField(fieldKey, "" as ApplicantFormValues[typeof fieldKey]);
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextErrors = validateApplicantForm(values);
    setErrors(nextErrors);

    if (hasValidationErrors(nextErrors)) {
      return;
    }

    setIsPredicting(true);
    setPredictError("");
    setPrediction(null);

    try {
      const payload = mapFormToPredictPayload(values);
      const response = await predictRisk(payload);
      setPrediction(response);
      await refreshDrift(1);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Prediction failed.";
      setPredictError(message);
    } finally {
      setIsPredicting(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Production ML Suite</p>
          <h1>LendingClub Risk Intelligence</h1>
        </div>
        <div className="topbar-right">
          <nav className="nav-tabs" aria-label="Main Navigation">
            <button
              type="button"
              className={`nav-tab-btn ${activeTab === "checker" ? "active" : ""}`}
              onClick={() => setActiveTab("checker")}
            >
              <HeartPulse size={16} />
              <span>Loan Risk Checker</span>
            </button>
            <button
              type="button"
              className={`nav-tab-btn ${activeTab === "comparison" ? "active" : ""}`}
              onClick={() => setActiveTab("comparison")}
            >
              <Award size={16} />
              <span>Model Comparison & Selection</span>
            </button>
          </nav>
          <HealthBadge health={health} error={healthError} onRefresh={refreshHealth} />
        </div>
      </header>

      {activeTab === "comparison" ? (
        <ModelComparisonView onBackToChecker={() => setActiveTab("checker")} />
      ) : (
        <section className="workspace">
          <form className="panel form-panel" onSubmit={handleSubmit}>
            <div className="panel-header">
              <div>
                <p className="eyebrow">Applicant</p>
                <h2>Application Details</h2>
              </div>
              <span className="required-note">Required fields marked</span>
            </div>

            <div className="field-grid">
              <Field label="Annual income ($)" value={values.annualIncome} error={errors.annualIncome} required onChange={(value) => updateField("annualIncome", value)} />
              <Field label="Requested loan amount ($500 - $40,000)" value={values.loanAmount} error={errors.loanAmount} required onChange={(value) => updateField("loanAmount", value)} />
              <SelectField
                label="Loan term"
                value={values.term}
                options={[
                  { label: "36 months", value: "36" },
                  { label: "60 months", value: "60" },
                ]}
                onChange={(value) => updateField("term", value)}
              />
              <Field
                label="Monthly installment ($)"
                value={values.installment}
                error={errors.installment}
                hint="Auto-amortized baseline payment"
                onChange={(value) => updateField("installment", value)}
              />
              <Field label="Debt-to-income ratio (DTI %)" value={values.dti} error={errors.dti} required onChange={(value) => updateField("dti", value)} />
              <Field label="Loan ID (Lookup)" value={values.loanId} error={errors.loanId} onChange={(value) => updateField("loanId", value)} />
            </div>

            <section className="quality-section">
              <div>
                <p className="eyebrow">Traditional Credit Bureau (Optional)</p>
                <h3>Credit Scores</h3>
              </div>
              <p className="quality-copy">
                Optional for collateral-free cashflow underwriting. Leave blank if evaluating thin-file or unbanked borrowers.
              </p>
              <div className="field-grid">
                <Field label="FICO range low (300 - 850)" value={values.ficoRangeLow} error={errors.ficoRangeLow} onChange={(value) => updateField("ficoRangeLow", value)} />
                <Field label="FICO range high (300 - 850)" value={values.ficoRangeHigh} error={errors.ficoRangeHigh} onChange={(value) => updateField("ficoRangeHigh", value)} />
              </div>
            </section>

            <section className="quality-section">
              <div>
                <p className="eyebrow">Recommended</p>
                <h3>Better estimate inputs</h3>
              </div>
              <p className="quality-copy">These are optional, but they help the backend build stronger derived features.</p>
              <div className="field-grid">
                <Field label="Employment length" value={values.empLength} error={errors.empLength} onChange={(value) => updateField("empLength", value)} />
                <SelectField
                  label="Loan purpose"
                  value={values.purpose}
                  options={optionalFeatureCatalog.find((item) => item.key === "purpose")?.options ?? []}
                  onChange={(value) => updateField("purpose", value)}
                />
                <SelectField
                  label="Home ownership"
                  value={values.homeOwnership}
                  options={optionalFeatureCatalog.find((item) => item.key === "homeOwnership")?.options ?? []}
                  onChange={(value) => updateField("homeOwnership", value)}
                />
                <Field label="Revolving utilization" value={values.revolvingUtilization} error={errors.revolvingUtilization} onChange={(value) => updateField("revolvingUtilization", value)} />
              </div>
            </section>

            <section className="optional-section">
              <div className="optional-header">
                <div>
                  <p className="eyebrow">Optional inputs</p>
                  <h3>Add more details</h3>
                </div>
                <label className="add-input-control">
                  <Plus size={18} />
                  <select
                    value=""
                    disabled={availableOptionalFields.length === 0}
                    onChange={(event) => addOptionalField(event.target.value as OptionalFieldKey)}
                  >
                    <option value="">{availableOptionalFields.length === 0 ? "All inputs added" : "Add input"}</option>
                    {availableOptionalFields.map((feature) => (
                      <option key={feature.key} value={feature.key}>
                        {feature.label}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              {selectedOptionalFields.length > 0 && (
                <div className="field-grid more-grid">
                  {selectedOptionalFields.map((fieldKey) => {
                    const feature = optionalFeatureCatalog.find((item) => item.key === fieldKey);
                    if (!feature) {
                      return null;
                    }

                    return (
                      <DynamicOptionalField
                        key={feature.key}
                        feature={feature}
                        value={values[feature.key]}
                        error={errors[feature.key]}
                        onChange={(value) => updateField(feature.key, value as never)}
                        onRemove={() => removeOptionalField(feature.key)}
                      />
                    );
                  })}
                </div>
              )}
            </section>

            {predictError && <p className="error-banner">{predictError}</p>}

            <button className="primary-button" type="submit" disabled={!modelReady || isPredicting}>
              {isPredicting ? <Loader2 className="spin" size={18} /> : <HeartPulse size={18} />}
              {isPredicting ? "Checking" : "Check Risk"}
            </button>
          </form>

          <aside className="result-stack">
            <section className={`panel result-panel ${riskLabel ? `risk-${riskLabel.level}` : ""}`}>
              <p className="eyebrow">Risk estimate</p>
              {prediction && riskLabel ? (
                <>
                  <div className="risk-header-row">
                    <h2>{prediction.decision_label || riskLabel.title}</h2>
                    {prediction.feature_lookup && (
                      <span className={`lookup-badge ${prediction.feature_lookup.found ? "found" : "synthetic"}`}>
                        {prediction.feature_lookup.found
                          ? `Record Found (ID ${prediction.feature_lookup.customer_id})`
                          : "Synthetic Profile"}
                      </span>
                    )}
                  </div>
                  <p className="probability">{formatProbability(prediction.default_probability)}</p>
                  <p className="muted">{riskLabel.message}</p>
                  <p className="estimate-note">{estimateStrength}</p>

                  {prediction.solvency && (
                    <div className="solvency-panel" style={{ marginTop: "1rem", padding: "0.85rem", background: "#f8fafc", borderRadius: "8px", border: "1px solid #e2e8f0" }}>
                      <div style={{ fontSize: "0.75rem", fontWeight: 700, textTransform: "uppercase", color: "#64748b", marginBottom: "0.5rem" }}>
                        Cashflow & Solvency Breakdown
                      </div>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", fontSize: "0.8rem" }}>
                        <div>
                          <span style={{ color: "#64748b" }}>Disposable Cashflow: </span>
                          <strong style={{ color: prediction.solvency.disposable_monthly_cashflow < 0 ? "#dc2626" : "#16a34a" }}>
                            ${prediction.solvency.disposable_monthly_cashflow.toLocaleString()}/mo
                          </strong>
                        </div>
                        <div>
                          <span style={{ color: "#64748b" }}>Debt Service Burden: </span>
                          <strong style={{ color: prediction.solvency.debt_service_burden_pct > 40 ? "#dc2626" : "#334155" }}>
                            {prediction.solvency.debt_service_burden_pct.toFixed(1)}%
                          </strong>
                        </div>
                        <div>
                          <span style={{ color: "#64748b" }}>Repayment Coverage: </span>
                          <strong style={{ color: prediction.solvency.repayment_coverage_ratio < 1.0 ? "#dc2626" : "#334155" }}>
                            {prediction.solvency.repayment_coverage_ratio.toFixed(2)}x
                          </strong>
                        </div>
                        <div>
                          <span style={{ color: "#64748b" }}>Salary Multiplier: </span>
                          <strong style={{ color: prediction.solvency.salary_multiplier > 2.5 ? "#d97706" : "#334155" }}>
                            {prediction.solvency.salary_multiplier.toFixed(2)}x
                          </strong>
                        </div>
                      </div>
                    </div>
                  )}

                  {prediction.warnings && prediction.warnings.length > 0 && (
                    <div className="ood-warnings-panel">
                      <div className="ood-warnings-title">
                        <AlertTriangle size={14} />
                        <span>Underwriting Diagnostics</span>
                      </div>
                      {prediction.warnings.map((w, idx) => (
                        <div key={idx} className="ood-warning-item">
                          {w.field ? <strong>{w.field}: </strong> : null}
                          {w.message}
                        </div>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <>
                  <h2>Ready for a check</h2>
                  <p className="muted">Enter applicant details and run a risk estimate.</p>
                </>
              )}
            </section>

            <section className="panel status-panel">
              <div className="panel-header compact">
                <div>
                  <p className="eyebrow">System</p>
                  <h2>Status</h2>
                </div>
                <button type="button" className="ghost-button" onClick={() => refreshDrift(1)}>
                  Refresh
                </button>
              </div>
              <StatusLine
                label="Model"
                value={
                  modelReady
                    ? health?.model_name || "LendingClub XGBoost Default Risk Classifier"
                    : healthError
                    ? "Offline"
                    : "Degraded"
                }
              />
              <StatusLine label="Drift" value={drift ? (drift.alert ? "Needs technical review" : "Normal") : driftMessage || "Not checked yet"} />
            </section>

            <section className="panel technical-panel">
              <button type="button" className="toggle-row technical-toggle" onClick={() => setShowTechnical((current) => !current)}>
                <span>Technical details</span>
                {showTechnical ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
              </button>
              {showTechnical && (
                <div className="technical-content">
                  <StatusLine
                    label="Model name"
                    value={prediction?.model_name || health?.model_name || "LendingClub XGBoost Default Risk Classifier"}
                  />
                  <StatusLine label="Model version" value={prediction?.model_version || health?.model_version || "Unavailable"} />
                  <StatusLine label="Model type" value={health?.model_type ? health.model_type.toUpperCase() : "XGBOOST GBDT"} />
                  <StatusLine label="Registry stage" value={health?.model_stage || "Production"} />
                  <StatusLine label="Request ID" value={prediction?.request_id || "No request yet"} />
                  <StatusLine
                    label="Calibrated probability"
                    value={prediction ? formatProbability(prediction.default_probability) : "No request yet"}
                  />
                  <StatusLine
                    label="Raw tree probability"
                    value={
                      prediction?.raw_model_probability != null
                        ? `${(prediction.raw_model_probability * 100).toFixed(2)}%`
                        : "No request yet"
                    }
                  />
                  <StatusLine
                    label="Calibration method"
                    value={prediction?.calibration_method || "scale_pos_weight_unbias"}
                  />
                  <StatusLine
                    label="Decision label"
                    value={prediction?.decision_label || "Pending"}
                  />
                  <StatusLine
                    label="Feature baseline"
                    value={
                      prediction?.feature_lookup?.found
                        ? `Feature Store (${prediction.feature_lookup.source})`
                        : "Clean-slate defaults"
                    }
                  />
                  <StatusLine label="Drift records" value={drift ? String(drift.n_records_evaluated) : "Unavailable"} />
                  <StatusLine label="Drift threshold" value={drift ? String(drift.threshold) : "Unavailable"} />
                  {drift && <TopPsiValues values={drift.psi_values} />}
                  <button
                    type="button"
                    className="view-comparison-btn"
                    onClick={() => setActiveTab("comparison")}
                  >
                    <Award size={15} />
                    <span>View Model Benchmark & Rationale →</span>
                  </button>
                </div>
              )}
            </section>
          </aside>
        </section>
      )}

    </main>
  );
}

type FieldProps = {
  label: string;
  value: string;
  error?: string;
  required?: boolean;
  hint?: string;
  onChange: (value: string) => void;
};

function Field({ label, value, error, required, hint, onChange }: FieldProps) {
  return (
    <label className="field">
      <span>
        {label}
        {required && <strong> *</strong>}
      </span>
      <input value={value} inputMode="decimal" onChange={(event) => onChange(event.target.value)} />
      {hint && !error && <small style={{ color: "#64748b", fontSize: "0.75rem", marginTop: "2px" }}>{hint}</small>}
      {error && <small>{error}</small>}
    </label>
  );
}

type SelectFieldProps = {
  label: string;
  value: string;
  options: FieldOption[];
  onChange: (value: string) => void;
};

function SelectField({ label, value, options, onChange }: SelectFieldProps) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

type DynamicOptionalFieldProps = {
  feature: FeatureCatalogItem;
  value: string;
  error?: string;
  onChange: (value: string) => void;
  onRemove: () => void;
};

function DynamicOptionalField({ feature, value, error, onChange, onRemove }: DynamicOptionalFieldProps) {
  return (
    <div className="dynamic-field">
      <button type="button" className="remove-field-button" onClick={onRemove} aria-label={`Remove ${feature.label}`}>
        <X size={16} />
      </button>
      {feature.kind === "select" ? (
        <SelectField label={feature.label} value={value} options={feature.options ?? []} onChange={onChange} />
      ) : (
        <Field label={feature.label} value={value} error={error} onChange={onChange} />
      )}
    </div>
  );
}

type HealthBadgeProps = {
  health: HealthResponse | null;
  error: string;
  onRefresh: () => void;
};

function HealthBadge({ health, error, onRefresh }: HealthBadgeProps) {
  const loaded = Boolean(health?.model_loaded);
  const label = loaded ? "Model Online" : error ? "API Offline" : "Model Degraded";

  return (
    <button type="button" className={`health-badge ${loaded ? "is-online" : "is-offline"}`} onClick={onRefresh}>
      {loaded ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}
      <span>{label}</span>
    </button>
  );
}

function StatusLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="status-line">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function TopPsiValues({ values }: { values: Record<string, number> }) {
  const topValues = Object.entries(values)
    .sort(([, left], [, right]) => right - left)
    .slice(0, 5);

  if (topValues.length === 0) {
    return <p className="muted">No PSI values available.</p>;
  }

  return (
    <div className="psi-list">
      <p className="eyebrow">Top PSI values</p>
      {topValues.map(([feature, value]) => (
        <StatusLine key={feature} label={feature} value={value.toFixed(3)} />
      ))}
    </div>
  );
}

export default App;
