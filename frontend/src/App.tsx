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

const initialValues: ApplicantFormValues = {
  customerId: "999999",
  ageYears: "41",
  annualIncome: "150000",
  loanAmount: "500000",
  yearsEmployed: "5.5",
  externalScore2: "0.5",
  annuityAmount: "",
  goodsPrice: "",
  externalScore1: "",
  externalScore3: "",
  gender: "",
  ownsCar: "",
  ownsProperty: "",
  childrenCount: "",
  familyMembers: "",
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
  const missingRecommendedCount = [
    values.annuityAmount,
    values.externalScore1,
    values.externalScore2,
    values.externalScore3,
  ].filter((value) => value.trim() === "").length;
  const estimateStrength = missingRecommendedCount === 0 ? "Stronger estimate" : "Basic estimate";
  const availableOptionalFields = optionalFeatureCatalog.filter(
    (feature) => !selectedOptionalFields.includes(feature.key),
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
    setValues((current) => ({ ...current, [field]: value }));
    setErrors((current) => ({ ...current, [field]: undefined }));
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
          <h1>Home Credit Risk Intelligence</h1>
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
              <Field label="Customer ID" value={values.customerId} error={errors.customerId} onChange={(value) => updateField("customerId", value)} />
              <Field label="Age" value={values.ageYears} error={errors.ageYears} required onChange={(value) => updateField("ageYears", value)} />
              <Field label="Annual income" value={values.annualIncome} error={errors.annualIncome} required onChange={(value) => updateField("annualIncome", value)} />
              <Field label="Requested loan amount" value={values.loanAmount} error={errors.loanAmount} required onChange={(value) => updateField("loanAmount", value)} />
              <Field label="Years employed" value={values.yearsEmployed} error={errors.yearsEmployed} required onChange={(value) => updateField("yearsEmployed", value)} />
            </div>

            <section className="quality-section">
              <div>
                <p className="eyebrow">Recommended</p>
                <h3>Better estimate inputs</h3>
              </div>
              <p className="quality-copy">These are optional, but they help the backend build stronger derived features.</p>
              <div className="field-grid">
                <Field label="Annuity amount (Loan payment)" value={values.annuityAmount} error={errors.annuityAmount} onChange={(value) => updateField("annuityAmount", value)} />
                <Field label="Credit Bureau Score 1 (Alternative, 0.0 - 1.0)" value={values.externalScore1} error={errors.externalScore1} onChange={(value) => updateField("externalScore1", value)} />
                <Field label="Credit Bureau Score 2 (Primary, 0.0 - 1.0)" value={values.externalScore2} error={errors.externalScore2} onChange={(value) => updateField("externalScore2", value)} />
                <Field label="Credit Bureau Score 3 (Secondary, 0.0 - 1.0)" value={values.externalScore3} error={errors.externalScore3} onChange={(value) => updateField("externalScore3", value)} />
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
                  <h2>{riskLabel.title}</h2>
                  <p className="probability">{formatProbability(prediction.default_probability)}</p>
                  <p className="muted">{riskLabel.message}</p>
                  <p className="estimate-note">{estimateStrength}</p>
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
                    ? health?.model_name || "XGBoost Risk Classifier (Production Champion)"
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
                    value={prediction?.model_name || health?.model_name || "XGBoost Risk Classifier (Production Champion)"}
                  />
                  <StatusLine label="Model version" value={prediction?.model_version || health?.model_version || "Unavailable"} />
                  <StatusLine label="Model type" value={health?.model_type ? health.model_type.toUpperCase() : "XGBOOST GBDT"} />
                  <StatusLine label="Registry stage" value={health?.model_stage || "Production"} />
                  <StatusLine label="Request ID" value={prediction?.request_id || "No request yet"} />
                  <StatusLine label="Raw probability" value={prediction ? String(prediction.default_probability) : "No request yet"} />
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
  onChange: (value: string) => void;
};

function Field({ label, value, error, required, onChange }: FieldProps) {
  return (
    <label className="field">
      <span>
        {label}
        {required && <strong> *</strong>}
      </span>
      <input value={value} inputMode="decimal" onChange={(event) => onChange(event.target.value)} />
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
