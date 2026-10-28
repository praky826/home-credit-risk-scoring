import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  Award,
  BarChart3,
  BadgeCheck,
  CalendarDays,
  CheckCircle2,
  Check,
  ChevronDown,
  ChevronUp,
  Circle,
  FileText,
  Gauge,
  HeartPulse,
  LayoutGrid,
  Loader2,
  Plus,
  RefreshCw,
  RotateCcw,
  Save,
  ShieldAlert,
  UserRound,
  WalletCards,
  X,
} from "lucide-react";
import { getDrift, getHealth, predictRisk } from "./api/client";
import { ModelComparisonView } from "./components/ModelComparisonView";
import { optionalFeatureCatalog } from "./domain/featureCatalog";
import { mapFormToPredictPayload } from "./domain/inputMapping";
import { getSelectedModel } from "./domain/modelSelection";
import { formatProbability, getRiskLabel } from "./domain/riskLabels";
import { hasValidationErrors, validateApplicantForm, type ValidationErrors } from "./domain/validation";
import { useAuth } from "./auth/AuthProvider";
import { normalizeLoanId } from "./domain/loanId";
import { loadApplicationState, saveDecision as persistDecision, saveDraft as persistDraft } from "./persistence/underwritingRepository";
import type { DecisionAuditEvent, DecisionValue, PersistedDecision } from "./persistence/types";
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

const workflowStageNames = ["draft", "details_complete", "risk_assessed", "decision_recorded"] as const;

function App() {
  const { signOut } = useAuth();
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
  const resultRef = useRef<HTMLElement>(null);
  const formRef = useRef<HTMLFormElement>(null);
  const [riskOutdated, setRiskOutdated] = useState(false);
  const [lastCalculatedAt, setLastCalculatedAt] = useState<Date | null>(null);
  const [draftMessage, setDraftMessage] = useState("");
  const [officerDecision, setOfficerDecision] = useState<"approve" | "refer" | "decline" | "">("");
  const [overrideReason, setOverrideReason] = useState("");
  const [decisionMessage, setDecisionMessage] = useState("");
  const [persistedDecision, setPersistedDecision] = useState<PersistedDecision | null>(null);
  const [decisionHistory, setDecisionHistory] = useState<DecisionAuditEvent[]>([]);
  const [isLoadingApplication, setIsLoadingApplication] = useState(false);
  const [isSavingDraft, setIsSavingDraft] = useState(false);
  const [isSavingDecision, setIsSavingDecision] = useState(false);
  const [persistenceError, setPersistenceError] = useState("");


  const modelReady = Boolean(health?.model_loaded);
  const validationErrors = validateApplicantForm(values);
  const requiredFields = ["annualIncome", "loanAmount", "term", "dti"] as const;
  const completedRequiredCount = requiredFields.filter((field) => values[field].trim() !== "" && !validationErrors[field]).length;
  const formReady = !hasValidationErrors(validationErrors);
  const selectedModel = getSelectedModel(health);
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
  const recommendation = riskLabel?.level === "low" ? "approve" : riskLabel?.level === "high" ? "decline" : riskLabel ? "refer" : "";
  const decisionRequiresReason = Boolean(recommendation && officerDecision && officerDecision !== recommendation);
  const workflowStage = !prediction ? (formReady ? 0 : 1) : riskOutdated ? 1 : officerDecision ? 3 : 2;

  useEffect(() => {
    refreshHealth();
    void restoreApplication(initialValues.loanId);
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

  async function restoreApplication(rawLoanId: string) {
    let loanId: string;
    try {
      loanId = normalizeLoanId(rawLoanId);
    } catch (error) {
      setPersistenceError(error instanceof Error ? error.message : "Enter a valid loan ID.");
      return;
    }

    setIsLoadingApplication(true);
    setPersistenceError("");
    try {
      const state = await loadApplicationState(loanId);
      setValues((current) => state.draft ? state.draft.formValues : { ...current, loanId });
      setPrediction(state.draft?.riskIsStale ? null : state.draft?.riskResult ?? null);
      setRiskOutdated(Boolean(state.draft?.riskIsStale));
      setPersistedDecision(state.decision);
      setDecisionHistory(state.history);
      setOfficerDecision(state.decision?.officerDecision ?? "");
      setOverrideReason(state.decision?.overrideReason ?? "");
      setDecisionMessage(state.decision ? "Previous decision found." : "");
    } catch (error) {
      setPersistenceError(error instanceof Error ? error.message : "Unable to load saved application state.");
    } finally {
      setIsLoadingApplication(false);
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
      const nextErrors = validateApplicantForm(next);
      setErrors(nextErrors);
      return next;
    });
    if (prediction) {
      setRiskOutdated(true);
      setDecisionMessage("");
    }
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
      requestAnimationFrame(() => {
        formRef.current?.querySelector<HTMLElement>('[aria-invalid="true"]')?.focus();
      });
      return;
    }

    setIsPredicting(true);
    setPredictError("");
    setPrediction(null);

    try {
      const payload = mapFormToPredictPayload(values);
      const response = await predictRisk(payload);
      setPrediction(response);
      setRiskOutdated(false);
      setLastCalculatedAt(new Date());
      setOfficerDecision("");
      setOverrideReason("");
      setDecisionMessage("");
      requestAnimationFrame(() => resultRef.current?.focus());
      await refreshDrift(1);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Prediction failed.";
      setPredictError(message);
    } finally {
      setIsPredicting(false);
    }
  }

  function resetApplication() {
    setValues(initialValues);
    setErrors({});
    setPrediction(null);
    setRiskOutdated(false);
    setLastCalculatedAt(null);
    setOfficerDecision("");
    setOverrideReason("");
    setDecisionMessage("");
    setDraftMessage("");
  }

  async function saveDraft() {
    setIsSavingDraft(true);
    setPersistenceError("");
    try {
      const saved = await persistDraft({
        loanId: values.loanId,
        formValues: values,
        workflowStage: workflowStageNames[workflowStage],
        riskResult: prediction,
        riskIsStale: riskOutdated || !prediction,
      });
      setDraftMessage(`Draft saved at ${new Date(saved.updatedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}.`);
    } catch (error) {
      setPersistenceError(error instanceof Error ? error.message : "Unable to save draft. Try again.");
    } finally {
      setIsSavingDraft(false);
    }
  }

  async function saveDecision() {
    if (!officerDecision || !prediction || !recommendation || (decisionRequiresReason && !overrideReason.trim())) return;
    setIsSavingDecision(true);
    setPersistenceError("");
    try {
      const saved = await persistDecision({
        loanId: values.loanId,
        systemRecommendation: recommendation as DecisionValue,
        officerDecision: officerDecision as DecisionValue,
        overrideReason: decisionRequiresReason ? overrideReason : null,
        prediction,
      });
      setPersistedDecision(saved);
      setDecisionMessage(saved.decisionVersion > 1 ? "Decision updated." : "Decision saved successfully.");
      const refreshed = await loadApplicationState(values.loanId);
      setDecisionHistory(refreshed.history);
    } catch (error) {
      setPersistenceError(error instanceof Error ? error.message : "Unable to save decision. Try again.");
    } finally {
      setIsSavingDecision(false);
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
          <button type="button" className="logout-button" onClick={signOut}>Sign out</button>
        </div>
      </header>

      {activeTab === "comparison" ? (
        <ModelComparisonView onBackToChecker={() => setActiveTab("checker")} />
      ) : (
        <>
          <section className="lookup-bar panel">
            <div className="lookup-label">
              <FileText size={18} />
              <div>
                <strong>Application lookup</strong>
                <span>Use the loan ID to anchor this review.</span>
              </div>
            </div>
            <div className="lookup-controls">
              <input aria-label="Loan ID lookup" value={values.loanId} inputMode="numeric" onChange={(event) => updateField("loanId", event.target.value)} />
              <button type="button" className="secondary-button" onClick={() => void restoreApplication(values.loanId)} disabled={isLoadingApplication}>
                <RefreshCw size={16} className={isLoadingApplication ? "spin" : ""} /> {isLoadingApplication ? "Loading" : "Find application"}
              </button>
            </div>
          </section>
          <WorkflowStepper activeStage={workflowStage} />
          <section className="workspace">
          <form ref={formRef} className="panel form-panel" onSubmit={handleSubmit} noValidate>
            <div className="panel-header">
              <div>
                <p className="eyebrow">Applicant profile</p>
                <h2><UserRound size={19} /> Loan application</h2>
              </div>
              <span className="required-note"><strong>*</strong> Required</span>
            </div>

            <div className="applicant-overview">
              <div className="applicant-identity">
                <span className="applicant-avatar">{values.loanId.slice(-2) || "LC"}</span>
                <div><strong>Application {values.loanId || "unassigned"}</strong><span>Loan review workspace</span></div>
              </div>
              <div className="overview-metrics">
                <OverviewMetric icon={<WalletCards size={16} />} label="Requested loan" value={currency(values.loanAmount)} />
                <OverviewMetric icon={<CalendarDays size={16} />} label="Term" value={`${values.term || "-"} months`} />
                <OverviewMetric icon={<Gauge size={16} />} label="DTI" value={values.dti ? `${values.dti}%` : "-"} />
              </div>
            </div>

            <div className="field-grid">
              <Field label="Annual income (USD)" value={values.annualIncome} error={errors.annualIncome} required prefix="$" onChange={(value) => updateField("annualIncome", value)} />
              <Field label="Requested loan amount (USD)" value={values.loanAmount} error={errors.loanAmount} required prefix="$" onChange={(value) => updateField("loanAmount", value)} />
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
                label="Calculated monthly payment (USD)"
                value={values.installment}
                error={errors.installment}
                hint="Auto-amortized baseline payment"
                readOnly
              />
              <Field label="Debt-to-income ratio (%)" value={values.dti} error={errors.dti} required suffix="%" onChange={(value) => updateField("dti", value)} />
              <Field label="Loan ID (Lookup)" value={values.loanId} error={errors.loanId} onChange={(value) => updateField("loanId", value)} />
            </div>

            <section className="quality-section section-card">
              <div>
                <p className="eyebrow">Traditional Credit Bureau (Optional)</p>
                <h3>Credit Scores</h3>
              </div>
              <p className="quality-copy">
                Optional for collateral-free cashflow underwriting. Leave blank if evaluating thin-file or unbanked borrowers.
              </p>
              <div className="field-grid">
                <Field label="FICO score range - low" value={values.ficoRangeLow} error={errors.ficoRangeLow} onChange={(value) => updateField("ficoRangeLow", value)} />
                <Field label="FICO score range - high" value={values.ficoRangeHigh} error={errors.ficoRangeHigh} onChange={(value) => updateField("ficoRangeHigh", value)} />
              </div>
            </section>

            <section className="quality-section section-card">
              <div>
                <p className="eyebrow">Recommended</p>
                <h3>Better estimate inputs</h3>
              </div>
              <p className="quality-copy">These are optional, but they help the backend build stronger derived features.</p>
              <div className="field-grid">
                <Field label="Employment length (years)" value={values.empLength} error={errors.empLength} onChange={(value) => updateField("empLength", value)} />
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
                <Field label="Revolving utilization (%)" value={values.revolvingUtilization} error={errors.revolvingUtilization} suffix="%" onChange={(value) => updateField("revolvingUtilization", value)} />
              </div>
            </section>

            <section className="optional-section section-card">
              <div className="optional-header">
                <div>
                  <p className="eyebrow">Optional inputs</p>
                  <h3>Loan context</h3>
                </div>
                <label className="add-input-control">
                  <Plus size={18} />
                  <select
                    value=""
                    disabled={availableOptionalFields.length === 0}
                    onChange={(event) => addOptionalField(event.target.value as OptionalFieldKey)}
                  >
                    <option value="">{availableOptionalFields.length === 0 ? "All inputs added" : "Add optional details"}</option>
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

            <section className="credit-signals section-card">
              <div className="panel-header compact">
                <div><p className="eyebrow">Credit profile</p><h3><BarChart3 size={17} /> Credit signals</h3></div>
                <BadgeCheck size={19} className="signal-icon" />
              </div>
              <div className="signal-grid">
                <SignalCard label="Revolving utilization" value={values.revolvingUtilization ? `${values.revolvingUtilization}%` : "Not supplied"} note="Current revolving credit use" />
                <SignalCard label="FICO range" value={values.ficoRangeLow || values.ficoRangeHigh ? `${values.ficoRangeLow || "-"} - ${values.ficoRangeHigh || "-"}` : "Not supplied"} note="Traditional credit profile" />
                <SignalCard label="Recent delinquencies" value={values.delinq2Yrs || "0"} note="Reported in the last 2 years" />
              </div>
            </section>

            <p className="readiness-note" aria-live="polite">
              {isPredicting
                ? "Checking risk estimate..."
                : predictError
                ? "Unable to check risk - try again."
                : !modelReady
                ? selectedModel.name
                : !formReady
                ? `${completedRequiredCount} of ${requiredFields.length} required fields complete`
                : prediction
                ? "Risk estimate ready"
                : "Ready to check risk"}
            </p>
            {predictError && <p className="error-banner">{predictError}</p>}

            <button className="primary-button" type="submit" disabled={!modelReady || !formReady || isPredicting}>
              {isPredicting ? <Loader2 className="spin" size={18} /> : <HeartPulse size={18} />}
              {isPredicting ? "Checking" : "Check Risk"}
            </button>
            <div className="form-actions">
              <button type="button" className="secondary-button" onClick={() => void saveDraft()} disabled={isSavingDraft}>
                <Save size={16} /> {isSavingDraft ? "Saving" : "Save draft"}
              </button>
              <button type="button" className="text-button" onClick={resetApplication}>
                <RotateCcw size={15} /> Reset changes
              </button>
            </div>
            {draftMessage && <p className="success-note" aria-live="polite"><CheckCircle2 size={15} /> {draftMessage}</p>}
            {persistenceError && <p className="error-banner" role="alert">{persistenceError}</p>}
          </form>

          <aside className="result-stack">
            <section className="panel summary-panel">
              <div className="panel-header compact">
                <div>
                  <p className="eyebrow">Application summary</p>
                  <h2><UserRound size={17} /> Loan {values.loanId || "unassigned"}</h2>
                </div>
                <span className="summary-stage">{workflowLabel(workflowStage)}</span>
              </div>
              <div className="summary-grid">
                <SummaryItem label="Requested amount" value={currency(values.loanAmount)} />
                <SummaryItem label="Term / payment" value={`${values.term || "-"} mo / ${currency(values.installment)}`} />
                <SummaryItem label="Income / DTI" value={`${currency(values.annualIncome)} / ${values.dti || "-"}%`} />
                <SummaryItem label="FICO range" value={values.ficoRangeLow || values.ficoRangeHigh ? `${values.ficoRangeLow || "-"} - ${values.ficoRangeHigh || "-"}` : "Not supplied"} />
              </div>
            </section>
            <section className={`panel result-panel ${riskLabel ? `risk-${riskLabel.level}` : ""}`} ref={resultRef} tabIndex={-1} aria-live="polite">
              <p className="eyebrow">Risk estimate</p>
              {riskOutdated && <div className="warning-chip"><ShieldAlert size={15} /> Risk assessment needs updating</div>}
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
                  <div className="recommendation-row"><span>Recommended action</span><strong>{decisionLabel(recommendation)}</strong></div>
                  {lastCalculatedAt && <p className="freshness-note">Calculated just now · {selectedModel.shortName}</p>}
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
                  <h2>{!modelReady ? "Service unavailable" : formReady ? "Ready to check risk" : "Complete the required fields"}</h2>
                  <p className="muted">{!modelReady ? "The production model is not available." : `Complete ${requiredFields.length - completedRequiredCount} required field${requiredFields.length - completedRequiredCount === 1 ? "" : "s"} to calculate risk.`}</p>
                  <div className="requirement-list">{requiredFields.map((field) => <span key={field}>{values[field].trim() && !validationErrors[field] ? <CheckCircle2 size={14} /> : <Circle size={14} />} {requiredFieldLabel(field)}</span>)}</div>
                </>
              )}
            </section>

            {persistedDecision && (!prediction || riskOutdated) && <PersistedDecisionCard decision={persistedDecision} history={decisionHistory} />}
            {prediction && !riskOutdated && riskLabel && <DecisionPanel recommendation={recommendation} decision={officerDecision} overrideReason={overrideReason} decisionRequiresReason={decisionRequiresReason} decisionMessage={decisionMessage} isSaving={isSavingDecision} persistedDecision={persistedDecision} history={decisionHistory} onDecisionChange={setOfficerDecision} onReasonChange={setOverrideReason} onSave={() => void saveDecision()} />}

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
                    : "No production model selected"
                }
              />
              <StatusLine label="Drift" value={drift ? (drift.alert ? "Review needed" : "Healthy") : driftMessage || "Drift check has not run"} />
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
                    value={prediction?.model_name || selectedModel.name}
                  />
                  <StatusLine label="Model version" value={prediction?.model_version || health?.model_version || "Unavailable"} />
                  <StatusLine label="Model type" value={health?.model_type ? health.model_type.toUpperCase() : "Unavailable"} />
                  <StatusLine label="Registry stage" value={health?.model_stage || selectedModel.status} />
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
        </>
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
  prefix?: string;
  suffix?: string;
  readOnly?: boolean;
  onChange?: (value: string) => void;
};

function Field({ label, value, error, required, hint, prefix, suffix, readOnly, onChange }: FieldProps) {
  return (
    <label className="field">
      <span>
        {label}
        {required && <strong> *</strong>}
      </span>
      <span className="input-shell">
        {prefix && <span className="input-affix">{prefix}</span>}
        <input value={value} inputMode="decimal" readOnly={readOnly} aria-invalid={Boolean(error)} onChange={(event) => onChange?.(event.target.value)} />
        {suffix && <span className="input-affix">{suffix}</span>}
      </span>
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

function currency(value: string): string {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? `$${parsed.toLocaleString(undefined, { maximumFractionDigits: 2 })}` : "-";
}

function decisionLabel(decision: string): string {
  if (decision === "approve") return "Approve";
  if (decision === "decline") return "Decline";
  if (decision === "refer") return "Refer for review";
  return "Pending";
}

function workflowLabel(stage: number): string {
  return ["Draft", "Details complete", "Risk assessed", "Decision recorded"][stage] ?? "Draft";
}

function requiredFieldLabel(field: "annualIncome" | "loanAmount" | "term" | "dti"): string {
  return { annualIncome: "Income", loanAmount: "Loan amount", term: "Loan term", dti: "Debt-to-income ratio" }[field];
}

function SummaryItem({ label, value }: { label: string; value: string }) {
  return <div className="summary-item"><span>{label}</span><strong>{value}</strong></div>;
}

function PersistedDecisionCard({ decision, history }: { decision: PersistedDecision; history: DecisionAuditEvent[] }) {
  return <section className="panel persisted-decision-panel" aria-live="polite">
    <p className="eyebrow">Saved underwriting decision</p>
    <h2>Previous decision found</h2>
    <div className="persisted-decision-details">
      <SummaryItem label="Officer decision" value={decisionLabel(decision.officerDecision)} />
      <SummaryItem label="System recommendation" value={decisionLabel(decision.systemRecommendation)} />
      <SummaryItem label="Decision recorded" value={new Date(decision.updatedAt).toLocaleString()} />
      <SummaryItem label="Decision version" value={String(decision.decisionVersion)} />
    </div>
    {decision.overrideReason && <p className="persisted-override"><strong>Override reason:</strong> {decision.overrideReason}</p>}
    {history.length > 1 && <p className="muted">This loan has {history.length} recorded decision events.</p>}
  </section>;
}

function OverviewMetric({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return <div className="overview-metric"><span className="metric-icon">{icon}</span><span>{label}</span><strong>{value}</strong></div>;
}

function SignalCard({ label, value, note }: { label: string; value: string; note: string }) {
  return <div className="signal-card"><span className="metric-icon"><BarChart3 size={15} /></span><div><strong>{value}</strong><span>{label}</span><small>{note}</small></div></div>;
}

function WorkflowStepper({ activeStage }: { activeStage: number }) {
  const stages = ["Draft", "Details complete", "Risk assessed", "Decision"];
  return <nav className="workflow-stepper" aria-label="Application workflow">{stages.map((stage, index) => <div key={stage} className={`workflow-step ${index <= activeStage ? "complete" : ""} ${index === activeStage ? "current" : ""}`}><span className="workflow-dot">{index < activeStage ? <Check size={14} /> : index + 1}</span><span>{stage}</span>{index < stages.length - 1 && <span className="workflow-line" />}</div>)}</nav>;
}

type DecisionPanelProps = {
  recommendation: string;
  decision: "approve" | "refer" | "decline" | "";
  overrideReason: string;
  decisionRequiresReason: boolean;
  decisionMessage: string;
  isSaving: boolean;
  persistedDecision: PersistedDecision | null;
  history: DecisionAuditEvent[];
  onDecisionChange: (decision: "approve" | "refer" | "decline") => void;
  onReasonChange: (reason: string) => void;
  onSave: () => void;
};

function DecisionPanel({ recommendation, decision, overrideReason, decisionRequiresReason, decisionMessage, isSaving, persistedDecision, history, onDecisionChange, onReasonChange, onSave }: DecisionPanelProps) {
  const decisionOptions: Array<"approve" | "refer" | "decline"> = ["approve", "refer", "decline"];
  return <section className="panel decision-panel" aria-live="polite">
    <div className="panel-header compact"><div><p className="eyebrow">Officer decision</p><h2>Record an action</h2></div><ShieldAlert size={19} /></div>
    <p className="recommendation-copy">System recommendation: <strong>{decisionLabel(recommendation)}</strong></p>
    <div className="decision-options">{decisionOptions.map((option) => <button key={option} type="button" className={`decision-button decision-${option} ${decision === option ? "selected" : ""}`} onClick={() => onDecisionChange(option)}>{option === recommendation && <CheckCircle2 size={15} />} {decisionLabel(option)}</button>)}</div>
    {persistedDecision && <div className="persisted-decision"><strong>Previous decision found</strong><span>{decisionLabel(persistedDecision.officerDecision)} · saved {new Date(persistedDecision.updatedAt).toLocaleString()}</span>{persistedDecision.overrideReason && <span>Override: {persistedDecision.overrideReason}</span>}</div>}
    {decisionRequiresReason && <label className="field override-field"><span>Reason for override <strong>*</strong></span><textarea value={overrideReason} onChange={(event) => onReasonChange(event.target.value)} placeholder="Explain why the officer decision differs from the system recommendation." rows={3} /></label>}
    <button type="button" className="primary-button" disabled={!decision || isSaving || (decisionRequiresReason && !overrideReason.trim())} onClick={onSave}><Save size={17} /> {isSaving ? "Saving decision" : "Save decision"}</button>
    {decisionMessage && <p className="success-note"><CheckCircle2 size={15} /> {decisionMessage}</p>}
    {history.length > 1 && <details className="decision-history"><summary>Decision history ({history.length})</summary>{history.map((event) => <div key={event.id}><strong>{event.eventType === "created" ? "Created" : "Updated"}</strong><span>{new Date(event.createdAt).toLocaleString()}</span></div>)}</details>}
  </section>;
}

export default App;
