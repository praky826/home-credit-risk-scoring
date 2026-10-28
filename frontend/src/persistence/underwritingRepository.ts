import type { ApplicantFormValues, PredictResponse } from "../domain/types";
import { normalizeLoanId } from "../domain/loanId";
import { supabase } from "../lib/supabase";
import type {
  ApplicationState,
  DecisionAuditEvent,
  DecisionValue,
  PersistedDecision,
  PersistedDraft,
  WorkflowStage,
} from "./types";

function requireSupabase() {
  if (!supabase) throw new Error("Persistence is not configured for this environment.");
  return supabase;
}

async function currentUserId(): Promise<string> {
  const client = requireSupabase();
  const { data, error } = await client.auth.getUser();
  if (error || !data.user) throw new Error("Your session has expired. Please sign in again.");
  return data.user.id;
}

function mapDraft(row: Record<string, unknown>): PersistedDraft {
  return {
    id: String(row.id),
    loanId: String(row.loan_id),
    officerUserId: String(row.officer_user_id),
    formValues: row.form_values as ApplicantFormValues,
    workflowStage: row.workflow_stage as WorkflowStage,
    riskResult: (row.risk_result as PredictResponse | null) ?? null,
    riskIsStale: Boolean(row.risk_is_stale),
    createdAt: String(row.created_at),
    updatedAt: String(row.updated_at),
  };
}

function mapDecision(row: Record<string, unknown>): PersistedDecision {
  return {
    id: String(row.id),
    loanId: String(row.loan_id),
    officerUserId: String(row.officer_user_id),
    systemRecommendation: row.system_recommendation as DecisionValue,
    officerDecision: row.officer_decision as DecisionValue,
    overrideReason: (row.override_reason as string | null) ?? null,
    predictionRequestId: (row.prediction_request_id as string | null) ?? null,
    modelVersion: (row.model_version as string | null) ?? null,
    defaultProbability: row.default_probability == null ? null : Number(row.default_probability),
    decisionVersion: Number(row.decision_version),
    createdAt: String(row.created_at),
    updatedAt: String(row.updated_at),
  };
}

function mapAuditEvent(row: Record<string, unknown>): DecisionAuditEvent {
  return {
    id: String(row.id),
    decisionId: String(row.decision_id),
    loanId: String(row.loan_id),
    officerUserId: String(row.officer_user_id),
    eventType: row.event_type as "created" | "updated",
    previousSnapshot: (row.previous_snapshot as Record<string, unknown> | null) ?? null,
    newSnapshot: row.new_snapshot as Record<string, unknown>,
    createdAt: String(row.created_at),
  };
}

export async function loadApplicationState(loanId: string): Promise<ApplicationState> {
  const client = requireSupabase();
  const normalizedLoanId = normalizeLoanId(loanId);
  const userId = await currentUserId();
  const [draftResult, decisionResult, historyResult] = await Promise.all([
    client.from("application_drafts").select("*").eq("officer_user_id", userId).eq("loan_id", normalizedLoanId).maybeSingle(),
    client.from("underwriting_decisions").select("*").eq("officer_user_id", userId).eq("loan_id", normalizedLoanId).maybeSingle(),
    client.from("decision_audit_events").select("*").eq("officer_user_id", userId).eq("loan_id", normalizedLoanId).order("created_at", { ascending: false }),
  ]);
  const firstError = draftResult.error || decisionResult.error || historyResult.error;
  if (firstError) throw new Error(`Unable to load saved application state: ${firstError.message}`);
  return {
    draft: draftResult.data ? mapDraft(draftResult.data) : null,
    decision: decisionResult.data ? mapDecision(decisionResult.data) : null,
    history: (historyResult.data ?? []).map(mapAuditEvent),
  };
}

export async function saveDraft(input: {
  loanId: string;
  formValues: ApplicantFormValues;
  workflowStage: WorkflowStage;
  riskResult: PredictResponse | null;
  riskIsStale: boolean;
}): Promise<PersistedDraft> {
  const client = requireSupabase();
  const normalizedLoanId = normalizeLoanId(input.loanId);
  const userId = await currentUserId();
  const { data, error } = await client
    .from("application_drafts")
    .upsert({
      loan_id: normalizedLoanId,
      officer_user_id: userId,
      form_values: input.formValues,
      workflow_stage: input.workflowStage,
      risk_result: input.riskResult,
      risk_is_stale: input.riskIsStale,
    }, { onConflict: "officer_user_id,loan_id" })
    .select()
    .single();
  if (error || !data) throw new Error(`Unable to save draft: ${error?.message ?? "No record returned."}`);
  return mapDraft(data);
}

export async function saveDecision(input: {
  loanId: string;
  systemRecommendation: DecisionValue;
  officerDecision: DecisionValue;
  overrideReason: string | null;
  prediction: PredictResponse;
}): Promise<PersistedDecision> {
  const client = requireSupabase();
  const normalizedLoanId = normalizeLoanId(input.loanId);
  const userId = await currentUserId();
  const { data, error } = await client
    .from("underwriting_decisions")
    .upsert({
      loan_id: normalizedLoanId,
      officer_user_id: userId,
      system_recommendation: input.systemRecommendation,
      officer_decision: input.officerDecision,
      override_reason: input.overrideReason?.trim() || null,
      prediction_request_id: input.prediction.request_id,
      model_version: input.prediction.model_version,
      default_probability: input.prediction.default_probability,
    }, { onConflict: "officer_user_id,loan_id" })
    .select()
    .single();
  if (error || !data) throw new Error(`Unable to save decision: ${error?.message ?? "No record returned."}`);
  return mapDecision(data);
}
