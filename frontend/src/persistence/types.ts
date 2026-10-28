import type { ApplicantFormValues, PredictResponse } from "../domain/types";

export type WorkflowStage = "draft" | "details_complete" | "risk_assessed" | "decision_recorded";
export type DecisionValue = "approve" | "refer" | "decline";

export type PersistedDraft = {
  id: string;
  loanId: string;
  officerUserId: string;
  formValues: ApplicantFormValues;
  workflowStage: WorkflowStage;
  riskResult: PredictResponse | null;
  riskIsStale: boolean;
  createdAt: string;
  updatedAt: string;
};

export type PersistedDecision = {
  id: string;
  loanId: string;
  officerUserId: string;
  systemRecommendation: DecisionValue;
  officerDecision: DecisionValue;
  overrideReason: string | null;
  predictionRequestId: string | null;
  modelVersion: string | null;
  defaultProbability: number | null;
  decisionVersion: number;
  createdAt: string;
  updatedAt: string;
};

export type DecisionAuditEvent = {
  id: string;
  decisionId: string;
  loanId: string;
  officerUserId: string;
  eventType: "created" | "updated";
  previousSnapshot: Record<string, unknown> | null;
  newSnapshot: Record<string, unknown>;
  createdAt: string;
};

export type ApplicationState = {
  draft: PersistedDraft | null;
  decision: PersistedDecision | null;
  history: DecisionAuditEvent[];
};
