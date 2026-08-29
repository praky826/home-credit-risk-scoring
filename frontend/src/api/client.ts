import type {
  DriftResponse,
  HealthResponse,
  ModelComparisonResponse,
  PredictPayload,
  PredictResponse,
} from "../domain/types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function parseJsonResponse<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => ({}));

  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : "Request failed.";
    throw new Error(detail);
  }

  return body as T;
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);
  return parseJsonResponse<HealthResponse>(response);
}

export async function predictRisk(payload: PredictPayload): Promise<PredictResponse> {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  return parseJsonResponse<PredictResponse>(response);
}

export async function getDrift(minRecords = 1): Promise<DriftResponse> {
  const response = await fetch(`${API_BASE_URL}/drift?min_records=${minRecords}`);
  return parseJsonResponse<DriftResponse>(response);
}

export async function getModelComparison(): Promise<ModelComparisonResponse> {
  const response = await fetch(`${API_BASE_URL}/model-comparison`);
  return parseJsonResponse<ModelComparisonResponse>(response);
}

