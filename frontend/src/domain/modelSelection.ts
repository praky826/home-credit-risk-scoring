import type { HealthResponse, ModelBenchmarkItem, ModelComparisonResponse } from "./types";

export type SelectedModel = {
  id: string;
  name: string;
  shortName: string;
  status: string;
  isConfigured: boolean;
};

function shortModelName(name: string): string {
  return name
    .replace(/^LendingClub\s+/i, "")
    .replace(/\s+Default Risk Classifier$/i, "")
    .trim();
}

export function getSelectedModel(
  health: HealthResponse | null,
  comparison?: ModelComparisonResponse | null,
): SelectedModel {
  const comparisonModel = comparison?.models.find(
    (model) => model.model_id === comparison.champion_model_id,
  );
  const name = health?.model_name || comparison?.champion_model_name || comparisonModel?.name;

  if (!name || health?.model_loaded === false) {
    return {
      id: comparison?.champion_model_id || "",
      name: "No production model selected",
      shortName: "No production model",
      status: "Not configured",
      isConfigured: false,
    };
  }

  return {
    id: comparison?.champion_model_id || health?.model_type || "production",
    name,
    shortName: shortModelName(name),
    status: health?.model_stage || "Production",
    isConfigured: true,
  };
}

export function getComparisonChampion(data: ModelComparisonResponse): ModelBenchmarkItem | null {
  return data.models.find((model) => model.model_id === data.champion_model_id) || null;
}