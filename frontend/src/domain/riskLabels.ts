import type { RiskLabel } from "./types";

export function getRiskLabel(probability: number): RiskLabel {
  if (probability < 0.1) {
    return {
      level: "low",
      title: "Low Risk",
      message: "This application is below the initial review threshold.",
    };
  }

  if (probability <= 0.25) {
    return {
      level: "review",
      title: "Needs Review",
      message: "This application should be reviewed before moving forward.",
    };
  }

  return {
    level: "high",
    title: "High Attention",
    message: "This application has a high estimated default probability.",
  };
}

export function formatProbability(probability: number): string {
  return `${(probability * 100).toFixed(1)}%`;
}
