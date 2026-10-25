import type { RiskLabel } from "./types";

export function getRiskLabel(probability: number): RiskLabel {
  if (probability < 0.15) {
    return {
      level: "low",
      title: "Low Risk",
      message: "This application is within low default probability benchmarks.",
    };
  }

  if (probability <= 0.30) {
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
