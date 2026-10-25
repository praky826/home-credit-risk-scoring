import { useEffect, useState } from "react";
import {
  Award,
  BarChart2,
  CheckCircle2,
  Cpu,
  Database,
  FileCheck,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  Zap,
} from "lucide-react";
import { getModelComparison } from "../api/client";
import { defaultModelComparison } from "../domain/modelComparisonData";
import { getComparisonChampion, getSelectedModel } from "../domain/modelSelection";
import type { ModelBenchmarkItem, ModelComparisonResponse } from "../domain/types";

type ModelComparisonViewProps = {
  onBackToChecker?: () => void;
};

export function ModelComparisonView({ onBackToChecker }: ModelComparisonViewProps) {
  const [data, setData] = useState<ModelComparisonResponse>(defaultModelComparison);
  const [loading, setLoading] = useState(false);
  const [activeTabMetric, setActiveTabMetric] = useState<"all" | "trees" | "neural">("all");

  useEffect(() => {
    fetchComparison();
  }, []);

  async function fetchComparison() {
    setLoading(true);
    try {
      const res = await getModelComparison();
      if (res && res.models && res.models.length > 0) {
        setData(res);
      }
    } catch {
      // Fallback to rich embedded default data
      setData(defaultModelComparison);
    } finally {
      setLoading(false);
    }
  }

  const filteredModels = data.models.filter((m) => {
    if (activeTabMetric === "trees") return m.model_id === "xgboost" || m.model_id === "lgbm";
    if (activeTabMetric === "neural")
      return (
        m.model_id === "entity_embedding" ||
        m.model_id === "mlp" ||
        m.model_id === "tab_transformer"
      );
    return true;
  });

  const champion = getComparisonChampion(data);
  const selectedModel = getSelectedModel(null, data);
  const championConfigured = Boolean(champion && selectedModel.isConfigured);

  return (
    <div className="comparison-container">
      {/* Top action header */}
      <div className="comparison-header">
        <div>
          <p className="eyebrow">Model Governance & Benchmarking</p>
          <h2>Model Comparison & Selection Protocol</h2>
        </div>
        <div className="header-actions">
          {onBackToChecker && (
            <button type="button" className="ghost-button" onClick={onBackToChecker}>
              ← Back to Risk Checker
            </button>
          )}
          <button
            type="button"
            className="ghost-button refresh-btn"
            onClick={fetchComparison}
            title="Refresh comparison data"
          >
            <RefreshCw size={15} className={loading ? "spin" : ""} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Production Champion Hero Banner */}
      <section className={`champion-card ${!championConfigured ? "champion-card-empty" : ""}`}>
        <div className="champion-badge-row">
          <span className="champion-pill">
            <Award size={16} />
            <span>{championConfigured ? "Production Champion" : "No production model selected"}</span>
          </span>
          <span className="live-status-pill">
            {championConfigured ? <CheckCircle2 size={15} /> : <ShieldCheck size={15} />}
            <span>{championConfigured ? "Serving Active" : "Serving unavailable"}</span>
          </span>
        </div>

        <div className="champion-body">
          <div className="champion-main-info">
            <h3 className="champion-title">{championConfigured ? selectedModel.name : "No production model selected"}</h3>
            <p className="champion-desc">
              {championConfigured
                ? `Selected from the current benchmark on ${data.evaluation_dataset.train_samples?.toLocaleString() ?? "the available"} training loans.`
                : "Comparison data does not identify a production champion. Review model registration before serving risk estimates."}
            </p>
          </div>

          {champion && <div className="champion-metrics-grid">
            <div className="metric-box highlight">
              <span className="metric-box-label">Test ROC-AUC</span>
              <strong className="metric-box-val">{champion.roc_auc.toFixed(5)}</strong>
              <span className="metric-box-sub">Rank #1 (Best Discrimination)</span>
            </div>
            <div className="metric-box">
              <span className="metric-box-label">KS Statistic</span>
              <strong className="metric-box-val">{champion.ks_stat.toFixed(5)}</strong>
              <span className="metric-box-sub">Max Default Separation</span>
            </div>
            <div className="metric-box">
              <span className="metric-box-label">Inference Latency</span>
              <strong className="metric-box-val">{champion.inference_latency_ms} ms</strong>
              <span className="metric-box-sub">Sub-2ms CPU Single Score</span>
            </div>
            <div className="metric-box">
              <span className="metric-box-label">Brier Loss</span>
              <strong className="metric-box-val">{champion.brier_score.toFixed(4)}</strong>
              <span className="metric-box-sub">High Calibration Fidelity</span>
            </div>
          </div>}
        </div>
      </section>

      <section className="selection-summary" aria-label="Selection summary">
        <strong>Selection summary</strong>
        <span>
          {championConfigured
            ? `${selectedModel.shortName} is the current champion because it balances ranking performance, calibration, and inference latency.`
            : "No selection summary is available until a production champion is configured."}
        </span>
      </section>

      {/* Benchmark Table Section */}
      <section className="panel comparison-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Comprehensive Benchmark</p>
            <h3>Multi-Model Evaluation Matrix (Holdout Test Split)</h3>
          </div>
          <div className="filter-pill-group">
            <button
              type="button"
              className={`filter-pill ${activeTabMetric === "all" ? "active" : ""}`}
              onClick={() => setActiveTabMetric("all")}
            >
              All 5 Models
            </button>
            <button
              type="button"
              className={`filter-pill ${activeTabMetric === "trees" ? "active" : ""}`}
              onClick={() => setActiveTabMetric("trees")}
            >
              Gradient Boosted Trees
            </button>
            <button
              type="button"
              className={`filter-pill ${activeTabMetric === "neural" ? "active" : ""}`}
              onClick={() => setActiveTabMetric("neural")}
            >
              Neural Networks
            </button>
          </div>
        </div>

        <div className="table-responsive">
          <table className="benchmark-table">
            <thead>
              <tr>
                <th>Model & Architecture</th>
                <th className="num-col" title="How well the model ranks higher-risk loans above lower-risk loans">ROC-AUC<br /><small>Higher is better</small></th>
                <th className="num-col" title="How clearly the model separates likely defaults from non-defaults">KS Stat<br /><small>Higher is better</small></th>
                <th className="num-col" title="How accurate predicted probabilities are">Brier Score<br /><small>Lower is better</small></th>
                <th className="num-col" title="How closely predicted risk matches observed risk">Calib. Slope<br /><small>Closer to 1</small></th>
                <th className="num-col" title="Time required to generate one prediction">Latency<br /><small>Lower is better</small></th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {filteredModels.map((item) => (
                <ModelRow key={item.model_id} model={item} isChampion={item.model_id === data.champion_model_id} />
              ))}
            </tbody>
          </table>
        </div>
        <p className="table-footnote">
          * Evaluated on holdout test set (30,751 rows, ~8.1% positive default rate) using the standardized 235 engineered features.
          * First-pass LendingClub evaluation. Final champion status should be confirmed after tuning and calibration review.
        </p>
      </section>

      {/* Why the Model Was Selected Section */}
      <section className="panel rationale-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Decision Rationale</p>
            <h3>Why {selectedModel.shortName} Was Selected for Production</h3>
          </div>
        </div>

        <div className="rationale-grid">
          {data.selection_rationales.map((item, idx) => (
            <RationaleCard key={item.title} item={item} index={idx + 1} />
          ))}
        </div>
      </section>

      {/* Statistical Significance & Protocol Details */}
      <div className="details-two-col">
        {/* Wilcoxon Matrix */}
        <section className="panel sub-panel">
          <div className="panel-header compact">
            <div>
              <p className="eyebrow">Statistical Validation</p>
              <h3>Pairwise Wilcoxon Tests</h3>
            </div>
            <span className="significance-badge">p &lt; 0.05 Threshold</span>
          </div>
          <div className="wilcoxon-list">
            {data.significance_tests.map((test) => (
              <div key={test.comparison} className="wilcoxon-card">
                <div className="wilcoxon-header">
                  <strong>{test.comparison}</strong>
                  <span className="p-tag">
                    p = {test.p_value.toFixed(5)} ({test.significant ? "Significant" : "Not Sig."})
                  </span>
                </div>
                <p className="wilcoxon-text">{test.interpretation}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Dataset & Protocol Card */}
        <section className="panel sub-panel">
          <div className="panel-header compact">
            <div>
              <p className="eyebrow">Methodology</p>
              <h3>Dataset & Split Protocol</h3>
            </div>
          </div>
          <div className="dataset-meta-list">
            <div className="meta-row">
              <span className="meta-label">
                <Database size={15} /> Training Split
              </span>
              <strong>{data.evaluation_dataset.train_samples?.toLocaleString() ?? "246,009"} rows (80%)</strong>
            </div>
            <div className="meta-row">
              <span className="meta-label">
                <BarChart2 size={15} /> Validation Split
              </span>
              <strong>{data.evaluation_dataset.val_samples?.toLocaleString() ?? "30,751"} rows (10%)</strong>
            </div>
            <div className="meta-row">
              <span className="meta-label">
                <TrendingUp size={15} /> Holdout Test Split
              </span>
              <strong>{data.evaluation_dataset.test_samples?.toLocaleString() ?? "30,751"} rows (10%)</strong>
            </div>
            <div className="meta-row">
              <span className="meta-label">
                <Cpu size={15} /> Engineered Features
              </span>
              <strong>{data.evaluation_dataset.features_evaluated ?? 235} features</strong>
            </div>
            <div className="meta-row full">
              <span className="meta-label">
                <FileCheck size={15} /> Split Construction
              </span>
              <span className="meta-desc">
                {data.evaluation_dataset.evaluation_protocol}
              </span>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

function ModelRow({ model, isChampion }: { model: ModelBenchmarkItem; isChampion: boolean }) {
  return (
    <tr className={`model-row ${isChampion ? "is-champion-row" : ""}`}>
      <td>
        <div className="model-name-cell">
          <div className="name-and-badge">
            <span className="model-main-name">{model.name}</span>
            {isChampion && <span className="winner-tag">Production champion</span>}
          </div>
          <small className="model-arch-text">{model.architecture}</small>
        </div>
      </td>
      <td className="num-col">
        <strong className={isChampion ? "metric-highlight" : ""}>
          {model.roc_auc.toFixed(5)}
        </strong>
      </td>
      <td className="num-col">{model.ks_stat.toFixed(5)}</td>
      <td className="num-col">{model.brier_score.toFixed(4)}</td>
      <td className="num-col">{model.calibration_slope.toFixed(4)}</td>
      <td className="num-col">{model.inference_latency_ms} ms</td>
      <td>
        <span
          className={`status-pill ${
            isChampion
              ? "status-champion"
              : model.status.includes("Runner")
              ? "status-runner"
              : "status-other"
          }`}
        >
          {isChampion ? "Production champion" : normalizeStatus(model.status)}
        </span>
      </td>
    </tr>
  );
}

function normalizeStatus(status: string): string {
  const normalized = status.toLowerCase();
  if (normalized.includes("experimental")) return "Experimental";
  if (normalized.includes("challenger") || normalized.includes("runner")) return "Challenger";
  if (normalized.includes("baseline")) return "Baseline";
  return "Challenger";
}

function RationaleCard({ item, index }: { item: { title: string; summary: string; details: string }; index: number }) {
  const icons = [
    <TrendingUp key="1" size={18} />,
    <Sparkles key="2" size={18} />,
    <Zap key="3" size={18} />,
    <ShieldCheck key="4" size={18} />,
    <FileCheck key="5" size={18} />,
  ];

  return (
    <div className="rationale-card">
      <div className="rationale-card-header">
        <span className="rationale-icon">{icons[(index - 1) % icons.length]}</span>
        <h4>{item.title}</h4>
      </div>
      <p className="rationale-summary">{item.summary}</p>
      <p className="rationale-details">{item.details}</p>
    </div>
  );
}
