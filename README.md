# Home Credit Default Risk — Machine Learning & Production Scoring Suite

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-3178C6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-EB5424.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.14-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-grade credit risk scoring system built on the **Home Credit Default Risk** dataset (307,511 loan applications). Features an automated feature engineering pipeline, rigorous multi-model benchmarking (GBDTs vs. Tabular Deep Learning), statistical validation (Wilcoxon signed-rank tests), real-time population stability index (PSI) drift detection, and an interactive loan underwriter frontend.

---

## 🌟 Key Highlights

- **Production Champion Model**: Optuna-tuned **XGBoost Classifier** achieving **0.78084 Test ROC-AUC** and **0.42753 KS Statistic** with **sub-2ms CPU inference latency**.
- **Comprehensive 5-Model Benchmark**: Rigorous comparative study evaluating Gradient Boosted Decision Trees (XGBoost, LightGBM) against Tabular Deep Learning architectures (Entity Embeddings, MLP, TabTransformer).
- **Loan Underwriter Workspace**: Clean, responsive React + TypeScript interface for real-time risk assessment, loan quality estimation, and system health tracking.
- **Model Governance Dashboard**: Embedded frontend benchmark viewer displaying side-by-side metric tables, statistical significance matrices, and model selection rationales.
- **Real-Time PSI Drift Monitoring**: Rolling-window drift detection comparing production traffic distributions against reference baseline statistics with alerting hooks.

---

## 📊 Model Benchmark & Comparison

All models were evaluated on the exact same temporal holdout test set (30,751 rows, ~8.1% positive default rate) using **235 engineered features**:

| Rank | Model Architecture | Test ROC-AUC | KS Statistic | Brier Score | Calib. Slope | Latency (CPU) | Production Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **XGBoost Classifier** | **0.78084** | **0.42753** | 0.17519 | 0.35894 | **~1.2 ms** | **Production Champion** |
| 🥈 | **LightGBM Booster** | 0.77810 | 0.42641 | **0.17258** | 0.34641 | **~1.0 ms** | Production Runner-up |
| 🥉 | **Entity Embedding Network** | 0.76804 | 0.41592 | 0.20476 | 0.32133 | ~4.5 ms | Deep Learning Champion |
| 4 | **Multi-Layer Perceptron (MLP)** | 0.70078 | 0.29616 | 0.21878 | 0.31383 | ~3.8 ms | Deep Learning Baseline |
| 5 | **TabTransformer Network** | 0.67678 | 0.26067 | 0.22255 | 0.30194 | ~14.2 ms | Experimental Transformer |

### 🏆 Why XGBoost Was Selected for Production

1. **Maximum Discriminatory Power**: Delivered the highest ROC-AUC and maximum Kolmogorov-Smirnov separation (KS = 0.4275), minimizing bad-debt write-offs while maximizing approval volume for creditworthy applicants.
2. **Tabular Inductive Biases**: Tree ensembles natively partition unnormalized skewed credit distributions, irregular scales, multi-table aggregations, and high missingness without artificial manifold smoothing assumptions required by neural networks.
3. **Sub-2ms Inference SLA**: Compact compiled binary trees score requests in ~1.2ms on standard CPU server hardware, eliminating expensive GPU infrastructure dependencies.
4. **Explainability & Fair Lending**: Exact split gains and TreeSHAP compatibility guarantee regulatory compliance (e.g. ECOA, FCRA adverse action notices).
5. **Statistical Significance**: Pairwise Wilcoxon signed-rank tests confirmed statistically significant performance advantages ($p < 0.001$ over neural models; $p = 0.042$ over LightGBM).

---

## 🏗️ System Architecture

```
home-credit-risk-scoring/
├── configs/                  # Model & serving YAML configurations
│   ├── serving.yaml          # Host, port, drift threshold, rolling window
│   ├── xgboost.yaml          # XGBoost hyperparameters
│   ├── lgbm.yaml             # LightGBM hyperparameters
│   └── ...
├── data/                     # Raw, processed, split datasets (gitignored)
├── frontend/                 # React 19 + TypeScript + Vite UI
│   ├── src/
│   │   ├── api/client.ts     # FastAPI fetch client
│   │   ├── components/       # ModelComparisonView dashboard & forms
│   │   ├── domain/           # Types, feature catalogs, validation rules
│   │   └── styles/global.css # Responsive design system
│   └── package.json
├── src/                      # Backend Python package
│   ├── features/             # Multi-table feature pipeline (Bureau, POS, Cards, Installments)
│   ├── models/               # XGBoost, LightGBM, MLP, EntityEmbedding, TabTransformer
│   ├── evaluation/           # Metrics, ROC/PR curves, Wilcoxon significance tests
│   ├── monitoring/           # Population Stability Index (PSI) drift monitoring
│   └── serving/              # FastAPI application, schemas, router, model loader
├── tests/                    # Pytest test suite (features, models, serving)
├── requirements.txt          # Production dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Clone Repository
```bash
git clone https://github.com/praky826/home-credit-risk-scoring.git
cd home-credit-risk-scoring
```

### 2. Python Environment Setup
```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Start Backend FastAPI Service
```powershell
$env:PYTHONPATH="."
.venv\Scripts\python -m uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
```
- **API Endpoint:** `http://127.0.0.1:8000`
- **Interactive Swagger Docs:** `http://127.0.0.1:8000/docs`
- **Health Status:** `GET http://127.0.0.1:8000/health`
- **Model Comparison API:** `GET http://127.0.0.1:8000/model-comparison`

### 4. Start Frontend Web Application
In a separate terminal:
```bash
npm --prefix frontend install
npm --prefix frontend run dev
```
- Open `http://localhost:5173` in your browser.

---

## 🧪 Testing & Validation

Run unit and integration tests across data features and FastAPI serving endpoints:

```powershell
$env:PYTHONPATH="."
.venv\Scripts\pytest tests/test_serving/
```

Build and validate frontend TypeScript bundle:
```bash
npm --prefix frontend run build
```

---

## 🐳 Docker Deployment

Run the complete scoring stack via Docker Compose:

```bash
docker-compose -f docker/docker-compose.yml up --build
```

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
