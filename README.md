# 🏦 TCB Fraud Detection — MLOps Pipeline

> **End-to-end Machine Learning Operations system for real-time credit card fraud detection.**
>
> University Final Project — DDM501 | Python 3.10.9 | XGBoost 2.1.1 | FastAPI 0.112 | MLflow 2.14.3 | Evidently AI 0.4.39

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [ML Pipeline](#-ml-pipeline)
- [Model Performance](#-model-performance)
- [Feature Engineering](#-feature-engineering)
- [Quick Start](#-quick-start)
- [Docker Stack](#-docker-stack)
- [API Usage](#-api-usage)
- [Monitoring & Observability](#-monitoring--observability)
- [Responsible AI](#-responsible-ai)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Testing](#-testing)
- [Tech Stack](#-tech-stack)

---

## 🎯 Project Overview

This project implements a **production-grade MLOps pipeline** for detecting fraudulent credit card transactions at TCB (Techcombank). The system covers the full ML lifecycle from raw data ingestion to model serving, monitoring, and automated retraining.

| Stage | Description |
|---|---|
| **Data Ingestion** | Load and validate raw transaction data (CSV → schema guard → cleaned DataFrame) |
| **Preprocessing** | Clean, engineer features, time-based split (no data leakage) |
| **Training** | XGBoost with `scale_pos_weight` class imbalance handling + MLflow tracking |
| **Evaluation** | Threshold optimization (Recall ≥ 95%), SHAP explainability, per-segment fairness |
| **Serving** | FastAPI REST API: `/predict`, `/predict/batch`, `/health`, `/metrics`, `/docs` |
| **Monitoring** | Data drift (Evidently AI) + infrastructure metrics (Prometheus + Grafana) |
| **Orchestration** | Apache Airflow DAG — nightly retraining at 02:00 UTC |

### Dataset

| Metric | Value |
|---|---|
| Total transactions | 100,000 |
| Fraud transactions | 2,844 |
| Fraud rate | 2.84% |
| Raw features | 33 columns |
| Features (after engineering) | 35 (numeric, model-ready) |
| Customer segments | PRIVATE, PRIORITY, INSPIRE, MASS |

> ⚠️ **Class imbalance**: ~97:3 ratio. **Accuracy is never used as a metric.** Primary metric is **PR-AUC** (more robust than ROC-AUC for severely imbalanced datasets). We also track F1-Score, Precision, and Recall.

---

## 🏗 System Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw Data    │────▶│ Preprocessing│────▶│   Training   │────▶│  Evaluation  │
│  (CSV)       │     │ (preprocess) │     │  (XGBoost)   │     │ (SHAP/Fair)  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                       │
                     ┌──────────────┐     ┌──────────────┐            │
                     │  Monitoring  │◀────│  Serving API │◀───────────┘
                     │(Evidently +  │     │  (FastAPI)   │
                     │ Prometheus)  │     │              │
                     └──────────────┘     └──────┬───────┘
                            ▲                    │
                     ┌──────┴───────┐            │
                     │  Grafana     │            │
                     │  Dashboards  │     ┌──────▼───────┐
                     └──────────────┘     │ Orchestration│
                                          │  (Airflow)   │
                                          └──────────────┘
```

**MLflow** (artifact store: MinIO/S3) tracks every experiment run across training and evaluation phases.

---

## 📂 Project Structure

```
tcb-fraud-detection-mlops/
│
├── ml_pipeline/                    # 🧠 Core ML pipeline
│   ├── src/
│   │   ├── preprocess.py           #   683 lines — full preprocessing pipeline
│   │   ├── train.py                #   599 lines — XGBoost training + MLflow
│   │   ├── evaluate.py             #   518 lines — SHAP + fairness + threshold tuning
│   │   └── inference.py            #   588 lines — FraudDetector class (stateful engine)
│   ├── tests/
│   │   ├── test_preprocess.py      #   Unit tests for preprocessing (coverage >80%)
│   │   └── test_model.py           #   Unit tests for model training and evaluation
│   ├── requirements.txt            #   Pinned dependencies (pandas, xgboost, shap, mlflow...)
│   └── mlruns/                     #   MLflow experiment tracking (auto-generated)
│
├── data/
│   ├── raw/
│   │   └── tcb_credit_fraud_dataset.csv   # 100,000 transactions
│   └── processed/                  # Generated by preprocess.py
│       ├── train.parquet           #   80,000 rows (time-based split)
│       ├── test.parquet            #   20,000 rows (time-based split)
│       ├── features.json           #   Canonical feature list + count + created_at
│       ├── customer_stats.parquet  #   Per-customer tx count + avg amount (train-fitted)
│       ├── segment_label_map.json  #   customer_tier → int encoding
│       ├── amount_median_train.json#   Median amount for unseen-customer imputation
│       └── categorical_maps.json   #   Low-cardinality categorical encodings
│
├── models/                         # Generated by train.py + evaluate.py
│   ├── xgb_fraud_model.joblib      #   Serialized XGBoost model
│   ├── metrics.json                #   Training metrics (ROC-AUC, PR-AUC, F1...)
│   ├── feature_importance.csv      #   Feature importance by gain (top 35)
│   └── evaluation/
│       ├── evaluation.json         #   Optimal threshold + baseline comparison
│       ├── segment_report.csv      #   Per-segment fairness metrics
│       ├── pr_curve.png            #   PR curve + metrics-vs-threshold sweep
│       ├── shap_summary.png        #   SHAP beeswarm (top 20 global features)
│       └── shap_waterfall.png      #   SHAP waterfall for highest-risk prediction
│
├── serving_api/                    # 🚀 FastAPI serving layer
│   ├── app/
│   │   ├── main.py                 #   317 lines — 5 endpoints + CORS + lifespan
│   │   ├── model_loader.py         #   Singleton FraudDetector + env config
│   │   ├── observability.py        #   250 lines — 15+ Prometheus metric definitions
│   │   └── schemas.py              #   213 lines — Pydantic v2 request/response schemas
│   ├── tests/
│   │   ├── test_api.py             #   API endpoint integration tests
│   │   ├── test_loader.py          #   Model loader unit tests
│   │   └── conftest.py             #   Pytest fixtures
│   ├── Dockerfile                  #   Multi-stage build (python:3.10-slim)
│   └── requirements.txt            #   fastapi, uvicorn, prometheus-client...
│
├── monitoring/
│   ├── evidently_ai/
│   │   └── drift_monitor.py        #   DriftMonitor class — sliding-window drift detection
│   ├── prometheus/
│   │   └── prometheus.yml          #   Scrape config: FastAPI + node-exporter + cAdvisor
│   └── grafana/
│       ├── provisioning/           #   Auto-provisioned datasources + dashboards
│       └── dashboards/             #   Pre-built Grafana dashboard JSON definitions
│
├── dags/
│   └── fraud_pipeline.py           #   Airflow DAG: preprocess >> train >> evaluate
│                                   #   Schedule: cron "0 2 * * *" (daily at 02:00 UTC)
│
├── docs/
│   ├── api_docs.md                 #   Full API documentation
│   ├── architecture.drawio         #   Editable architecture diagram
│   └── cicd_proposal.md            #   Deployment flow + environment setup guide
│
├── .github/workflows/              #   GitHub Actions CI/CD (lint + test + build + deploy)
├── docker-compose.yml              #   8 services: fastapi, mlflow, minio, airflow,
│                                   #   prometheus, grafana, node-exporter, cadvisor
├── .env.example                    #   All required environment variables with defaults
├── .dvc/                           #   DVC configuration for data versioning
├── .flake8                         #   Linting configuration
├── Makefile                        #   CLI shortcuts
└── .gitignore
```

---

## 🧪 ML Pipeline

The pipeline is executed **sequentially**. Each stage produces artifacts consumed by the next.

### 1. Preprocessing (`preprocess.py`)

```bash
cd ml_pipeline && python src/preprocess.py
```

**Full pipeline order**: `analyze_dataset → load_dataset → validate_schema → clean_data → split_dataset → fit_feature_generators(train) → transform_features(train+test) → save_processed_data → save_feature_state → save_feature_metadata`

| Step | Function | Description |
|---|---|---|
| Analysis | `analyze_dataset` | Log class distribution, amount outliers, timestamp samples |
| Load | `load_dataset` | Parse CSV, cast `timestamp` to datetime64, `amount` to float |
| Validate | `validate_schema` | Assert required columns, non-null constraints, binary `is_fraud`, no duplicate `transaction_id` |
| Impute | `handle_missing_values` | `os` → `"UNKNOWN"`, `is_3d_secure` → `"N"`, numerics → `0` |
| Clean | `clean_data` | Remove duplicate transactions, encode Y/N/N/A → int8, encode APPROVED/DECLINED → int8 |
| Split | `split_dataset` | **Time-based** 80/20 chronological split (`train_ratio=0.8`) |
| Fit | `fit_feature_generators` | Customer aggregates + segment map + category maps — **fit on TRAIN ONLY** |
| Transform | `transform_features` | Apply to train then test independently (fit/transform pattern) |
| Save | `save_processed_data` | Export `train.parquet` + `test.parquet` |
| Persist | `save_feature_state` | 4 artifact files for inference reproducibility |

> 🔒 **Data leakage prevention**: `customer_tx_count` and `customer_avg_amount` are computed **only from training data**, then applied to the test set via a strict fit/transform pattern. Unseen customers in test/inference receive median imputation.

---

### 2. Training (`train.py`)

```bash
cd ml_pipeline && python src/train.py
```

**Model**: `XGBClassifier` with `binary:logistic` objective and `aucpr` eval metric.

**XGBoost Hyperparameters** (all logged to MLflow):

| Parameter | Value | Rationale |
|---|---|---|
| `objective` | `binary:logistic` | Binary fraud classification |
| `eval_metric` | `aucpr` | PR-AUC — primary metric for imbalanced data |
| `scale_pos_weight` | `n_negative / n_positive` ≈ 34 | Compensates for ~2.84% fraud rate |
| `n_estimators` | 1000 (early stopping on 50) | Prevents overfitting via early stop |
| `learning_rate` | 0.05 | Conservative step size |
| `max_depth` | 6 | Balanced tree complexity |
| `min_child_weight` | 5 | Regularization against rare patterns |
| `subsample` | 0.8 | Row sub-sampling per tree |
| `colsample_bytree` | 0.8 | Feature sub-sampling per tree |
| `gamma` | 1 | Minimum split loss (regularization) |
| `reg_alpha` | 0.1 | L1 regularization |
| `reg_lambda` | 1.0 | L2 regularization |
| `random_state` | 42 | Reproducibility |

**Threshold optimization** (`find_optimal_threshold`): sweeps the PR curve to find the highest-precision threshold satisfying **Recall ≥ 95%** AND threshold ≤ 0.70. Falls back to recall-only if strict constraint is unmet.

**Artifacts saved**: `xgb_fraud_model.joblib`, `metrics.json`, `feature_importance.csv` (gain-based, top 35).

All hyperparameters, metrics, model binary, and artifacts are logged to **MLflow** (`mlflow.xgboost.log_model`).

---

### 3. Evaluation (`evaluate.py`)

```bash
cd ml_pipeline && python src/evaluate.py
```

| Step | Function | Description |
|---|---|---|
| Threshold analysis | `evaluate_threshold` | Sweep PR curve, find optimal (Recall ≥ 95%). Plot PR curve + metrics-vs-threshold sweep |
| Fairness analysis | `evaluate_segments` | Per-segment Precision/Recall/F1/PR-AUC (PRIVATE, PRIORITY, INSPIRE, MASS + ALL) |
| Explainability | `explain_shap` | SHAP beeswarm (top 20 global) + waterfall for highest-risk transaction (sample: 2000) |
| Regression gate | `compare_baseline` | Compare current vs saved `metrics.json` with tolerances |
| Save report | `save_evaluation_report` | Export `evaluation.json` + `segment_report.csv` |

**Regression tolerance** (hard-coded guards):

| Metric | Max Allowed Degradation |
|---|---|
| PR-AUC | ≤ 2% |
| F1-Score | ≤ 3% |
| Recall | ≤ 2% |

All evaluation metrics and PNG artifacts are logged to **MLflow** (evaluation run).

---

### 4. Inference (`inference.py`)

The `FraudDetector` class is a **stateful inference engine** that loads all artifacts once and exposes a clean prediction API. The internal `_transform()` method **mirrors `transform_features()` in `preprocess.py` exactly**.

```python
from ml_pipeline.src.inference import FraudDetector

detector = FraudDetector("models", "data/processed")

# Single prediction
result = detector.predict_single({
    "transaction_id": "TX_001",
    "timestamp": "2026-03-14 10:23:00",
    "customer_id": "CUST_123",
    "amount": 5_000_000,
    "customer_tier": "PRIORITY",
    # ... remaining raw fields (all optional except the above)
})
# → {"transaction_id": "TX_001", "fraud_score": 0.023, "is_fraud_pred": False,
#    "threshold": 0.6788, "risk_level": "LOW"}

# Batch prediction
import pandas as pd
results_df = detector.predict_batch(pd.read_csv("new_transactions.csv"))
# → original DataFrame + ["fraud_score", "is_fraud_pred", "risk_level"]

# Health check
print(detector.health_check())
# → {"status": "OK", "feature_count": 35, "threshold": 0.6788, ...}
```

**Risk level classification** (independent of binary threshold):

| Risk Level | Fraud Score Range |
|---|---|
| LOW | [0.00 — 0.30) |
| MEDIUM | [0.30 — 0.60) |
| HIGH | [0.60 — 1.00] |

**Artifact dependencies** loaded by `FraudDetector` at startup:
- `models/xgb_fraud_model.joblib` — trained model
- `models/evaluation/evaluation.json` — optimal threshold
- `data/processed/features.json` — canonical feature list
- `data/processed/customer_stats.parquet` — train-fitted customer aggregates
- `data/processed/segment_label_map.json` — tier → int encoding
- `data/processed/amount_median_train.json` — imputation fallback
- `data/processed/categorical_maps.json` — categorical encodings

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| **PR-AUC** | 0.9999 |
| **ROC-AUC** | 0.9999 |
| **F1-Score** | 0.9912 |
| **Precision** | 0.9825 |
| **Recall** | 1.0000 |
| Best XGBoost Iteration | 373 / 1000 |

> **PR-AUC is the primary metric** — it is significantly more informative than ROC-AUC for datasets with severe class imbalance (2.84% fraud rate).

---

## 🔧 Feature Engineering

### Raw Features (33 columns from CSV)

**Identifiers** (excluded from model input): `transaction_id`, `customer_id`, `timestamp`, `is_fraud`

**Behavioural features** (pre-computed in dataset):

| Feature | Description |
|---|---|
| `tx_count_last_1h` | Number of transactions in the past 1 hour |
| `tx_count_last_24h` | Number of transactions in the past 24 hours |
| `time_since_last_tx_min` | Minutes since customer's last transaction |
| `avg_amount_last_30d` | Customer's average transaction amount last 30 days |
| `amount_ratio_vs_avg` | Current amount ratio vs 30-day average |
| `distance_from_home_km` | Geographic distance from customer's home |
| `is_new_device` | Whether transaction uses a new/unseen device |
| `is_new_merchant` | Whether merchant is new for this customer |
| `cvv_match` | CVV match status (Y/N/N/A → 1/0) |
| `is_3d_secure` | 3D Secure authentication used (Y/N → 1/0) |

**Categorical features** (label-encoded via train-fitted maps):

| Feature | Values |
|---|---|
| `card_type` | VISA, MASTERCARD, etc. |
| `card_tier` | GOLD, PLATINUM, etc. |
| `currency` | VND, USD, etc. |
| `merchant_category` | RETAIL, Transport, etc. |
| `merchant_country` | VN, US, etc. |
| `device_type` | Mobile, Desktop, etc. |
| `os` | iOS, Android, UNKNOWN |
| `ip_country` | VN, US, etc. |

**High-cardinality columns dropped**: `merchant_name`, `merchant_city`

### Derived Features (7 columns added by `transform_features`)

| Feature | Description |
|---|---|
| `transaction_hour` | Hour of day (0–23) from timestamp |
| `transaction_day_of_week` | Day of week (0=Monday, 6=Sunday) |
| `is_night_transaction` | 1 if hour ≥ 23 or hour ≤ 5 |
| `amount_log` | `log1p(amount)` — handles skewed distribution |
| `segment_encoded` | `customer_tier` → integer via train-fitted map |
| `customer_tx_count` | Customer's historical transaction count (train-fitted) |
| `customer_avg_amount` | Customer's historical average amount (train-fitted) |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10.9
- Docker & Docker Compose (for full stack)

### Installation

```bash
# Clone repository
git clone https://github.com/KienNT792/tcb-fraud-detection-mlops.git
cd tcb-fraud-detection-mlops

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install ML pipeline dependencies
pip install -r ml_pipeline/requirements.txt

# Install API dependencies
pip install -r serving_api/requirements.txt
```

### Run Full ML Pipeline

```bash
cd ml_pipeline

# Step 1 — Preprocess raw data
python src/preprocess.py

# Step 2 — Train XGBoost model (logs to MLflow)
python src/train.py

# Step 3 — Evaluate: SHAP + Fairness + Threshold tuning
python src/evaluate.py

# Step 4 — Run inference smoke test (assertions on real data)
python src/inference.py
```

### Launch API Only

```bash
cd serving_api
uvicorn app.main:app --reload --port 8000
# API → http://localhost:8000
# Swagger UI → http://localhost:8000/docs
# ReDoc → http://localhost:8000/redoc
```

---

## 🐳 Docker Stack

Copy `.env.example` to `.env` and (optionally) update credentials, then:

```bash
cp .env.example .env
docker compose up -d
```

### Services & Default Ports

| Service | Image | Port | Purpose |
|---|---|---|---|
| **fastapi** | `tcb-fraud-fastapi` | `8000` | Fraud prediction REST API |
| **mlflow** | `ghcr.io/mlflow/mlflow:v2.14.3` | `5000` | Experiment tracking UI |
| **minio** | `minio/minio` | `9000` (API), `9001` (Console) | S3-compatible artifact store |
| **airflow** | `apache/airflow:2.10.3-python3.10` | `8080` | DAG orchestration UI |
| **prometheus** | `prom/prometheus` | `9090` | Metrics scraping & storage |
| **grafana** | `grafana/grafana:12.4.1` | `3000` | Monitoring dashboards |
| **node-exporter** | `prom/node-exporter` | — | Host system metrics |
| **cadvisor** | `gcr.io/cadvisor/cadvisor` | `8081` | Container resource metrics |

All services communicate over the `tcb-mlops-network` Docker bridge network.

### Environment Variables (`.env`)

```bash
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123
MINIO_BUCKET=tcb-mlops-artifacts
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin123
FASTAPI_PORT=8000
MLFLOW_PORT=5000
AIRFLOW_PORT=8080
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
DRIFT_WINDOW_SIZE=500          # Sliding window size for drift detection
DRIFT_ALERT_THRESHOLD=0.2      # Drift score above this triggers alert
```

---

## 🌐 API Usage

Base URL: `http://localhost:8000` | Swagger: `/docs` | ReDoc: `/redoc`

### `GET /` — API Info

Returns basic metadata: name, version, available endpoints.

### `GET /health` — Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "OK",
  "model_type": "XGBClassifier",
  "feature_count": 35,
  "threshold": 0.6788,
  "best_iteration": 372,
  "loaded_at": "2026-03-14T10:00:00+00:00",
  "api_version": "1.0.0"
}
```

Use this endpoint for **liveness/readiness probes** in Docker or Kubernetes.

### `POST /predict` — Single Transaction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TX_001",
    "timestamp": "2026-03-14 10:23:00",
    "customer_id": "CUST_12345",
    "amount": 350000,
    "customer_tier": "PRIORITY",
    "card_type": "VISA",
    "currency": "VND",
    "merchant_category": "Transport",
    "merchant_country": "VN",
    "cvv_match": "Y",
    "is_3d_secure": "Y",
    "transaction_status": "APPROVED",
    "tx_count_last_1h": 1,
    "tx_count_last_24h": 3
  }'
```

**Required fields**: `transaction_id`, `timestamp`, `customer_id`, `amount`, `customer_tier`
**Optional fields**: All other fields — missing values are handled gracefully (imputed to 0 or "UNKNOWN").

**Response:**

```json
{
  "transaction_id": "TX_001",
  "fraud_score": 0.0231,
  "is_fraud_pred": false,
  "threshold": 0.6788,
  "risk_level": "LOW"
}
```

| Response Field | Type | Description |
|---|---|---|
| `fraud_score` | float [0,1] | Raw fraud probability from XGBoost |
| `is_fraud_pred` | bool | `true` if `fraud_score ≥ threshold` |
| `threshold` | float | Decision threshold from `evaluation.json` |
| `risk_level` | string | `LOW` / `MEDIUM` / `HIGH` (score-based bands) |

### `POST /predict/batch` — Batch Scoring

```json
{
  "transactions": [
    { "transaction_id": "TX_001", "timestamp": "...", "customer_id": "...", "amount": 350000, "customer_tier": "PRIORITY" },
    { "transaction_id": "TX_002", "timestamp": "...", "customer_id": "...", "amount": 5000000, "customer_tier": "MASS" }
  ]
}
```

**Response:**

```json
{
  "total": 2,
  "fraud_detected": 1,
  "fraud_rate": 0.5,
  "threshold": 0.6788,
  "predictions": [
    { "transaction_id": "TX_001", "fraud_score": 0.021, "is_fraud_pred": false, "risk_level": "LOW" },
    { "transaction_id": "TX_002", "fraud_score": 0.892, "is_fraud_pred": true,  "risk_level": "HIGH" }
  ]
}
```

Accepts **up to 1,000 transactions** per request (enforced by Pydantic `max_length=1000`).

### `GET /metrics` — Prometheus Metrics

Exposes runtime metrics for Prometheus scraping (auto-instrumented via `prometheus-fastapi-instrumentator`).

---

## 📈 Monitoring & Observability

### Prometheus Metrics (`observability.py`)

The API exposes **15+ custom Prometheus metrics** across 4 categories:

**HTTP Layer:**

| Metric | Type | Description |
|---|---|---|
| `tcb_http_requests_total` | Counter | Total requests by method/path/status |
| `tcb_http_request_duration_seconds` | Histogram | Request latency (buckets: 10ms–5s) |

**Model & Predictions:**

| Metric | Type | Description |
|---|---|---|
| `tcb_model_loaded` | Gauge | 1 if FraudDetector is ready |
| `tcb_model_threshold` | Gauge | Current decision threshold |
| `tcb_model_feature_count` | Gauge | Number of features in loaded model |
| `tcb_prediction_requests_total` | Counter | API calls by endpoint |
| `tcb_prediction_samples_total` | Counter | Transactions scored by endpoint |
| `tcb_prediction_fraud_total` | Counter | Transactions predicted as fraud |
| `tcb_prediction_risk_level_total` | Counter | Predictions by risk level `[LOW/MEDIUM/HIGH]` |
| `tcb_prediction_score` | Histogram | Fraud score distribution (10 buckets) |
| `tcb_prediction_amount_vnd` | Histogram | Transaction amount distribution (VND) |

**Data Drift (Evidently AI):**

| Metric | Type | Description |
|---|---|---|
| `tcb_drift_baseline_ready` | Gauge | 1 if drift monitor has reference window |
| `tcb_drift_reference_samples` | Gauge | Samples in reference window |
| `tcb_drift_current_samples` | Gauge | Samples in live window |
| `tcb_drift_features_alerting` | Gauge | Features above drift alert threshold |
| `tcb_drift_feature_score` | Gauge | Per-feature drift score (labeled) |

### Drift Detection (`drift_monitor.py`)

`DriftMonitor` uses an **Evidently AI sliding-window** approach:

- **Reference window**: Loaded from `train.parquet` or filled from warm-up traffic
- **Current window**: Sliding window of recent API requests (`DRIFT_WINDOW_SIZE=500`)
- **Alert threshold**: `DRIFT_ALERT_THRESHOLD=0.2` (configurable via `.env`)
- Automatic bootstrap from `train.parquet` on API startup

### Grafana Dashboards

Pre-provisioned dashboards are available at `http://localhost:3000` (default: `admin/admin123`). Datasource (Prometheus) is auto-configured via provisioning YAML.

---

## 🤖 Responsible AI

This project adheres to **Responsible AI** principles as mandated by the DDM501 grading criteria:

### Explainability (SHAP)

- **Global** (`shap_summary.png`): SHAP beeswarm plot revealing the top 20 features most influencing fraud predictions across 2,000 sampled test transactions (`shap.TreeExplainer`).
- **Local** (`shap_waterfall.png`): SHAP waterfall plot explaining *why* the highest-scoring (most suspicious) transaction was flagged — showing each feature's contribution.

```
models/evaluation/shap_summary.png    # Global feature importance
models/evaluation/shap_waterfall.png  # Local single-instance explanation
```

### Fairness

Model performance is evaluated **per customer segment** to detect disparate impact:

| Segment | Metrics Tracked |
|---|---|
| PRIVATE | Precision, Recall, F1, PR-AUC, fraud rate |
| PRIORITY | Precision, Recall, F1, PR-AUC, fraud rate |
| INSPIRE | Precision, Recall, F1, PR-AUC, fraud rate |
| MASS | Precision, Recall, F1, PR-AUC, fraud rate |
| **ALL** | Aggregate across all segments |

Artifact: `models/evaluation/segment_report.csv`

### Regression Gate (Model Quality Guard)

Before any model is accepted, `compare_baseline()` validates it against the previously saved `metrics.json`:

| Metric | Max Regression Allowed |
|---|---|
| PR-AUC | ≤ 2 percentage points |
| F1-Score | ≤ 3 percentage points |
| Recall | ≤ 2 percentage points |

Gate result (`PASS` / `FAIL`) is saved in `evaluation.json` and tagged in MLflow.

---

## 🚢 CI/CD Pipeline

### GitHub Actions (`.github/workflows/ci-cd-pipeline.yml`)

Full CI/CD pipeline triggered on **push to `main` / `dev/ver2`** and **pull requests**:

#### CI Job (runs on every push/PR)

| Stage | Description |
|---|---|
| **Checkout** | `actions/checkout@v4` |
| **Setup Python** | Python 3.10 via `actions/setup-python@v5` |
| **Install Dependencies** | `pip install flake8 pytest pytest-cov` + project requirements |
| **Lint** | `flake8 ml_pipeline/src serving_api/app dags` |
| **Test Preprocessing** | `pytest test_preprocess.py --cov-fail-under=80` |
| **Test Model Pipeline** | `pytest test_model.py --cov-fail-under=80` (train + evaluate + inference) |
| **Test Serving API** | `pytest serving_api/tests --cov-fail-under=80` |
| **Build Docker Image** | `docker build --file serving_api/Dockerfile` → tag `tcb-fraud-fastapi:<sha>` |

#### CD Job (only on push to `main` / `dev/ver2`)

| Stage | Description |
|---|---|
| **Deploy to GCP VPS** | SSH → `git pull` → `docker compose up -d --build` |
| **Health Checks** | Poll FastAPI, MLflow, Airflow, Grafana — up to 10 retries × 10s |

Required config: `SSH_DEPLOY_KEY` in Secrets, plus `GCP_DEPLOY_HOST`, `GCP_DEPLOY_USER`, `GIT_REPO_URL` in either Secrets or Variables.

---

## ⏰ Airflow Orchestration

DAG: `fraud_detection_training_pipeline`
Schedule: **`0 2 * * *`** (daily at 02:00 UTC)
Max active runs: 1 (no concurrent retraining)

```python
check_model_quality >> [preprocess, skip_retraining]
preprocess >> train >> evaluate >> stage_candidate >> verify_candidate
```

| Task | Type | Description |
|---|---|---|
| `check_model_quality` | BranchPythonOperator | Query MLflow `eval_f1` vs threshold (0.80) |
| `preprocess` | BashOperator | Run `preprocess.py` |
| `train` | BashOperator | Run `train.py` + MLflow Registry |
| `evaluate` | BashOperator | Run `evaluate.py` + SHAP |
| `stage_candidate` | PythonOperator | Copy artifacts to candidate dir + write manifest |
| `verify_candidate` | PythonOperator | Poll candidate FastAPI `/health` to confirm model loaded |

---

## ✅ Testing

```bash
# Run preprocessing unit tests
pytest ml_pipeline/tests/test_preprocess.py -v

# Run model/evaluation tests
pytest ml_pipeline/tests/test_model.py -v

# Run API integration tests
pytest serving_api/tests/test_api.py -v

# Run model loader tests
pytest serving_api/tests/test_loader.py -v

# Full coverage report — MUST exceed 80%
pytest ml_pipeline/tests/ --cov=ml_pipeline.src --cov-report=term-missing --cov-fail-under=80
pytest serving_api/tests/ --cov=serving_api.app --cov-report=term-missing --cov-fail-under=80
```

### Coverage Requirements

| Component | Target | Enforced By |
|---|---|---|
| `ml_pipeline.src` | > 80% | `pytest --cov-fail-under=80` |
| `serving_api.app` | > 80% | `pytest --cov-fail-under=80` |
| GitHub Actions CI | > 80% | `--cov-fail-under=80` gate (blocks deployment) |

---

## 🛠 Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.10.9 |
| ML Model | XGBoost | 2.1.1 |
| Data Processing | pandas, numpy, pyarrow | 2.2.2, 1.26.4, 15.0.2 |
| Explainability | SHAP, LIME | 0.46.0, 0.2.0.1 |
| Experiment Tracking | MLflow | 2.14.3 |
| Data Versioning | DVC | 3.51.2 |
| API Framework | FastAPI | 0.112.0 |
| API Server | Uvicorn | 0.30.6 |
| Data Validation | Pydantic v2 | 2.8.2 |
| Containerization | Docker, Docker Compose | multi-stage build |
| Artifact Storage | MinIO (S3-compatible) | latest |
| Data Monitoring | Evidently AI | 0.4.39 |
| Metrics Collection | Prometheus, prometheus-client | — , 0.21.0 |
| Dashboarding | Grafana | 12.4.1 |
| Orchestration | Apache Airflow | 2.10.3-python3.10 |
| CI/CD | GitHub Actions | CI + CD (deploy to GCP VPS) |
| Testing | pytest, pytest-cov | 8.2.2, 5.0.0 |
| HTTP Testing | httpx (FastAPI TestClient) | 0.27.0 |
| Linting | flake8 | — |

---

## 👥 Authors

- **Nguyen Trung Kien** — DDM501 Final Project

---

## 📄 License

This project is developed for academic purposes (DDM501 — University Final Project).
