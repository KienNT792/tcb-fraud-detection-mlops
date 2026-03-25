# 🏦 TCB Fraud Detection — End-to-End MLOps Pipeline

> **Production-grade Machine Learning Operations system for real-time credit card fraud detection with canary deployment, automated rollback, and continuous monitoring.**
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
- [Model Registry & Runtime Bundle](#-model-registry--runtime-bundle)
- [Quick Start](#-quick-start)
- [Docker Stack](#-docker-stack)
- [API Usage](#-api-usage)
- [Canary Deployment & Rollout Control](#-canary-deployment--rollout-control)
- [Monitoring & Observability](#-monitoring--observability)
- [Alerting & Auto-Rollback](#-alerting--auto-rollback)
- [Traffic Simulator](#-traffic-simulator)
- [Responsible AI](#-responsible-ai)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Airflow Orchestration](#-airflow-orchestration)
- [Testing](#-testing)
- [Makefile Shortcuts](#-makefile-shortcuts)
- [Tech Stack](#-tech-stack)
- [Authors](#-authors)
- [License](#-license)

---

## 🎯 Project Overview

This project implements a **production-grade MLOps pipeline** for detecting fraudulent credit card transactions at TCB (Techcombank). The system covers the **full ML lifecycle** — from raw data ingestion to model serving, canary deployment, continuous monitoring, and automated rollback.

| Stage                 | Description                                                                                |
| --------------------- | ------------------------------------------------------------------------------------------ |
| **Data Ingestion**    | Load and validate raw transaction data (CSV → schema guard → cleaned DataFrame)            |
| **Preprocessing**     | Clean, engineer features, time-based split (no data leakage)                               |
| **Training**          | XGBoost with `scale_pos_weight` class imbalance handling + MLflow tracking                 |
| **Evaluation**        | Threshold optimization (Recall ≥ 95%), SHAP explainability, per-segment fairness           |
| **Registry**          | MLflow Model Registry: version, stage (Staging → Production), runtime bundle packaging     |
| **Serving**           | FastAPI REST API: `/predict`, `/predict/batch`, `/health`, `/metrics`, `/monitoring/drift` |
| **Canary Deployment** | Nginx load balancer: split traffic between stable and candidate model containers           |
| **Monitoring**        | Data drift (Evidently AI) + infrastructure metrics (Prometheus + Grafana) + alerting rules |
| **Auto-Rollback**     | Alertmanager → rollback receiver → GitHub Actions workflow dispatch (automatic)            |
| **Orchestration**     | Apache Airflow DAG — nightly retraining at 02:00 UTC with candidate staging                |

### Dataset

| Metric                       | Value                            |
| ---------------------------- | -------------------------------- |
| Total transactions           | 100,000                          |
| Fraud transactions           | 2,844                            |
| Fraud rate                   | 2.84%                            |
| Raw features                 | 33 columns                       |
| Features (after engineering) | 35 (numeric, model-ready)        |
| Customer segments            | PRIVATE, PRIORITY, INSPIRE, MASS |

> ⚠️ **Class imbalance**: ~97:3 ratio. **Accuracy is never used as a metric.** Primary metric is **PR-AUC** (more robust than ROC-AUC for severely imbalanced datasets). We also track F1-Score, Precision, and Recall.

---

## 🏗 System Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw Data    │────▶│ Preprocessing│────▶│   Training   │────▶│  Evaluation  │
│  (CSV/DVC)   │     │ (preprocess) │     │  (XGBoost)   │     │ (SHAP/Fair)  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                       │
                                          ┌────────────────────────────┘
                                          ▼
                      ┌──────────────────────────────────┐
                      │     MLflow Model Registry        │
                      │  (Version → Stage → Bundle)      │
                      └──────────┬───────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
            ┌──────────────┐          ┌──────────────┐
            │ FastAPI      │          │ FastAPI      │
            │ (Stable)     │          │ (Candidate)  │
            └──────┬───────┘          └──────┬───────┘
                   │                         │
            ┌──────┴─────────────────────────┴──────┐
            │          Nginx Load Balancer           │
            │     (Canary Traffic Splitting)         │
            └──────────────────┬────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                      ▼
  ┌──────────────┐     ┌──────────────┐      ┌──────────────┐
  │  Prometheus  │     │  Evidently   │      │  Alertmanager│
  │  (Metrics)   │     │  (Drift)     │      │  + Rollback  │
  └──────┬───────┘     └──────────────┘      │  Receiver    │
         │                                   └──────┬───────┘
  ┌──────▼───────┐                           ┌──────▼───────┐
  │   Grafana    │                           │ GitHub Action │
  │ (Dashboards) │                           │ (Auto-Rollbk)│
  └──────────────┘                           └──────────────┘

  ┌──────────────┐     ┌──────────────┐      ┌──────────────┐
  │  Airflow     │     │    MinIO     │      │    DVC       │
  │ (Scheduling) │     │ (S3 Artifacts│      │ (Data Version│
  └──────────────┘     └──────────────┘      └──────────────┘
```

**MLflow** (artifact store: MinIO/S3) tracks every experiment run. **Nginx** splits traffic between stable and candidate model containers for canary rollouts.

---

## 📂 Project Structure

```
tcb-fraud-detection-mlops/
│
├── ml_pipeline/                        # 🧠 Core ML pipeline
│   ├── src/
│   │   ├── preprocess.py               #   Full preprocessing pipeline
│   │   ├── train.py                    #   XGBoost training + MLflow logging
│   │   ├── evaluate.py                 #   SHAP + fairness + threshold tuning
│   │   ├── inference.py                #   FraudDetector stateful inference engine
│   │   ├── model_registry.py           #   MLflow Model Registry operations
│   │   ├── promote_model.py            #   CLI: promote model version to target stage
│   │   ├── runtime_bundle.py           #   Package model + artifacts into MLflow bundle
│   │   ├── registry_metadata.py        #   Registry metadata read/write helpers
│   │   ├── mlflow_utils.py             #   MLflow tracking URI configuration
│   │   └── logging_config.py           #   Structured logging setup
│   ├── tests/
│   │   ├── test_preprocess.py          #   Preprocessing unit tests
│   │   ├── test_model.py               #   Training + evaluation + inference tests
│   │   └── test_coverage.py            #   Coverage enforcement tests
│   └── requirements.txt                #   Pinned ML dependencies
│
├── serving_api/                        # 🚀 FastAPI serving layer
│   ├── app/
│   │   ├── main.py                     #   5+ endpoints + CORS + lifespan
│   │   ├── model_loader.py             #   Singleton FraudDetector + env config
│   │   ├── observability.py            #   15+ Prometheus metric definitions
│   │   └── schemas.py                  #   Pydantic v2 request/response schemas
│   ├── tests/
│   │   ├── test_api.py                 #   API endpoint integration tests
│   │   ├── test_loader.py              #   Model loader unit tests
│   │   └── conftest.py                 #   Pytest fixtures
│   ├── Dockerfile                      #   Multi-stage build (python:3.10-slim)
│   └── requirements.txt                #   FastAPI, Uvicorn, Prometheus...
│
├── monitoring/                         # Full observability stack
│   ├── evidently_ai/
│   │   └── drift_monitor.py            #   DriftMonitor: sliding-window drift detection
│   ├── prometheus/
│   │   ├── prometheus.yml              #   Scrape config: FastAPI, Nginx, node, cAdvisor
│   │   └── alerts.yml                  #   6 alert rules (API, drift, canary)
│   ├── alertmanager/
│   │   └── alertmanager.yml            #   Alert routing → rollback receiver
│   ├── automation/
│   │   └── alertmanager_rollback_receiver.py  # Auto-rollback via GitHub API
│   ├── loadbalancer/
│   │   ├── nginx.conf                  #   Canary traffic routing (stable/candidate)
│   │   └── canary_split.conf           #   Weighted traffic split configuration
│   ├── simulator/                      #   Traffic simulation framework
│   │   ├── simulator.py                #   FraudPredictionSimulator engine
│   │   ├── fraud_data_generator.py     #   Synthetic fraud data generator
│   │   ├── rollout_ctl.py              #   CLI: canary set/rollback/promote/status
│   │   ├── common.py                   #   Shared rollout & deployment utilities
│   │   ├── sim_config.yaml             #   4 drift scenarios (normal → severe)
│   │   └── scenarios/                  #   8 scripted simulation scenarios
│   │       ├── baseline_5rps.py        #     Normal traffic at 5 RPS
│   │       ├── high_volume_burst.py    #     Burst traffic test
│   │       ├── traffic_ramp.py         #     Gradual traffic ramp-up
│   │       ├── feature_drift_5rps.py   #     Drift injection at 5 RPS
│   │       ├── candidate_soak_test.py  #     Soak test for candidate model
│   │       ├── auto_retrain_trigger.py #     End-to-end retrain + stage
│   │       ├── train_and_stage_candidate.py  # Train & stage new candidate
│   │       └── full_cycle_drift_to_rollout.py  # Full drift-to-rollback cycle
│   └── grafana/
│       ├── provisioning/               #   Auto-provisioned datasources
│       └── dashboards/                 #   Pre-built Grafana dashboard JSON
│
├── dags/
│   └── fraud_pipeline.py              #   Airflow DAG: retrain + canary staging
│
├── scripts/
│   ├── deploy_vps.sh                  #   GCP VPS deployment script
│   ├── setup_vps_prereqs.sh           #   VPS prerequisite installer
│   └── runtime_bundle_registry.py     #   CLI: publish/download/bootstrap bundles
│
├── bootstrap_runtime_bundle/          #   Fallback model artifacts for cold start
│   ├── mlflow_model/                  #   Pre-trained model checkpoint
│   └── processed/                     #   Pre-computed feature artifacts
│
├── data/
│   ├── raw/
│   │   └── tcb_credit_fraud_dataset.csv  # 100,000 transactions (DVC-tracked)
│   └── processed/                     # Generated by preprocess.py
│       ├── train.parquet              #   80,000 rows (time-based split)
│       ├── test.parquet               #   20,000 rows
│       ├── features.json              #   Canonical feature list
│       ├── customer_stats.parquet     #   Per-customer aggregates (train-fitted)
│       ├── segment_label_map.json     #   customer_tier → int encoding
│       ├── amount_median_train.json   #   Median for unseen-customer imputation
│       └── categorical_maps.json      #   Low-cardinality categorical encodings
│
├── models/                            # Generated by train.py + evaluate.py
│   ├── xgb_fraud_model.joblib         #   Serialized XGBoost model
│   ├── metrics.json                   #   Training metrics
│   ├── feature_importance.csv         #   Feature importance by gain (top 35)
│   └── evaluation/
│       ├── evaluation.json            #   Optimal threshold + baseline comparison
│       ├── segment_report.csv         #   Per-segment fairness metrics
│       ├── pr_curve.png               #   PR curve + metrics-vs-threshold
│       ├── shap_summary.png           #   SHAP beeswarm (top 20 global)
│       └── shap_waterfall.png         #   SHAP waterfall for highest-risk prediction
│
├── docs/
│   ├── api_docs.md                    #   Full API documentation
│   ├── architecture.drawio            #   Editable architecture diagram
│   ├── cicd_proposal.md               #   CI/CD design & deployment flow
│   ├── cloud_ssh_setup.md             #   GCP SSH configuration guide
│   └── overview.md                    #   Project overview document
│
├── .github/workflows/
│   ├── ci-cd-pipeline.yml             #   CI (lint + test + coverage) + CD (build + deploy)
│   ├── promote-model.yml              #   Manual: promote model version in MLflow Registry
│   └── rollback-canary.yml            #   Manual/Auto: canary rollback/promote/set-canary
│
├── docker-compose.yml                 #   14 services (full production stack)
├── dvc.yaml                           #   DVC pipeline: preprocess → train → evaluate
├── params.yaml                        #   Pipeline parameters (rollout, monitoring thresholds)
├── Makefile                           #   Developer CLI shortcuts
├── .env.example                       #   All required environment variables
├── .flake8                            #   Linting configuration
├── .dvc/                              #   DVC configuration for data versioning
└── .gitignore
```

---

## 🧪 ML Pipeline

The pipeline is executed **sequentially**. Each stage produces artifacts consumed by the next. Data versioning is handled by **DVC** (`dvc.yaml`).

### 1. Preprocessing (`preprocess.py`)

```bash
python -m ml_pipeline.src.preprocess
```

**Full pipeline order**: `analyze_dataset → load_dataset → validate_schema → clean_data → split_dataset → fit_feature_generators(train) → transform_features(train+test) → save_processed_data → save_feature_state → save_feature_metadata`

| Step      | Function                 | Description                                                                                     |
| --------- | ------------------------ | ----------------------------------------------------------------------------------------------- |
| Analysis  | `analyze_dataset`        | Log class distribution, amount outliers, timestamp samples                                      |
| Load      | `load_dataset`           | Parse CSV, cast `timestamp` to datetime64, `amount` to float                                    |
| Validate  | `validate_schema`        | Assert required columns, non-null constraints, binary `is_fraud`, no duplicate `transaction_id` |
| Impute    | `handle_missing_values`  | `os` → `"UNKNOWN"`, `is_3d_secure` → `"N"`, numerics → `0`                                      |
| Clean     | `clean_data`             | Remove duplicate transactions, encode Y/N/N/A → int8, encode APPROVED/DECLINED → int8           |
| Split     | `split_dataset`          | **Time-based** 80/20 chronological split (`train_ratio=0.8`)                                    |
| Fit       | `fit_feature_generators` | Customer aggregates + segment map + category maps — **fit on TRAIN ONLY**                       |
| Transform | `transform_features`     | Apply to train then test independently (fit/transform pattern)                                  |
| Save      | `save_processed_data`    | Export `train.parquet` + `test.parquet`                                                         |
| Persist   | `save_feature_state`     | 4 artifact files for inference reproducibility                                                  |

> 🔒 **Data leakage prevention**: `customer_tx_count` and `customer_avg_amount` are computed **only from training data**, then applied to the test set via a strict fit/transform pattern. Unseen customers in test/inference receive median imputation.

---

### 2. Training (`train.py`)

```bash
python -m ml_pipeline.src.train
```

**Model**: `XGBClassifier` with `binary:logistic` objective and `aucpr` eval metric.

**XGBoost Hyperparameters** (all logged to MLflow):

| Parameter          | Value                          | Rationale                                   |
| ------------------ | ------------------------------ | ------------------------------------------- |
| `objective`        | `binary:logistic`              | Binary fraud classification                 |
| `eval_metric`      | `aucpr`                        | PR-AUC — primary metric for imbalanced data |
| `scale_pos_weight` | `n_negative / n_positive` ≈ 34 | Compensates for ~2.84% fraud rate           |
| `n_estimators`     | 1000 (early stopping on 50)    | Prevents overfitting via early stop         |
| `learning_rate`    | 0.05                           | Conservative step size                      |
| `max_depth`        | 6                              | Balanced tree complexity                    |
| `min_child_weight` | 5                              | Regularization against rare patterns        |
| `subsample`        | 0.8                            | Row sub-sampling per tree                   |
| `colsample_bytree` | 0.8                            | Feature sub-sampling per tree               |
| `gamma`            | 1                              | Minimum split loss (regularization)         |
| `reg_alpha`        | 0.1                            | L1 regularization                           |
| `reg_lambda`       | 1.0                            | L2 regularization                           |
| `random_state`     | 42                             | Reproducibility                             |

**Threshold optimization** (`find_optimal_threshold`): sweeps the PR curve to find the highest-precision threshold satisfying **Recall ≥ 95%** AND threshold ≤ 0.70. Falls back to recall-only if strict constraint is unmet.

**Artifacts saved**: `xgb_fraud_model.joblib`, `metrics.json`, `feature_importance.csv` (gain-based, top 35).

All hyperparameters, metrics, model binary, and runtime bundle are logged to **MLflow** (`mlflow.xgboost.log_model`).

---

### 3. Evaluation (`evaluate.py`)

```bash
python -m ml_pipeline.src.evaluate
```

| Step               | Function                 | Description                                                                             |
| ------------------ | ------------------------ | --------------------------------------------------------------------------------------- |
| Threshold analysis | `evaluate_threshold`     | Sweep PR curve, find optimal (Recall ≥ 95%). Plot PR curve + metrics-vs-threshold sweep |
| Fairness analysis  | `evaluate_segments`      | Per-segment Precision/Recall/F1/PR-AUC (PRIVATE, PRIORITY, INSPIRE, MASS + ALL)         |
| Explainability     | `explain_shap`           | SHAP beeswarm (top 20 global) + waterfall for highest-risk transaction (sample: 2000)   |
| Regression gate    | `compare_baseline`       | Compare current vs saved `metrics.json` with tolerances                                 |
| Save report        | `save_evaluation_report` | Export `evaluation.json` + `segment_report.csv`                                         |

**Regression tolerance** (hard-coded guards):

| Metric   | Max Allowed Degradation |
| -------- | ----------------------- |
| PR-AUC   | ≤ 2%                    |
| F1-Score | ≤ 3%                    |
| Recall   | ≤ 2%                    |

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
| ---------- | ----------------- |
| LOW        | [0.00 — 0.30)     |
| MEDIUM     | [0.30 — 0.60)     |
| HIGH       | [0.60 — 1.00]     |

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

| Metric                 | Score      |
| ---------------------- | ---------- |
| **PR-AUC**             | 0.9999     |
| **ROC-AUC**            | 0.9999     |
| **F1-Score**           | 0.9912     |
| **Precision**          | 0.9825     |
| **Recall**             | 1.0000     |
| Best XGBoost Iteration | 373 / 1000 |

**Confusion Matrix** (20,000 test transactions):

|                  | Predicted Legit | Predicted Fraud |
| ---------------- | --------------- | --------------- |
| **Actual Legit** | 19,427          | 10              |
| **Actual Fraud** | 0               | 563             |

> **PR-AUC is the primary metric** — it is significantly more informative than ROC-AUC for datasets with severe class imbalance (2.84% fraud rate).

---

## 🔧 Feature Engineering

### Raw Features (33 columns from CSV)

**Identifiers** (excluded from model input): `transaction_id`, `customer_id`, `timestamp`, `is_fraud`

**Behavioural features** (pre-computed in dataset):

| Feature                  | Description                                        |
| ------------------------ | -------------------------------------------------- |
| `tx_count_last_1h`       | Number of transactions in the past 1 hour          |
| `tx_count_last_24h`      | Number of transactions in the past 24 hours        |
| `time_since_last_tx_min` | Minutes since customer's last transaction          |
| `avg_amount_last_30d`    | Customer's average transaction amount last 30 days |
| `amount_ratio_vs_avg`    | Current amount ratio vs 30-day average             |
| `distance_from_home_km`  | Geographic distance from customer's home           |
| `is_new_device`          | Whether transaction uses a new/unseen device       |
| `is_new_merchant`        | Whether merchant is new for this customer          |
| `cvv_match`              | CVV match status (Y/N/N/A → 1/0)                   |
| `is_3d_secure`           | 3D Secure authentication used (Y/N → 1/0)          |

**Categorical features** (label-encoded via train-fitted maps):

| Feature             | Values                  |
| ------------------- | ----------------------- |
| `card_type`         | VISA, MASTERCARD, etc.  |
| `card_tier`         | GOLD, PLATINUM, etc.    |
| `currency`          | VND, USD, etc.          |
| `merchant_category` | RETAIL, Transport, etc. |
| `merchant_country`  | VN, US, etc.            |
| `device_type`       | Mobile, Desktop, etc.   |
| `os`                | iOS, Android, UNKNOWN   |
| `ip_country`        | VN, US, etc.            |

**High-cardinality columns dropped**: `merchant_name`, `merchant_city`

### Derived Features (7 columns added by `transform_features`)

| Feature                   | Description                                            |
| ------------------------- | ------------------------------------------------------ |
| `transaction_hour`        | Hour of day (0–23) from timestamp                      |
| `transaction_day_of_week` | Day of week (0=Monday, 6=Sunday)                       |
| `is_night_transaction`    | 1 if hour ≥ 23 or hour ≤ 5                             |
| `amount_log`              | `log1p(amount)` — handles skewed distribution          |
| `segment_encoded`         | `customer_tier` → integer via train-fitted map         |
| `customer_tx_count`       | Customer's historical transaction count (train-fitted) |
| `customer_avg_amount`     | Customer's historical average amount (train-fitted)    |

---

## 📦 Model Registry & Runtime Bundle

### MLflow Model Registry

The system uses **MLflow Model Registry** to manage model versions and lifecycle stages:

| Module                 | Purpose                                                                  |
| ---------------------- | ------------------------------------------------------------------------ |
| `model_registry.py`    | Register model from run, transition stage, find version by run/stage     |
| `promote_model.py`     | CLI tool: promote a model version to Staging/Production                  |
| `runtime_bundle.py`    | Package model + processed artifacts into a single MLflow artifact bundle |
| `registry_metadata.py` | Read/write registry metadata for deployment tracking                     |

**Lifecycle Stages**: `None` → `Staging` → `Production` → `Archived`

### Runtime Bundle

A **runtime bundle** packages everything needed to serve a model version:

```
runtime_bundle/
├── models/
│   ├── xgb_fraud_model.joblib
│   ├── metrics.json
│   ├── feature_importance.csv
│   └── evaluation/
│       └── evaluation.json
├── processed/
│   ├── features.json
│   ├── customer_stats.parquet
│   ├── segment_label_map.json
│   ├── amount_median_train.json
│   └── categorical_maps.json
└── metadata.json
```

The `runtime_bundle_registry.py` CLI supports:

- `publish` — Upload local artifacts as a runtime bundle to MLflow
- `download` — Download a bundle by stage or version
- `bootstrap` — Cold-start: upload local artifacts and register in MLflow

### Bootstrap Runtime Bundle

The `bootstrap_runtime_bundle/` directory contains pre-trained model artifacts for **cold-start deployment** — when the MLflow registry is empty, the system falls back to these artifacts to bootstrap the first model version.

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
# Step 1 — Preprocess raw data
python -m ml_pipeline.src.preprocess

# Step 2 — Train XGBoost model (logs to MLflow)
python -m ml_pipeline.src.train

# Step 3 — Evaluate: SHAP + Fairness + Threshold tuning
python -m ml_pipeline.src.evaluate

# Step 4 — Run inference smoke test (assertions on real data)
python -m ml_pipeline.src.inference
```

Or use **DVC** to reproduce the full pipeline:

```bash
dvc repro
```

### Launch API Only

```bash
uvicorn serving_api.app.main:app --host 0.0.0.0 --port 8000 --reload
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

The `docker-compose.yml` defines **14 services** organized into 5 layers:

#### Serving Layer

| Service               | Image                         | Port   | Purpose                                         |
| --------------------- | ----------------------------- | ------ | ----------------------------------------------- |
| **fastapi-stable**    | `tungb12ok/tcb-detect-credit` | `8002` | Stable model serving                            |
| **fastapi-candidate** | `tungb12ok/tcb-detect-credit` | `8003` | Candidate model serving (profile: `candidate`)  |
| **loadbalancer**      | `nginx:1.27-alpine`           | `8000` | Canary traffic routing between stable/candidate |

#### ML Platform

| Service        | Image                              | Port          | Purpose                                 |
| -------------- | ---------------------------------- | ------------- | --------------------------------------- |
| **mlflow**     | `ghcr.io/mlflow/mlflow:v2.14.3`    | `5000`        | Experiment tracking + Model Registry UI |
| **minio**      | `minio/minio`                      | `9000`/`9001` | S3-compatible artifact store            |
| **minio-init** | `minio/mc`                         | —             | Auto-create MinIO bucket                |
| **airflow**    | `apache/airflow:2.10.3-python3.10` | `8080`        | DAG orchestration UI                    |

#### Monitoring & Alerting

| Service                 | Image                    | Port   | Purpose                                            |
| ----------------------- | ------------------------ | ------ | -------------------------------------------------- |
| **prometheus**          | `prom/prometheus`        | `9090` | Metrics scraping & storage + alerting rules        |
| **alertmanager**        | `prom/alertmanager`      | `9093` | Alert routing & notification                       |
| **rollback-automation** | `python:3.10-slim`       | `8085` | Auto-rollback receiver (webhook from Alertmanager) |
| **grafana**             | `grafana/grafana:12.4.1` | `3000` | Monitoring dashboards                              |

#### Infrastructure Metrics

| Service            | Image                             | Port   | Purpose                      |
| ------------------ | --------------------------------- | ------ | ---------------------------- |
| **nginx-exporter** | `nginx/nginx-prometheus-exporter` | —      | Nginx metrics for Prometheus |
| **node-exporter**  | `prom/node-exporter`              | —      | Host system metrics          |
| **cadvisor**       | `gcr.io/cadvisor/cadvisor`        | `8081` | Container resource metrics   |

All services communicate over the `tcb-mlops-network` Docker bridge network.

### Key Environment Variables (`.env`)

```bash
# Storage
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123
MINIO_BUCKET=tcb-mlops-artifacts

# UI Credentials
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin123

# Ports
FASTAPI_PORT=8000
MLFLOW_PORT=5000
AIRFLOW_PORT=8080
PROMETHEUS_PORT=9090
ALERTMANAGER_PORT=9093
GRAFANA_PORT=3000

# Drift Detection
DRIFT_WINDOW_SIZE=500
DRIFT_ALERT_THRESHOLD=0.2

# Model Registry
MLFLOW_REGISTERED_MODEL_NAME=tcb-fraud-xgboost
MLFLOW_DEPLOY_STAGE=Production
SYNC_RUNTIME_BUNDLE_FROM_REGISTRY=true

# Auto-Rollback
AUTO_ROLLBACK_GITHUB_TOKEN=
AUTO_ROLLBACK_GITHUB_REPOSITORY=
AUTO_ROLLBACK_COOLDOWN_SECONDS=900
AUTO_ROLLBACK_ALERT_NAMES=CandidateModelBehaviorRegression
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

| Response Field  | Type        | Description                                   |
| --------------- | ----------- | --------------------------------------------- |
| `fraud_score`   | float [0,1] | Raw fraud probability from XGBoost            |
| `is_fraud_pred` | bool        | `true` if `fraud_score ≥ threshold`           |
| `threshold`     | float       | Decision threshold from `evaluation.json`     |
| `risk_level`    | string      | `LOW` / `MEDIUM` / `HIGH` (score-based bands) |

### `POST /predict/batch` — Batch Scoring

```json
{
  "transactions": [
    {
      "transaction_id": "TX_001",
      "timestamp": "...",
      "customer_id": "...",
      "amount": 350000,
      "customer_tier": "PRIORITY"
    },
    {
      "transaction_id": "TX_002",
      "timestamp": "...",
      "customer_id": "...",
      "amount": 5000000,
      "customer_tier": "MASS"
    }
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
    {
      "transaction_id": "TX_001",
      "fraud_score": 0.021,
      "is_fraud_pred": false,
      "risk_level": "LOW"
    },
    {
      "transaction_id": "TX_002",
      "fraud_score": 0.892,
      "is_fraud_pred": true,
      "risk_level": "HIGH"
    }
  ]
}
```

Accepts **up to 1,000 transactions** per request (enforced by Pydantic `max_length=1000`).

### `GET /metrics` — Prometheus Metrics

Exposes runtime metrics for Prometheus scraping (auto-instrumented via `prometheus-fastapi-instrumentator`).

### `GET /monitoring/drift` — Drift Status

Returns the current data drift detection status from the Evidently AI sliding-window monitor.

---

## 🐤 Canary Deployment & Rollout Control

The system supports **canary deployment** — gradually routing traffic from the stable model to a candidate model to validate its behavior in production before full promotion.

### Architecture

```
                    Nginx Load Balancer
                ┌────────────────────────┐
   Requests ──▶ │   split_clients        │
                │   ┌─────────┐          │
                │   │ 90% ────│──▶ fastapi-stable    (current production model)
                │   │ 10% ────│──▶ fastapi-candidate (newly trained model)
                │   └─────────┘          │
                └────────────────────────┘
```

Traffic is split using Nginx's `split_clients` directive. The split ratio is configured in `monitoring/loadbalancer/canary_split.conf` and can be updated dynamically.

### Rollout Control CLI

```bash
# Set candidate traffic to 10%
python -m monitoring.simulator.rollout_ctl set-canary --percent 10

# Check current rollout state
python -m monitoring.simulator.rollout_ctl status

# Promote candidate to stable (becomes the new production model)
python -m monitoring.simulator.rollout_ctl promote

# Rollback: route 100% to stable, stop candidate, clear artifacts
python -m monitoring.simulator.rollout_ctl rollback --registry-action archive-current-production
```

**Registry rollback actions**: `none`, `archive-current-production`, `restore-previous-production`

### GitHub Actions Workflow

The `rollback-canary.yml` workflow allows remote rollout control via GitHub:

| Action       | Description                                                |
| ------------ | ---------------------------------------------------------- |
| `set-canary` | Set candidate traffic percentage (0–100%)                  |
| `promote`    | Promote candidate to stable, reset canary to 0%            |
| `rollback`   | Hard rollback to stable, optionally update MLflow Registry |

---

## 📈 Monitoring & Observability

### Prometheus Metrics (`observability.py`)

The API exposes **15+ custom Prometheus metrics** across 4 categories:

**HTTP Layer:**

| Metric                              | Type      | Description                          |
| ----------------------------------- | --------- | ------------------------------------ |
| `tcb_http_requests_total`           | Counter   | Total requests by method/path/status |
| `tcb_http_request_duration_seconds` | Histogram | Request latency (buckets: 10ms–5s)   |

**Model & Predictions:**

| Metric                            | Type      | Description                                   |
| --------------------------------- | --------- | --------------------------------------------- |
| `tcb_model_loaded`                | Gauge     | 1 if FraudDetector is ready                   |
| `tcb_model_threshold`             | Gauge     | Current decision threshold                    |
| `tcb_model_feature_count`         | Gauge     | Number of features in loaded model            |
| `tcb_prediction_requests_total`   | Counter   | API calls by endpoint                         |
| `tcb_prediction_samples_total`    | Counter   | Transactions scored by endpoint               |
| `tcb_prediction_fraud_total`      | Counter   | Transactions predicted as fraud               |
| `tcb_prediction_risk_level_total` | Counter   | Predictions by risk level `[LOW/MEDIUM/HIGH]` |
| `tcb_prediction_score`            | Histogram | Fraud score distribution (10 buckets)         |
| `tcb_prediction_amount_vnd`       | Histogram | Transaction amount distribution (VND)         |

**Data Drift (Evidently AI):**

| Metric                        | Type  | Description                             |
| ----------------------------- | ----- | --------------------------------------- |
| `tcb_drift_baseline_ready`    | Gauge | 1 if drift monitor has reference window |
| `tcb_drift_reference_samples` | Gauge | Samples in reference window             |
| `tcb_drift_current_samples`   | Gauge | Samples in live window                  |
| `tcb_drift_features_alerting` | Gauge | Features above drift alert threshold    |
| `tcb_drift_feature_score`     | Gauge | Per-feature drift score (labeled)       |

### Drift Detection (`drift_monitor.py`)

`DriftMonitor` uses an **Evidently AI sliding-window** approach:

- **Reference window**: Loaded from `train.parquet` or filled from warm-up traffic
- **Current window**: Sliding window of recent API requests (`DRIFT_WINDOW_SIZE=500`)
- **Alert threshold**: `DRIFT_ALERT_THRESHOLD=0.2` (configurable via `.env`)
- Automatic bootstrap from `train.parquet` on API startup

### Prometheus Scrape Targets

| Job                 | Target                           | Description                      |
| ------------------- | -------------------------------- | -------------------------------- |
| `fastapi-stable`    | `fastapi-stable:8000/metrics`    | Stable model metrics             |
| `fastapi-candidate` | `fastapi-candidate:8000/metrics` | Candidate model metrics          |
| `loadbalancer`      | `nginx-exporter:9113`            | Nginx connection/request metrics |
| `node-exporter`     | `node-exporter:9100`             | Host system metrics              |
| `cadvisor`          | `cadvisor:8080`                  | Docker container metrics         |

### Grafana Dashboards

Pre-provisioned dashboards are available at `http://localhost:3000` (default: `admin/admin123`). Datasource (Prometheus) is auto-configured via provisioning YAML.

---

## 🚨 Alerting & Auto-Rollback

### Prometheus Alert Rules (`alerts.yml`)

6 alert rules across 2 groups:

**API Health Alerts** (`fraud-api-alerts`):

| Alert                    | Condition              | Severity | For |
| ------------------------ | ---------------------- | -------- | --- |
| `FraudApiDown`           | Stable API unreachable | critical | 2m  |
| `FraudApiHighErrorRate`  | 5xx ratio > 5%         | warning  | 5m  |
| `FraudApiHighLatencyP95` | P95 latency > 1s       | warning  | 10m |
| `FraudModelNotLoaded`    | `tcb_model_loaded < 1` | critical | 3m  |
| `FraudDriftRatioHigh`    | Drift ratio > 0.20     | warning  | 10m |

**Canary Alerts** (`fraud-canary-alerts`):

| Alert                              | Condition                            | Severity | For |
| ---------------------------------- | ------------------------------------ | -------- | --- |
| `CandidateModelBehaviorRegression` | Candidate fraud rate < 70% of stable | warning  | 15m |

### Automatic Rollback Flow

```
Prometheus Alert ──▶ Alertmanager ──▶ Rollback Receiver ──▶ GitHub API ──▶ rollback-canary.yml
                                     (HTTP webhook)          (workflow_dispatch)
```

When `CandidateModelBehaviorRegression` fires:

1. **Alertmanager** routes the alert to the rollback receiver (`http://rollback-automation:8085/alertmanager`)
2. **Rollback Receiver** (`alertmanager_rollback_receiver.py`) validates the alert, checks cooldown (15min default), and dispatches a GitHub Actions workflow
3. **GitHub Actions** (`rollback-canary.yml`) SSHs to the VPS and executes `rollout_ctl rollback`
4. Traffic is immediately routed 100% to stable, candidate container is stopped

---

## 🎮 Traffic Simulator

The traffic simulator framework enables realistic load testing, drift injection, and end-to-end validation.

### Configuration (`sim_config.yaml`)

4 built-in drift scenarios:

| Scenario         | Amount Multiplier | High Risk Bias | Description                                 |
| ---------------- | ----------------- | -------------- | ------------------------------------------- |
| `normal`         | 1.0–1.05×         | 0%             | Baseline — no perturbation                  |
| `slight_drift`   | 1.05–1.25×        | 8%             | Mild shift in amount and geography          |
| `moderate_drift` | 1.2–1.8×          | 18%            | Moderate drift in amount, country, behavior |
| `severe_drift`   | 1.8–3.2×          | 35%            | Strong drift to stress alerts               |

### Running Simulations

```bash
# Run baseline simulation (100 requests at 2 RPS)
python -m monitoring.simulator.scenarios.baseline_5rps

# Inject feature drift at 5 RPS
python -m monitoring.simulator.scenarios.feature_drift_5rps

# Full cycle: drift injection → alert trigger → auto-rollback
python -m monitoring.simulator.scenarios.full_cycle_drift_to_rollout
```

### Pre-built Scenarios

| Scenario Script                  | Purpose                                 |
| -------------------------------- | --------------------------------------- |
| `baseline_5rps.py`               | Normal traffic at 5 RPS                 |
| `high_volume_burst.py`           | Burst traffic stress test               |
| `traffic_ramp.py`                | Gradual traffic ramp-up                 |
| `feature_drift_5rps.py`          | Drift injection at 5 RPS                |
| `candidate_soak_test.py`         | Soak test for candidate model stability |
| `auto_retrain_trigger.py`        | End-to-end: retrain + stage candidate   |
| `train_and_stage_candidate.py`   | Train new model + stage as candidate    |
| `full_cycle_drift_to_rollout.py` | Full drift → alert → rollback cycle     |

---

## 🤖 Responsible AI

This project adheres to **Responsible AI** principles as mandated by the DDM501 grading criteria:

### Explainability (SHAP)

- **Global** (`shap_summary.png`): SHAP beeswarm plot revealing the top 20 features most influencing fraud predictions across 2,000 sampled test transactions (`shap.TreeExplainer`).
- **Local** (`shap_waterfall.png`): SHAP waterfall plot explaining _why_ the highest-scoring (most suspicious) transaction was flagged — showing each feature's contribution.

```
models/evaluation/shap_summary.png    # Global feature importance
models/evaluation/shap_waterfall.png  # Local single-instance explanation
```

### Fairness

Model performance is evaluated **per customer segment** to detect disparate impact:

| Segment  | Metrics Tracked                           |
| -------- | ----------------------------------------- |
| PRIVATE  | Precision, Recall, F1, PR-AUC, fraud rate |
| PRIORITY | Precision, Recall, F1, PR-AUC, fraud rate |
| INSPIRE  | Precision, Recall, F1, PR-AUC, fraud rate |
| MASS     | Precision, Recall, F1, PR-AUC, fraud rate |
| **ALL**  | Aggregate across all segments             |

Artifact: `models/evaluation/segment_report.csv`

### Regression Gate (Model Quality Guard)

Before any model is accepted, `compare_baseline()` validates it against the previously saved `metrics.json`:

| Metric   | Max Regression Allowed |
| -------- | ---------------------- |
| PR-AUC   | ≤ 2 percentage points  |
| F1-Score | ≤ 3 percentage points  |
| Recall   | ≤ 2 percentage points  |

Gate result (`PASS` / `FAIL`) is saved in `evaluation.json` and tagged in MLflow.

---

## 🚢 CI/CD Pipeline

### GitHub Actions Workflows

3 workflows covering the full CI/CD + model management lifecycle:

#### 1. CI/CD Pipeline (`ci-cd-pipeline.yml`)

Triggers: **push to `main`** | **pull request to `main`** | **manual dispatch**

| Job        | Description                                                                |
| ---------- | -------------------------------------------------------------------------- |
| **lint**   | `flake8` on `ml_pipeline/src`, `serving_api/app`, `dags`                   |
| **test**   | `pytest` with `--cov-fail-under=80` across all modules                     |
| **docker** | Build & push `tungb12ok/tcb-detect-credit:<sha>` to Docker Hub (push only) |
| **deploy** | SSH to GCP VPS → run `scripts/deploy_vps.sh` (main branch only)            |

Concurrency control: cancels in-progress runs for the same branch/PR.

#### 2. Promote MLflow Model (`promote-model.yml`)

Triggers: **manual dispatch only**

Promotes a model version in the MLflow Registry to a target stage (Staging/Production). Accepts either a specific version number or a training run ID.

#### 3. Rollback Canary Traffic (`rollback-canary.yml`)

Triggers: **manual dispatch** | **auto-triggered by rollback receiver**

| Action       | Description                                         |
| ------------ | --------------------------------------------------- |
| `rollback`   | Hard rollback: route 100% to stable, stop candidate |
| `set-canary` | Set candidate traffic percentage                    |
| `promote`    | Promote candidate to stable                         |

#### Deploy Script (`scripts/deploy_vps.sh`)

The VPS deployment script handles:

- Git fetch & checkout of the target ref
- Docker Hub login and image pull
- Docker Compose service orchestration
- Runtime bundle sync from MLflow Registry
- Health checks for all critical services (FastAPI, MLflow, Airflow, Grafana)

---

## ⏰ Airflow Orchestration

DAG: `fraud_detection_training_pipeline`
Schedule: **`0 2 * * *`** (daily at 02:00 UTC)
Max active runs: 1 (no concurrent retraining)

```python
check_model_quality >> [preprocess, skip_retraining]
preprocess >> train >> evaluate >> stage_candidate >> verify_candidate
```

| Task                  | Type                 | Description                                              |
| --------------------- | -------------------- | -------------------------------------------------------- |
| `check_model_quality` | BranchPythonOperator | Query MLflow `eval_f1` vs threshold (0.80)               |
| `preprocess`          | BashOperator         | Run `preprocess.py`                                      |
| `train`               | BashOperator         | Run `train.py` + MLflow Registry                         |
| `evaluate`            | BashOperator         | Run `evaluate.py` + SHAP                                 |
| `stage_candidate`     | PythonOperator       | Publish runtime bundle + sync to candidate slot          |
| `verify_candidate`    | PythonOperator       | Poll candidate FastAPI `/health` to confirm model loaded |

The DAG integrates with the **canary deployment** system: after evaluation passes, the new model is staged as a candidate, and traffic can be gradually shifted to validate it in production.

---

## ✅ Testing

```bash
# Run all tests with coverage gate (≥80%)
make test

# Or run individually:
pytest ml_pipeline/tests/test_preprocess.py -v    # Preprocessing tests
pytest ml_pipeline/tests/test_model.py -v          # Training + evaluation + inference
pytest ml_pipeline/tests/test_coverage.py -v       # Coverage enforcement
pytest serving_api/tests/test_api.py -v             # API integration tests
pytest serving_api/tests/test_loader.py -v          # Model loader tests

# Full coverage report
pytest ml_pipeline/tests/ serving_api/tests/ \
  --cov=ml_pipeline.src --cov=serving_api.app \
  --cov-report=term-missing --cov-fail-under=80
```

### Coverage Requirements

| Component         | Target | Enforced By                                    |
| ----------------- | ------ | ---------------------------------------------- |
| `ml_pipeline.src` | > 80%  | `pytest --cov-fail-under=80`                   |
| `serving_api.app` | > 80%  | `pytest --cov-fail-under=80`                   |
| GitHub Actions CI | > 80%  | `--cov-fail-under=80` gate (blocks deployment) |

---

## 🔨 Makefile Shortcuts

```bash
make help              # Show all available targets
make lint              # Run flake8 linter
make test              # Run full test suite with coverage ≥80%
make test-quick        # Run tests without coverage (faster)
make check             # Lint + test (same as CI)

make preprocess        # Run preprocessing pipeline
make train             # Train XGBoost model
make evaluate          # Evaluate: SHAP + segment fairness
make pipeline          # Full pipeline: preprocess → train → evaluate
make inference-smoke   # Run inference smoke test

make dvc-repro         # Reproduce DVC pipeline
make dvc-dag           # Show DVC pipeline DAG

make serve             # Start FastAPI dev server locally
make up                # Start all Docker services
make down              # Stop all Docker services
make restart           # Restart all Docker services
make logs              # Tail logs from all containers
make logs-api          # Tail logs from FastAPI containers
make ps                # Show running container status
make docker-build      # Build serving API Docker image
make clean             # Remove Python caches
make env-check         # Verify .env file exists
```

---

## 🛠 Tech Stack

| Layer                   | Technology                 | Version                                  |
| ----------------------- | -------------------------- | ---------------------------------------- |
| **Language**            | Python                     | 3.10.9                                   |
| **ML Model**            | XGBoost                    | 2.1.1                                    |
| **Data Processing**     | pandas, numpy, pyarrow     | 2.2.2, 1.26.4, 15.0.2                    |
| **Explainability**      | SHAP                       | 0.46.0                                   |
| **Experiment Tracking** | MLflow (+ Model Registry)  | 2.14.3                                   |
| **Data Versioning**     | DVC                        | 3.51.2                                   |
| **API Framework**       | FastAPI                    | 0.112.0                                  |
| **API Server**          | Uvicorn                    | 0.30.6                                   |
| **Data Validation**     | Pydantic v2                | 2.8.2                                    |
| **Containerization**    | Docker, Docker Compose     | multi-stage build                        |
| **Load Balancer**       | Nginx (canary routing)     | 1.27-alpine                              |
| **Artifact Storage**    | MinIO (S3-compatible)      | latest                                   |
| **Data Monitoring**     | Evidently AI               | 0.4.39                                   |
| **Metrics & Alerting**  | Prometheus, Alertmanager   | —                                        |
| **Dashboarding**        | Grafana                    | 12.4.1                                   |
| **Orchestration**       | Apache Airflow             | 2.10.3-python3.10                        |
| **CI/CD**               | GitHub Actions             | 3 workflows (CI/CD + Promote + Rollback) |
| **Deployment**          | GCP VPS (SSH-based)        | appleboy/ssh-action                      |
| **Testing**             | pytest, pytest-cov         | 8.2.2, 5.0.0                             |
| **HTTP Testing**        | httpx (FastAPI TestClient) | 0.27.0                                   |
| **Linting**             | flake8                     | —                                        |
| **Traffic Simulation**  | Custom simulator + httpx   | —                                        |

---

## 👥 Authors

- **Nguyen Trung Kien**
- **Luu Duc Tung**
- **Le Trung Thanh**

DDM501 — University Final Project

---

## 📄 License

This project is developed for academic purposes (DDM501 — University Final Project).
