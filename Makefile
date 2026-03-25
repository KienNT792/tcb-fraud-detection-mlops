# ============================================================================
# TCB Fraud Detection MLOps — Developer Makefile
# ============================================================================
# Usage:
#   make help          Show all available targets
#   make lint          Run flake8 linter
#   make test          Run pytest with coverage gate (≥80%)
#   make pipeline      Run full ML pipeline: preprocess → train → evaluate
#   make up            Start all Docker services
#   make down          Stop all Docker services
# ============================================================================

.DEFAULT_GOAL := help
SHELL := /bin/bash

# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------
PYTHON     ?= python
PYTEST     ?= pytest
FLAKE8     ?= flake8

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
ML_SRC     := ml_pipeline/src
ML_TESTS   := ml_pipeline/tests
API_APP    := serving_api/app
API_TESTS  := serving_api/tests
DAGS_DIR   := dags
RAW_DATA   := data/raw/tcb_credit_fraud_dataset.csv
PROCESSED  := data/processed
MODELS     := models

# ============================================================================
# Quality
# ============================================================================

.PHONY: lint
lint: ## Run flake8 linter on all source directories
	$(FLAKE8) $(ML_SRC) $(API_APP) $(DAGS_DIR)

.PHONY: test
test: ## Run full test suite with coverage ≥80%
	$(PYTEST) $(ML_TESTS) $(API_TESTS) \
		--cov=ml_pipeline.src \
		--cov=serving_api.app \
		--cov-report=term-missing \
		--cov-fail-under=80

.PHONY: test-quick
test-quick: ## Run tests without coverage (faster)
	$(PYTEST) $(ML_TESTS) $(API_TESTS) -q

.PHONY: check
check: lint test ## Lint + test (same as CI)

# ============================================================================
# ML Pipeline
# ============================================================================

.PHONY: preprocess
preprocess: ## Run preprocessing pipeline
	$(PYTHON) -m ml_pipeline.src.preprocess

.PHONY: train
train: ## Train XGBoost model
	$(PYTHON) -m ml_pipeline.src.train

.PHONY: evaluate
evaluate: ## Evaluate model with SHAP + segment fairness
	$(PYTHON) -m ml_pipeline.src.evaluate

.PHONY: pipeline
pipeline: preprocess train evaluate ## Run full ML pipeline: preprocess → train → evaluate

.PHONY: inference-smoke
inference-smoke: ## Run inference smoke test on test.parquet
	$(PYTHON) -m ml_pipeline.src.inference

# ============================================================================
# DVC
# ============================================================================

.PHONY: dvc-repro
dvc-repro: ## Reproduce DVC pipeline (preprocess → train → evaluate)
	dvc repro

.PHONY: dvc-dag
dvc-dag: ## Show DVC pipeline DAG
	dvc dag

# ============================================================================
# Serving API
# ============================================================================

.PHONY: serve
serve: ## Start FastAPI dev server locally (port 8000)
	uvicorn serving_api.app.main:app --host 0.0.0.0 --port 8000 --reload

# ============================================================================
# Docker
# ============================================================================

.PHONY: up
up: ## Start all Docker services
	docker compose up -d --build

.PHONY: down
down: ## Stop all Docker services
	docker compose down

.PHONY: restart
restart: down up ## Restart all Docker services

.PHONY: logs
logs: ## Tail logs from all containers
	docker compose logs -f --tail=100

.PHONY: logs-api
logs-api: ## Tail logs from FastAPI containers only
	docker compose logs -f --tail=100 fastapi-stable fastapi-candidate

.PHONY: ps
ps: ## Show running container status
	docker compose ps

.PHONY: docker-build
docker-build: ## Build serving API Docker image
	docker compose build fastapi-stable

.PHONY: clean-volumes
clean-volumes: ## Remove all Docker volumes (⚠️ destructive)
	docker compose down -v

# ============================================================================
# Utility
# ============================================================================

.PHONY: clean
clean: ## Remove Python caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov

.PHONY: env-check
env-check: ## Verify .env file exists
	@if [ ! -f .env ]; then \
		echo "⚠️  .env not found. Copy from .env.example:"; \
		echo "   cp .env.example .env"; \
		exit 1; \
	fi
	@echo "✓ .env exists"

.PHONY: help
help: ## Show this help message
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║     TCB Fraud Detection MLOps — Available Targets       ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
