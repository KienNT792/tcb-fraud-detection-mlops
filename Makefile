PYTHON ?= python
DOCKER_COMPOSE ?= docker compose
API_URL ?= http://localhost:8000

.PHONY: bootstrap-demo lint test run-api mlflow-server airflow compose-up compose-down kafka-producer kafka-consumer drift-monitor smoke-test

bootstrap-demo:
	$(PYTHON) scripts/bootstrap_demo_artifacts.py --if-missing

lint:
	ruff check .
	$(PYTHON) -m compileall ml_pipeline serving_api dags monitoring streaming scripts
	$(PYTHON) scripts/validate_manifests.py

test:
	pytest ml_pipeline/tests serving_api/tests -q

run-api:
	BOOTSTRAP_DEMO_ARTIFACTS=true $(PYTHON) -m uvicorn serving_api.app.main:app --host 0.0.0.0 --port 8000

mlflow-server:
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./artifacts/mlflow

airflow:
	airflow standalone

compose-up:
	$(DOCKER_COMPOSE) up --build -d

compose-down:
	$(DOCKER_COMPOSE) down --remove-orphans

kafka-producer:
	$(PYTHON) -m streaming.producer --count 10

kafka-consumer:
	$(PYTHON) -m streaming.scoring_consumer --once

drift-monitor:
	$(PYTHON) monitoring/evidently_ai/drift_monitor.py

smoke-test:
	$(PYTHON) scripts/smoke_test_api.py --url $(API_URL)
