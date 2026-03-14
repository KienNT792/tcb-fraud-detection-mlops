---
trigger: glob
globs: Dockerfile, docker-compose.yml, .github/**/*.yml, dags/*.py
---

# MLOps, Infrastructure, and CI/CD Rules
When writing configuration files, CI/CD pipelines, or Airflow DAGs, follow these standards:

# Tech Stack
- Orchestration: Apache Airflow (for scheduling data ingestion and training)
- Containerization: Docker, Docker Compose
- CI/CD: GitHub Actions

# Implementation Directives
1. Dockerfile: Always use multi-stage builds to minimize image size. Use `python:3.10-slim` as the base image. Do not run containers as the root user.
2. Docker Compose: Ensure all services (FastAPI, MLflow, Prometheus, Grafana) are on a shared custom Docker network. Map ports explicitly and set environment variables securely.
3. GitHub Actions: The `.github/workflows/ci-cd-pipeline.yml` must include stages for:
   - Checking out code.
   - Setting up Python.
   - Running `flake8` for linting.
   - Running `pytest` and asserting that code coverage is >80%.
   - Building the Docker image (only if tests pass).
4. Airflow: Keep DAGs clean. Use `PythonOperator` or `BashOperator` to trigger the scripts in `ml_pipeline/src/`.