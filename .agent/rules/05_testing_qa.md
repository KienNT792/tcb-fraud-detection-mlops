---
trigger: glob
globs: **/tests/*.py
---

# Testing and QA Rules
When writing tests in any `tests/` directory, follow these rules to ensure we hit the strictly required >80% coverage:

# Tech Stack
- Framework: `pytest`, `pytest-cov`
- API Testing: `fastapi.testclient.TestClient`
- Mocking: `unittest.mock`

# Implementation Directives
1. Unit Tests: Write atomic tests for individual functions (e.g., testing the preprocessing logic or Pydantic validation).
2. Integration Tests: Use FastAPI's `TestClient` to test the `/predict` endpoint end-to-end.
3. Data Tests: Write assertions to ensure that the synthetic data loading process doesn't contain unexpected NaNs in critical columns (like `amount` or `is_fraud`).
4. Mocking External Services: NEVER make actual calls to MLflow tracking servers or Prometheus during unit tests. Heavily use `@patch` to mock `mlflow.start_run()` and `mlflow.pyfunc.load_model()`.