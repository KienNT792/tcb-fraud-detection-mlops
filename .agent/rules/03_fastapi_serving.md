---
trigger: glob
globs: serving_api/**/*.py
---

# API Serving Rules
When writing code inside the `serving_api/` directory, adhere strictly to these rules:

# Tech Stack
- Web Framework: `fastapi`, `uvicorn`
- Data Validation: `pydantic`
- Monitoring: `prometheus-client`
- Model Loading: `mlflow.pyfunc`

# Implementation Directives
1. Dynamic Model Loading: The API MUST NOT train the model. It must load the latest model dynamically from the MLflow Model Registry (e.g., fetching the model with the "Production" alias/tag).
2. Input Validation: Use Pydantic `BaseModel` to strictly validate incoming JSON payloads. The schema must match the `vietnam_banking_fraud_dataset_v6` structure (e.g., amount, merchant_category, customer_tier).
3. Observability: Expose a `/metrics` endpoint using `prometheus-client`. You must track:
   - `api_request_duration_seconds` (Histogram for latency)
   - `api_requests_total` (Counter for traffic)
   - `fraud_predictions_total` (Counter for how many transactions are flagged as fraud)
4. Async: Use `async def` for API endpoints to ensure high concurrency.