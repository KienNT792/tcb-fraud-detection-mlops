from __future__ import annotations

import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from html import escape
from typing import Any, AsyncGenerator

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .model_loader import (
    API_VERSION,
    get_detector,
    get_rollout_manager,
    load_model,
    unload_model,
)
from .schemas import (
    BatchPredictionItem,
    BatchPredictionResponse,
    BatchTransactionRequest,
    ErrorResponse,
    HealthResponse,
    ProbeResponse,
    PredictionResponse,
    RolloutConfigRequest,
    RolloutStatusResponse,
    TransactionRequest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)
API_KEY = os.getenv("API_KEY", "")

REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Total API requests",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "fraud_api_request_latency_seconds",
    "Latency for API requests",
    ["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0),
)
PREDICTION_COUNTER = Counter(
    "fraud_predictions_total",
    "Total prediction responses",
    ["risk_level", "predicted"],
)
PREDICTION_SCORE = Histogram(
    "fraud_prediction_score",
    "Distribution of fraud scores",
    buckets=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
)
ROUTING_COUNTER = Counter(
    "fraud_prediction_routes_total",
    "Prediction traffic routed to stable or candidate model",
    ["lane", "model_version"],
)


def _record_prediction_metrics(result: dict[str, Any], lane: str) -> None:
    PREDICTION_COUNTER.labels(
        risk_level=result["risk_level"],
        predicted=str(bool(result["is_fraud_pred"])).lower(),
    ).inc()
    PREDICTION_SCORE.observe(float(result["fraud_score"]))
    ROUTING_COUNTER.labels(lane=lane, model_version=str(result["model_version"])).inc()


def _render_rollout_ui(status_payload: dict[str, Any]) -> str:
    default_candidate_models = escape(
        status_payload.get("candidate", {}).get("models_dir")
        or status_payload.get("stable", {}).get("models_dir")
        or ""
    )
    default_candidate_processed = escape(
        status_payload.get("candidate", {}).get("processed_dir")
        or status_payload.get("stable", {}).get("processed_dir")
        or ""
    )
    rollout_steps = ",".join(str(step) for step in status_payload.get("rollout_steps", []))
    status_json = escape(json.dumps(status_payload, indent=2))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Fraud Model Rollout</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f4ef;
      --panel: #fffdf8;
      --text: #1d2a31;
      --muted: #5c6a72;
      --accent: #0f766e;
      --border: #d7d2c8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #f6f4ef 0%, #ebe6db 100%);
      color: var(--text);
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1 {{
      font-size: 34px;
      margin: 0 0 12px;
    }}
    p {{
      margin: 0 0 20px;
      color: var(--muted);
    }}
    .grid {{
      display: grid;
      gap: 20px;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 12px 32px rgba(29, 42, 49, 0.08);
    }}
    label {{
      display: block;
      font-size: 13px;
      font-weight: 600;
      margin: 12px 0 6px;
    }}
    input, textarea {{
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      font: inherit;
      background: #fff;
    }}
    textarea {{
      min-height: 280px;
      resize: vertical;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 13px;
    }}
    .row {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    }}
    .checks {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-top: 12px;
    }}
    .checks label {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 0;
      font-weight: 500;
    }}
    .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 10px 16px;
      font: inherit;
      font-weight: 600;
      cursor: pointer;
      background: var(--accent);
      color: #fff;
    }}
    button.secondary {{
      background: #2d3748;
    }}
    button.warn {{
      background: #b45309;
    }}
    button.danger {{
      background: #b42318;
    }}
    code {{
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      background: rgba(15, 118, 110, 0.08);
      padding: 2px 6px;
      border-radius: 6px;
    }}
    .status-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin: 18px 0 24px;
    }}
    .pill {{
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(15, 118, 110, 0.1);
      color: var(--accent);
      font-weight: 600;
    }}
    #message {{
      margin-top: 12px;
      min-height: 20px;
      color: var(--muted);
      white-space: pre-wrap;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Canary Rollout Control</h1>
    <p>
      Kubernetes <code>Service</code> và <code>Ingress</code> đã chia request giữa nhiều replica.
      Trang này điều khiển traffic giữa model stable và candidate trong từng replica theo các bước 10% đến 100%.
    </p>
    <div class="status-strip">
      <div class="pill">status: <span id="status-label">{escape(str(status_payload.get("status", "unknown")))}</span></div>
      <div class="pill">candidate traffic: <span id="candidate-traffic">{int(status_payload.get("traffic_percent_candidate", 0))}</span>%</div>
      <div class="pill">next step: <span id="next-step">{escape(str(status_payload.get("next_step_at") or "manual"))}</span></div>
    </div>
    <div class="grid">
      <section>
        <label for="api-key">API key</label>
        <input id="api-key" type="password" placeholder="Optional if API_KEY is disabled">
        <label for="candidate-models-dir">Candidate models dir</label>
        <input id="candidate-models-dir" type="text" value="{default_candidate_models}">
        <label for="candidate-processed-dir">Candidate processed dir</label>
        <input id="candidate-processed-dir" type="text" value="{default_candidate_processed}">
        <div class="row">
          <div>
            <label for="rollout-steps">Rollout steps</label>
            <input id="rollout-steps" type="text" value="{escape(rollout_steps)}">
          </div>
          <div>
            <label for="step-interval">Step interval minutes</label>
            <input id="step-interval" type="number" min="1" value="{int(status_payload.get("step_interval_minutes", 30))}">
          </div>
        </div>
        <div class="checks">
          <label><input id="enabled" type="checkbox" {"checked" if status_payload.get("enabled") else ""}> Enable rollout</label>
          <label><input id="auto-advance" type="checkbox" {"checked" if status_payload.get("auto_advance") else ""}> Auto-advance by schedule</label>
          <label><input id="auto-promote" type="checkbox" {"checked" if status_payload.get("auto_promote_when_complete") else ""}> Auto-promote at final step</label>
        </div>
        <div class="actions">
          <button onclick="saveConfig()">Save config</button>
          <button class="secondary" onclick="advanceRollout()">Advance step</button>
          <button class="warn" onclick="pauseRollout()">Pause schedule</button>
          <button class="secondary" onclick="promoteRollout()">Promote candidate</button>
          <button class="danger" onclick="rollbackRollout()">Rollback to stable</button>
          <button class="secondary" onclick="refreshStatus()">Refresh</button>
        </div>
        <div id="message"></div>
      </section>
      <section>
        <label for="status-json">Rollout status</label>
        <textarea id="status-json" readonly>{status_json}</textarea>
      </section>
    </div>
  </main>
  <script>
    const statusBox = document.getElementById("status-json");
    const messageBox = document.getElementById("message");

    function headers() {{
      const apiKey = document.getElementById("api-key").value.trim();
      const base = {{"Content-Type": "application/json"}};
      if (apiKey) {{
        base["x-api-key"] = apiKey;
      }}
      return base;
    }}

    function payloadFromForm() {{
      const rawSteps = document.getElementById("rollout-steps").value.trim();
      return {{
        enabled: document.getElementById("enabled").checked,
        auto_advance: document.getElementById("auto-advance").checked,
        auto_promote_when_complete: document.getElementById("auto-promote").checked,
        candidate_models_dir: document.getElementById("candidate-models-dir").value.trim(),
        candidate_processed_dir: document.getElementById("candidate-processed-dir").value.trim(),
        rollout_steps: rawSteps ? rawSteps.split(",").map((item) => Number(item.trim())).filter((item) => !Number.isNaN(item) && item > 0) : undefined,
        step_interval_minutes: Number(document.getElementById("step-interval").value || 0),
      }};
    }}

    function syncStatus(status) {{
      document.getElementById("status-label").textContent = status.status;
      document.getElementById("candidate-traffic").textContent = String(status.traffic_percent_candidate);
      document.getElementById("next-step").textContent = status.next_step_at || "manual";
      document.getElementById("enabled").checked = Boolean(status.enabled);
      document.getElementById("auto-advance").checked = Boolean(status.auto_advance);
      document.getElementById("auto-promote").checked = Boolean(status.auto_promote_when_complete);
      document.getElementById("rollout-steps").value = status.rollout_steps.join(",");
      document.getElementById("step-interval").value = String(status.step_interval_minutes);
      if (status.candidate?.models_dir) {{
        document.getElementById("candidate-models-dir").value = status.candidate.models_dir;
      }}
      if (status.candidate?.processed_dir) {{
        document.getElementById("candidate-processed-dir").value = status.candidate.processed_dir;
      }}
      statusBox.value = JSON.stringify(status, null, 2);
    }}

    async function callApi(url, method, payload) {{
      messageBox.textContent = "Working...";
      const response = await fetch(url, {{
        method,
        headers: headers(),
        body: payload ? JSON.stringify(payload) : undefined,
      }});
      const data = await response.json();
      if (!response.ok) {{
        const detail = data.detail || data.error || JSON.stringify(data);
        throw new Error(detail);
      }}
      syncStatus(data);
      messageBox.textContent = "Updated at " + new Date().toISOString();
    }}

    async function saveConfig() {{
      const payload = payloadFromForm();
      await callApi("/admin/rollout/config", "POST", payload);
    }}

    async function advanceRollout() {{
      await callApi("/admin/rollout/advance", "POST");
    }}

    async function pauseRollout() {{
      await callApi("/admin/rollout/pause", "POST");
    }}

    async function promoteRollout() {{
      await callApi("/admin/rollout/promote", "POST");
    }}

    async function rollbackRollout() {{
      await callApi("/admin/rollout/rollback", "POST");
    }}

    async function refreshStatus() {{
      await callApi("/admin/rollout", "GET");
    }}

    refreshStatus().catch((error) => {{
      messageBox.textContent = error.message;
    }});
  </script>
</body>
</html>"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model artifacts at startup, release at shutdown."""
    logger.info("=" * 50)
    logger.info("TCB FRAUD DETECTION API — STARTING UP")
    logger.info("=" * 50)
    try:
        load_model()
        logger.info("Model loaded successfully. API ready.")
    except FileNotFoundError as exc:
        logger.error("STARTUP FAILED — artifact missing: %s", exc)
        raise

    yield  # API is live and serving requests

    logger.info("Shutting down — releasing model.")
    unload_model()
    logger.info("API shutdown complete.")


app = FastAPI(
    title="TCB Fraud Detection API",
    description=(
        "Real-time credit card fraud scoring using XGBoost. "
        "Accepts raw transaction data and returns a fraud probability score, "
        "binary prediction, and risk level classification."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins in development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time(request: Request, call_next: Any) -> Any:
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_seconds = time.perf_counter() - t0
    elapsed_ms = elapsed_seconds * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    REQUEST_COUNT.labels(
        method=request.method,
        path=request.url.path,
        status_code=str(response.status_code),
    ).inc()
    REQUEST_LATENCY.labels(
        method=request.method,
        path=request.url.path,
    ).observe(elapsed_seconds)
    logger.info(
        "%s %s → %d (%.1fms)",
        request.method, request.url.path,
        response.status_code, elapsed_ms,
    )
    return response


def verify_api_key(request: Request) -> None:
    if not API_KEY:
        return

    provided = request.headers.get("x-api-key", "")
    if not secrets.compare_digest(provided, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Validation error",
            detail=str(exc),
            status_code=422,
        ).model_dump(),
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            error="Model not available",
            detail=str(exc),
            status_code=503,
        ).model_dump(),
    )


@app.get(
    "/",
    summary="API info",
    tags=["Info"],
)
async def root() -> dict[str, str]:
    """Return basic API metadata."""
    return {
        "name":        "TCB Fraud Detection API",
        "version":     API_VERSION,
        "docs":        "/docs",
        "health":      "/health",
        "ready":       "/ready",
        "live":        "/live",
        "metrics":     "/metrics",
        "predict":     "POST /predict",
        "batch":       "POST /predict/batch",
        "rollout":     "/admin/rollout",
        "rollout_ui":  "/admin/rollout/ui",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Model health check",
    tags=["Health"],
)
async def health(detector=Depends(get_detector)) -> HealthResponse:
    """Return model health status and metadata.

    Use this endpoint for liveness/readiness probes in Kubernetes or
    any container orchestration system.
    """
    info = detector.health_check()
    return HealthResponse(
        status         = info["status"],
        model_type     = info["model_type"],
        feature_count  = info["feature_count"],
        threshold      = info["threshold"],
        best_iteration = info["best_iteration"],
        loaded_at      = info["loaded_at"],
        api_version    = API_VERSION,
        model_version  = info["model_version"],
    )


@app.get(
    "/live",
    response_model=ProbeResponse,
    summary="Liveness probe",
    tags=["Health"],
)
async def live() -> ProbeResponse:
    return ProbeResponse(status="alive", api_version=API_VERSION)


@app.get(
    "/ready",
    response_model=ProbeResponse,
    summary="Readiness probe",
    tags=["Health"],
)
async def ready(detector=Depends(get_detector)) -> ProbeResponse:
    detector.health_check()
    return ProbeResponse(status="ready", api_version=API_VERSION)


@app.get(
    "/metrics",
    summary="Prometheus metrics",
    tags=["Monitoring"],
)
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Score a single transaction",
    tags=["Prediction"],
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Prediction failed"},
    },
)
async def predict(
    request: TransactionRequest,
    rollout_manager=Depends(get_rollout_manager),
    _auth: None = Depends(verify_api_key),
) -> PredictionResponse:
    """Score a single raw transaction and return a fraud probability.

    The request body must include all required fields (transaction_id,
    timestamp, customer_id, amount, customer_tier). Optional behavioural
    fields default to 0 / N/A when absent.

    Returns a fraud_score in [0, 1], a binary is_fraud_pred flag based on
    the optimal threshold, and a risk_level classification.
    """
    try:
        lane, result = rollout_manager.route_single(request.model_dump())
        result["served_by"] = lane
        result["rollout_candidate_percent"] = int(
            rollout_manager.get_rollout_status()["traffic_percent_candidate"]
        )
    except Exception as exc:
        logger.exception("predict() failed for tx_id=%s", request.transaction_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc

    _record_prediction_metrics(result, lane)
    return PredictionResponse(**result)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Score a batch of transactions",
    tags=["Prediction"],
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Batch prediction failed"},
    },
)
async def predict_batch(
    request: BatchTransactionRequest,
    rollout_manager=Depends(get_rollout_manager),
    _auth: None = Depends(verify_api_key),
) -> BatchPredictionResponse:
    """Score up to 1000 transactions in a single request.

    Accepts a JSON body with a ``transactions`` list. Each item follows the
    same schema as the single ``/predict`` endpoint.

    Returns aggregated stats (total, fraud_detected, fraud_rate) plus a
    per-transaction predictions list.
    """
    try:
        raw_df = pd.DataFrame(
            [tx.model_dump() for tx in request.transactions]
        )
        routing, results_df = rollout_manager.route_batch(raw_df)
    except Exception as exc:
        logger.exception("predict_batch() failed — %d transactions", len(request.transactions))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {exc}",
        ) from exc

    predictions = [
        BatchPredictionItem(
            transaction_id = str(row["transaction_id"]),
            fraud_score    = float(row["fraud_score"]),
            is_fraud_pred  = bool(row["is_fraud_pred"]),
            threshold      = float(row["threshold"]),
            risk_level     = str(row["risk_level"]),
            model_version  = str(row["model_version"]),
            prediction_timestamp = str(row["prediction_timestamp"]),
            served_by      = str(row["served_by"]),
        )
        for _, row in results_df.iterrows()
    ]

    for _, row in results_df.iterrows():
        _record_prediction_metrics(
            {
                "risk_level": str(row["risk_level"]),
                "is_fraud_pred": bool(row["is_fraud_pred"]),
                "fraud_score": float(row["fraud_score"]),
                "model_version": str(row["model_version"]),
            },
            str(row["served_by"]),
        )

    n_fraud = int(results_df["is_fraud_pred"].sum())
    total   = len(predictions)
    rollout_status = rollout_manager.get_rollout_status()
    unique_versions = {str(value) for value in results_df["model_version"].tolist()}
    unique_thresholds = {float(value) for value in results_df["threshold"].tolist()}

    return BatchPredictionResponse(
        total          = total,
        fraud_detected = n_fraud,
        fraud_rate     = round(n_fraud / total, 6) if total > 0 else 0.0,
        threshold      = next(iter(unique_thresholds)) if len(unique_thresholds) == 1 else None,
        model_version  = next(iter(unique_versions)) if len(unique_versions) == 1 else "mixed",
        prediction_timestamp = (
            str(results_df["prediction_timestamp"].iloc[0]) if total > 0 else ""
        ),
        rollout_candidate_percent = int(rollout_status["traffic_percent_candidate"]),
        traffic_distribution = routing,
        predictions    = predictions,
    )


@app.get(
    "/admin/rollout",
    response_model=RolloutStatusResponse,
    summary="Current canary rollout status",
    tags=["Rollout"],
)
async def rollout_status(
    rollout_manager=Depends(get_rollout_manager),
    _auth: None = Depends(verify_api_key),
) -> RolloutStatusResponse:
    return RolloutStatusResponse(**rollout_manager.get_rollout_status())


@app.post(
    "/admin/rollout/config",
    response_model=RolloutStatusResponse,
    summary="Create or update canary rollout config",
    tags=["Rollout"],
)
async def rollout_config(
    request: RolloutConfigRequest,
    rollout_manager=Depends(get_rollout_manager),
    _auth: None = Depends(verify_api_key),
) -> RolloutStatusResponse:
    payload = request.model_dump(exclude_none=True)
    return RolloutStatusResponse(**rollout_manager.update_rollout(payload))


@app.post(
    "/admin/rollout/advance",
    response_model=RolloutStatusResponse,
    summary="Advance rollout to the next traffic step",
    tags=["Rollout"],
)
async def rollout_advance(
    rollout_manager=Depends(get_rollout_manager),
    _auth: None = Depends(verify_api_key),
) -> RolloutStatusResponse:
    return RolloutStatusResponse(**rollout_manager.advance_rollout())


@app.post(
    "/admin/rollout/pause",
    response_model=RolloutStatusResponse,
    summary="Pause scheduled rollout progression while keeping current traffic split",
    tags=["Rollout"],
)
async def rollout_pause(
    rollout_manager=Depends(get_rollout_manager),
    _auth: None = Depends(verify_api_key),
) -> RolloutStatusResponse:
    return RolloutStatusResponse(**rollout_manager.pause_rollout())


@app.post(
    "/admin/rollout/promote",
    response_model=RolloutStatusResponse,
    summary="Promote candidate model to stable",
    tags=["Rollout"],
)
async def rollout_promote(
    rollout_manager=Depends(get_rollout_manager),
    _auth: None = Depends(verify_api_key),
) -> RolloutStatusResponse:
    return RolloutStatusResponse(**rollout_manager.promote_candidate())


@app.post(
    "/admin/rollout/rollback",
    response_model=RolloutStatusResponse,
    summary="Rollback all traffic to the stable model",
    tags=["Rollout"],
)
async def rollout_rollback(
    rollout_manager=Depends(get_rollout_manager),
    _auth: None = Depends(verify_api_key),
) -> RolloutStatusResponse:
    return RolloutStatusResponse(**rollout_manager.rollback_rollout())


@app.get(
    "/admin/rollout/ui",
    response_class=HTMLResponse,
    summary="Simple UI for canary rollout control",
    tags=["Rollout"],
)
async def rollout_ui(rollout_manager=Depends(get_rollout_manager)) -> HTMLResponse:
    return HTMLResponse(_render_rollout_ui(rollout_manager.get_rollout_status()))
