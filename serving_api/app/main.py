from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from prometheus_fastapi_instrumentator import Instrumentator
except ImportError:  # pragma: no cover - local dev without optional dependency
    Instrumentator = None

from .model_loader import (
    API_VERSION,
    get_detector,
    load_model,
    unload_model,
)
from .observability import (
    bootstrap_observability,
    record_http_observation,
    record_prediction_observation,
    shutdown_observability,
)
from .schemas import (
    BatchPredictionItem,
    BatchPredictionResponse,
    BatchTransactionRequest,
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    TransactionRequest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model artifacts at startup, release at shutdown."""
    logger.info("=" * 50)
    logger.info("TCB FRAUD DETECTION API — STARTING UP")
    logger.info("=" * 50)
    try:
        detector = load_model()
        bootstrap_observability(detector)
        logger.info("Model loaded successfully. API ready.")
    except FileNotFoundError as exc:
        logger.error("STARTUP FAILED — artifact missing: %s", exc)
        raise

    yield  # API is live and serving requests

    logger.info("Shutting down — releasing model.")
    shutdown_observability()
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

if Instrumentator is not None:
    Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        excluded_handlers=["/health"],
    ).instrument(app).expose(
        app,
        endpoint="/metrics",
        include_in_schema=False,
        tags=["Monitoring"],
    )
else:
    logger.warning("Prometheus instrumentation disabled: dependency not installed.")

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
    elapsed_ms = (time.perf_counter() - t0) * 1000
    record_http_observation(
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_seconds=elapsed_ms / 1000,
    )
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    logger.info(
        "%s %s → %d (%.1fms)",
        request.method, request.url.path,
        response.status_code, elapsed_ms,
    )
    return response


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
        "predict":     "POST /predict",
        "batch":       "POST /predict/batch",
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
    )


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
    detector=Depends(get_detector),
) -> PredictionResponse:
    """Score a single raw transaction and return a fraud probability.

    The request body must include all required fields (transaction_id,
    timestamp, customer_id, amount, customer_tier). Optional behavioural
    fields default to 0 / N/A when absent.

    Returns a fraud_score in [0, 1], a binary is_fraud_pred flag based on
    the optimal threshold, and a risk_level classification.
    """
    payload = request.model_dump()
    try:
        result = detector.predict_single(payload)
    except Exception as exc:
        logger.exception("predict() failed for tx_id=%s", request.transaction_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc

    raw_df = pd.DataFrame([payload])
    predictions_df = pd.DataFrame([result])
    record_prediction_observation(
        endpoint="predict",
        raw_df=raw_df,
        predictions_df=predictions_df,
        detector=detector,
    )

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
    detector=Depends(get_detector),
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
        results_df = detector.predict_batch(raw_df)
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
            risk_level     = str(row["risk_level"]),
        )
        for _, row in results_df.iterrows()
    ]

    n_fraud = int(results_df["is_fraud_pred"].sum())
    total   = len(predictions)
    record_prediction_observation(
        endpoint="predict_batch",
        raw_df=raw_df,
        predictions_df=results_df,
        detector=detector,
    )

    return BatchPredictionResponse(
        total          = total,
        fraud_detected = n_fraud,
        fraud_rate     = round(n_fraud / total, 6) if total > 0 else 0.0,
        threshold      = detector._threshold,
        predictions    = predictions,
    )
