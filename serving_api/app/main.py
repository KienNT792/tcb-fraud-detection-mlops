from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
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
    get_runtime_info,
    load_model,
    reload_model,
    unload_model,
)
from .observability import (
    bootstrap_observability,
    get_drift_alert_threshold,
    get_drift_snapshot,
    record_http_observation,
    record_prediction_observation,
    shutdown_observability,
)
from .schemas import (
    BatchPredictionItem,
    BatchPredictionResponse,
    BatchTransactionRequest,
    DeploymentResponse,
    DriftStatusResponse,
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


def get_optional_detector():
    return get_detector(required=False)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("=" * 50)
    logger.info("TCB FRAUD DETECTION API — STARTING UP")
    logger.info("=" * 50)
    try:
        detector = load_model()
        bootstrap_observability(detector)
        if detector is None:
            logger.warning(
                "No model loaded at startup."
                " Service is in standby mode."
            )
        else:
            logger.info("Model loaded successfully. API ready.")
    except FileNotFoundError as exc:
        logger.error("STARTUP FAILED — artifact missing: %s", exc)
        raise

    yield

    logger.info("Shutting down — releasing model.")
    shutdown_observability()
    unload_model()
    logger.info("API shutdown complete.")


app = FastAPI(
    title="TCB Fraud Detection API",
    description=(
        "Real-time credit card fraud scoring using XGBoost with deployment "
        "metadata, drift monitoring, and reload hooks for canary rollout."
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
    logger.warning(
        "Prometheus instrumentation disabled: dependency not installed."
    )

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
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.exception_handler(ValueError)
async def value_error_handler(
    request: Request,
    exc: ValueError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Validation error",
            detail=str(exc),
            status_code=422,
        ).model_dump(),
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(
    request: Request,
    exc: RuntimeError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            error="Model not available",
            detail=str(exc),
            status_code=503,
        ).model_dump(),
    )


@app.get("/", summary="API info", tags=["Info"])
async def root() -> dict[str, str]:
    return {
        "name": "TCB Fraud Detection API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
        "batch": "POST /predict/batch",
        "deployment": "/deployment",
        "drift": "/monitoring/drift",
        "reload": "POST /admin/reload",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Model health check",
    tags=["Health"],
)
async def health(
    detector=Depends(get_optional_detector),
) -> HealthResponse:
    runtime = get_runtime_info()

    if detector is None:
        return HealthResponse(
            status="EMPTY" if runtime["allow_empty_model"] else "ERROR",
            model_type="UNLOADED",
            feature_count=0,
            threshold=None,
            best_iteration=None,
            loaded_at=runtime["loaded_at"],
            api_version=API_VERSION,
            model_slot=runtime["model_slot"],
            model_version=runtime["model_version"],
            model_loaded=False,
            load_error=runtime["load_error"],
        )

    info = detector.health_check()
    return HealthResponse(
        status=info["status"],
        model_type=info["model_type"],
        feature_count=info["feature_count"],
        threshold=info["threshold"],
        best_iteration=info["best_iteration"],
        loaded_at=info["loaded_at"],
        api_version=API_VERSION,
        model_slot=runtime["model_slot"],
        model_version=runtime["model_version"],
        model_loaded=True,
        load_error=runtime["load_error"],
    )


@app.get(
    "/deployment",
    response_model=DeploymentResponse,
    summary="Deployment metadata",
    tags=["Deployment"],
)
async def deployment_info() -> DeploymentResponse:
    return DeploymentResponse(**get_runtime_info())


@app.get(
    "/monitoring/drift",
    response_model=DriftStatusResponse,
    summary="Current drift monitor snapshot",
    tags=["Monitoring"],
)
async def drift_status() -> DriftStatusResponse:
    snapshot = asdict(get_drift_snapshot())
    snapshot["alert_threshold"] = get_drift_alert_threshold()
    return DriftStatusResponse(**snapshot)


@app.post(
    "/admin/reload",
    response_model=DeploymentResponse,
    summary="Reload model artifacts from disk",
    tags=["Admin"],
)
async def admin_reload() -> JSONResponse:
    try:
        detector = reload_model()
        bootstrap_observability(detector)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Reload failed: {exc}",
        ) from exc

    runtime = DeploymentResponse(**get_runtime_info()).model_dump()
    response_code = (
        status.HTTP_200_OK
        if runtime["model_loaded"]
        else status.HTTP_202_ACCEPTED
    )
    return JSONResponse(status_code=response_code, content=runtime)


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
    payload = request.model_dump()
    try:
        result = detector.predict_single(payload)
    except Exception as exc:
        logger.exception(
            "predict() failed for tx_id=%s",
            request.transaction_id,
        )
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
        500: {
            "model": ErrorResponse,
            "description": "Batch prediction failed",
        },
    },
)
async def predict_batch(
    request: BatchTransactionRequest,
    detector=Depends(get_detector),
) -> BatchPredictionResponse:
    try:
        raw_df = pd.DataFrame([tx.model_dump() for tx in request.transactions])
        results_df = detector.predict_batch(raw_df)
    except Exception as exc:
        logger.exception(
            "predict_batch() failed — %d transactions",
            len(request.transactions),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {exc}",
        ) from exc

    predictions = [
        BatchPredictionItem(
            transaction_id=str(row["transaction_id"]),
            fraud_score=float(row["fraud_score"]),
            is_fraud_pred=bool(row["is_fraud_pred"]),
            risk_level=str(row["risk_level"]),
        )
        for _, row in results_df.iterrows()
    ]

    n_fraud = int(results_df["is_fraud_pred"].sum())
    total = len(predictions)
    record_prediction_observation(
        endpoint="predict_batch",
        raw_df=raw_df,
        predictions_df=results_df,
        detector=detector,
    )

    return BatchPredictionResponse(
        total=total,
        fraud_detected=n_fraud,
        fraud_rate=round(n_fraud / total, 6) if total > 0 else 0.0,
        threshold=detector._threshold,
        predictions=predictions,
    )
