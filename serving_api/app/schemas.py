from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# Request
class TransactionRequest(BaseModel):
    """Single raw transaction for fraud scoring.

    Fields mirror the raw CSV schema. All behavioural features are optional
    with sensible defaults so the API can handle partial data gracefully.
    """

    # Identifiers
    transaction_id: str = Field(..., description="Unique transaction ID")
    timestamp: str = Field(
        ..., description="ISO datetime, e.g. '2026-03-14 10:23:00'"
    )
    customer_id: str = Field(..., description="Customer identifier")

    # Core financials
    amount: float = Field(..., gt=0, description="Transaction amount in VND")
    customer_tier: str = Field(
        ..., description="INSPIRE | PRIORITY | MASS | PRIVATE"
    )

    # Card info
    card_type: Optional[str] = Field(None, description="VISA | MASTERCARD")
    card_tier: Optional[str] = Field(
        None, description="GOLD | PLATINUM | ..."
    )
    card_bin: Optional[int] = Field(
        None, description="First 6 digits of card number"
    )
    currency: Optional[str] = Field("VND", description="Transaction currency")
    account_age_days: Optional[int] = Field(None, ge=0)

    # Merchant info
    merchant_name: Optional[str] = None
    mcc_code: Optional[int] = None
    merchant_category: Optional[str] = None
    merchant_city: Optional[str] = None
    merchant_country: Optional[str] = Field(
        None, description="ISO country code, e.g. 'VN'"
    )

    # Device / network
    device_type: Optional[str] = None
    os: Optional[str] = None
    ip_country: Optional[str] = None

    # Behavioural features
    distance_from_home_km: Optional[float] = Field(None, ge=0)
    cvv_match: Optional[str] = Field(None, description="Y | N | N/A")
    is_3d_secure: Optional[str] = Field(None, description="Y | N | N/A")
    transaction_status: Optional[str] = Field(
        None, description="APPROVED | DECLINED"
    )
    tx_count_last_1h: Optional[int] = Field(None, ge=0)
    tx_count_last_24h: Optional[int] = Field(None, ge=0)
    time_since_last_tx_min: Optional[float] = Field(None, ge=0)
    avg_amount_last_30d: Optional[float] = Field(None, ge=0)
    amount_ratio_vs_avg: Optional[float] = Field(None, ge=0)
    is_new_device: Optional[int] = Field(None, ge=0, le=1)
    is_new_merchant: Optional[int] = Field(None, ge=0, le=1)

    # Time features (pre-computed in raw data)
    hour_of_day: Optional[int] = Field(None, ge=0, le=23)
    is_weekend: Optional[int] = Field(None, ge=0, le=1)

    @field_validator("customer_tier")
    @classmethod
    def validate_tier(cls, v: str) -> str:
        valid = {"INSPIRE", "PRIORITY", "MASS", "PRIVATE"}
        if v.upper() not in valid:
            raise ValueError(
                f"customer_tier must be one of {valid}, "
                f"got '{v}'"
            )
        return v.upper()

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        from datetime import datetime
        try:
            datetime.fromisoformat(v.replace("T", " "))
        except ValueError:
            raise ValueError(
                f"timestamp must be ISO format "
                f"(e.g. '2026-03-14 10:23:00'), got '{v}'"
            )
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "transaction_id":        "TX_001",
            "timestamp":             "2026-03-14 10:23:00",
            "customer_id":           "CUST_12345",
            "amount":                350000,
            "customer_tier":         "PRIORITY",
            "card_type":             "VISA",
            "card_tier":             "GOLD",
            "currency":              "VND",
            "merchant_name":         "Grab",
            "mcc_code":              4121,
            "merchant_category":     "Transport",
            "merchant_city":         "Ha Noi",
            "merchant_country":      "VN",
            "device_type":           "Mobile",
            "os":                    "iOS",
            "ip_country":            "VN",
            "distance_from_home_km": 2.5,
            "cvv_match":             "Y",
            "is_3d_secure":          "Y",
            "transaction_status":    "APPROVED",
            "tx_count_last_1h":      1,
            "tx_count_last_24h":     3,
            "time_since_last_tx_min": 120.0,
            "avg_amount_last_30d":   400000,
            "amount_ratio_vs_avg":   0.875,
            "is_new_device":         0,
            "is_new_merchant":       0,
            "card_bin":              411111,
            "account_age_days":      730,
            "is_weekend":            0,
            "hour_of_day":           10,
        }
    }}


class BatchTransactionRequest(BaseModel):
    """Batch of raw transactions for bulk scoring."""

    transactions: list[TransactionRequest] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of transactions to score (max 1000 per request)",
    )


# Response schema
class PredictionResponse(BaseModel):
    """Response for a single transaction prediction."""

    transaction_id: str = Field(..., description="Echoed from request")
    fraud_score: float = Field(
        ..., ge=0, le=1, description="Fraud probability [0, 1]"
    )
    is_fraud_pred: bool = Field(
        ..., description="True if fraud_score >= threshold"
    )
    threshold: float = Field(..., description="Decision threshold used")
    risk_level: str = Field(..., description="LOW | MEDIUM | HIGH")

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": "TX_001",
                "fraud_score": 0.0231,
                "is_fraud_pred": False,
                "threshold": 0.6788,
                "risk_level": "LOW",
            }
        }
    }


class BatchPredictionItem(BaseModel):
    """Single item in a batch prediction response."""

    transaction_id: str
    fraud_score: float
    is_fraud_pred: bool
    risk_level: str


class BatchPredictionResponse(BaseModel):
    """Response for a batch prediction request."""

    total: int = Field(..., description="Total transactions scored")
    fraud_detected: int = Field(
        ..., description="Number predicted as fraud"
    )
    fraud_rate: float = Field(
        ..., description="Fraction predicted as fraud"
    )
    threshold: float = Field(..., description="Decision threshold used")
    predictions: list[BatchPredictionItem]


class HealthResponse(BaseModel):
    """API and model health status."""

    status: str = Field(..., description="OK | DEGRADED | ERROR")
    model_type: str
    feature_count: int
    threshold: float | None = None
    best_iteration: int | None = None
    loaded_at: str | None = None
    api_version: str
    model_slot: str
    model_version: str | None = None
    model_loaded: bool
    load_error: str | None = None
    model_config = {"protected_namespaces": ()}


class DeploymentResponse(BaseModel):
    model_slot: str
    model_loaded: bool
    model_version: str | None = None
    models_dir: str
    processed_dir: str
    allow_empty_model: bool
    load_error: str | None = None
    loaded_at: str | None = None
    manifest: dict[str, Any] = Field(default_factory=dict)
    model_config = {"protected_namespaces": ()}


class DriftStatusResponse(BaseModel):
    ready: bool
    reference_mode: str
    reference_samples: int
    current_samples: int
    features_total: int
    features_alerting: int
    overall_score: float
    drift_ratio: float
    reason: str
    alert_threshold: float
    feature_scores: dict[str, float] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Structured error response."""

    error: str
    detail: Any = None
    status_code: int
