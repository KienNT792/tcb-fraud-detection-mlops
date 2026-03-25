from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


_DEFAULT_SOURCE_CANDIDATES = (
    "modelv2/raw/synthetic_credit_fraud_v2.csv",
    "data-generation/tcb_credit_fraud_dataset.csv",
)


class FraudDataGenerator:
    def __init__(
        self,
        *,
        repo_root: Path,
        source_csv: str | None = None,
        source_candidates: list[str] | None = None,
        fallback_to_synthetic: bool = True,
        seed: int = 42,
    ) -> None:
        self.repo_root = repo_root
        self.random = random.Random(seed)
        self.fallback_to_synthetic = fallback_to_synthetic
        self.source_candidates = self._build_source_candidates(
            source_csv=source_csv,
            source_candidates=source_candidates,
        )
        self.source_path: Path | None = None
        self._rows: list[dict[str, Any]] = []

    def prepare(self) -> None:
        self._rows = []
        self.source_path = None

        for candidate in self.source_candidates:
            candidate_path = (self.repo_root / candidate).resolve()
            if not candidate_path.exists():
                continue
            rows = self._load_rows(candidate_path)
            if rows:
                self.source_path = candidate_path
                self._rows = rows
                return

        if not self.fallback_to_synthetic:
            searched = ", ".join(str((self.repo_root / path).resolve()) for path in self.source_candidates)
            raise FileNotFoundError(
                "Simulation data not found. Checked: "
                f"{searched}"
            )

    def sample_payload(
        self,
        scenario_cfg: dict[str, Any],
        *,
        index: int = 0,
    ) -> dict[str, Any]:
        base_payload = self._sample_base_payload(index=index)
        return self._apply_scenario(
            payload=base_payload,
            scenario_cfg=scenario_cfg,
            index=index,
        )

    def sample_batch(
        self,
        size: int,
        scenario_cfg: dict[str, Any],
        *,
        starting_index: int = 0,
    ) -> list[dict[str, Any]]:
        return [
            self.sample_payload(
                scenario_cfg,
                index=starting_index + offset,
            )
            for offset in range(size)
        ]

    def _build_source_candidates(
        self,
        *,
        source_csv: str | None,
        source_candidates: list[str] | None,
    ) -> list[str]:
        if source_candidates:
            return list(source_candidates)
        if source_csv:
            return [source_csv]
        return list(_DEFAULT_SOURCE_CANDIDATES)

    def _load_rows(self, source_path: Path) -> list[dict[str, Any]]:
        with open(source_path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            return [row for row in reader]

    def _sample_base_payload(self, *, index: int) -> dict[str, Any]:
        if self._rows:
            row = self.random.choice(self._rows)
            return self._row_to_payload(row=row, index=index)
        return self._synthetic_payload(index=index)

    def _row_to_payload(
        self,
        *,
        row: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        timestamp = self._normalize_timestamp(
            row.get("timestamp"),
            index=index,
        )
        hour_of_day = self._to_int(row.get("hour_of_day"))
        is_weekend = self._to_int(row.get("is_weekend"))
        if hour_of_day is None or is_weekend is None:
            parsed = datetime.fromisoformat(timestamp.replace("T", " "))
            hour_of_day = parsed.hour
            is_weekend = 1 if parsed.weekday() >= 5 else 0

        amount = self._to_float(row.get("amount"), 250_000.0)
        avg_amount_last_30d = self._to_float(
            row.get("avg_amount_last_30d"),
            max(amount * 1.2, 280_000.0),
        )
        amount_ratio_vs_avg = self._to_float(
            row.get("amount_ratio_vs_avg"),
            round(amount / max(avg_amount_last_30d, 1.0), 4),
        )

        return {
            "transaction_id": row.get("transaction_id") or self._random_id("TX"),
            "timestamp": timestamp,
            "customer_id": row.get("customer_id") or self._random_id("CUST"),
            "amount": round(max(1.0, amount), 2),
            "customer_tier": str(row.get("customer_tier") or "MASS").upper(),
            "card_type": row.get("card_type") or "VISA",
            "card_tier": row.get("card_tier") or "GOLD",
            "card_bin": self._to_int(row.get("card_bin"), 411111),
            "currency": row.get("currency") or "VND",
            "account_age_days": self._to_int(row.get("account_age_days"), 720),
            "merchant_name": row.get("merchant_name") or "Shopee",
            "mcc_code": self._to_int(row.get("mcc_code"), 5732),
            "merchant_category": row.get("merchant_category") or "Retail",
            "merchant_city": row.get("merchant_city") or "Ha Noi",
            "merchant_country": row.get("merchant_country") or "VN",
            "device_type": row.get("device_type") or "Mobile",
            "os": row.get("os") or "Android",
            "ip_country": row.get("ip_country") or "VN",
            "distance_from_home_km": self._to_float(
                row.get("distance_from_home_km"),
                3.0,
            ),
            "cvv_match": row.get("cvv_match") or "Y",
            "is_3d_secure": row.get("is_3d_secure") or "Y",
            "transaction_status": row.get("transaction_status") or "APPROVED",
            "tx_count_last_1h": self._to_int(row.get("tx_count_last_1h"), 1),
            "tx_count_last_24h": self._to_int(row.get("tx_count_last_24h"), 4),
            "time_since_last_tx_min": self._to_float(
                row.get("time_since_last_tx_min"),
                90.0,
            ),
            "avg_amount_last_30d": round(max(1.0, avg_amount_last_30d), 2),
            "amount_ratio_vs_avg": round(max(0.01, amount_ratio_vs_avg), 4),
            "is_new_device": self._bounded_binary(
                row.get("is_new_device"),
                default=0,
            ),
            "is_new_merchant": self._bounded_binary(
                row.get("is_new_merchant"),
                default=0,
            ),
            "hour_of_day": hour_of_day,
            "is_weekend": is_weekend,
        }

    def _synthetic_payload(self, *, index: int) -> dict[str, Any]:
        timestamp = datetime.now(tz=timezone.utc) - timedelta(minutes=index % 180)
        amount = max(150_000, int(self.random.gauss(260_000, 45_000)))
        avg_amount_last_30d = max(
            230_000,
            int(self.random.gauss(340_000, 55_000)),
        )
        amount_ratio_vs_avg = round(
            min(1.4, max(0.35, amount / max(avg_amount_last_30d, 1))),
            4,
        )
        return {
            "transaction_id": self._random_id("TX"),
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "customer_id": self._random_id("CUST"),
            "amount": float(amount),
            "customer_tier": self.random.choices(
                ["MASS", "PRIORITY", "PRIVATE"],
                weights=[0.65, 0.25, 0.10],
                k=1,
            )[0],
            "card_type": self.random.choice(["VISA", "MASTERCARD", "DEBIT"]),
            "card_tier": self.random.choice(["GOLD", "PLATINUM", "CLASSIC"]),
            "card_bin": self.random.choice([411111, 437797, 488122]),
            "currency": "VND",
            "account_age_days": max(60, int(self.random.gauss(840, 180))),
            "merchant_name": self.random.choice(["Shopee", "Tiki", "Grab"]),
            "mcc_code": self.random.choice([4121, 5411, 5732, 5812]),
            "merchant_category": self.random.choice(
                ["Retail", "Transport", "Electronics", "Food"]
            ),
            "merchant_city": self.random.choice(
                ["Ha Noi", "Ho Chi Minh", "Da Nang"]
            ),
            "merchant_country": "VN",
            "device_type": self.random.choice(["Mobile", "Desktop"]),
            "os": self.random.choice(["Android", "iOS", "Windows"]),
            "ip_country": "VN",
            "distance_from_home_km": round(abs(self.random.gauss(3.5, 1.4)), 3),
            "cvv_match": "Y",
            "is_3d_secure": "Y",
            "transaction_status": "APPROVED",
            "tx_count_last_1h": max(0, int(self.random.gauss(1.5, 0.8))),
            "tx_count_last_24h": max(1, int(self.random.gauss(5.0, 1.5))),
            "time_since_last_tx_min": round(abs(self.random.gauss(130.0, 45.0)), 3),
            "avg_amount_last_30d": float(avg_amount_last_30d),
            "amount_ratio_vs_avg": amount_ratio_vs_avg,
            "is_new_device": 0,
            "is_new_merchant": 0,
            "hour_of_day": timestamp.hour,
            "is_weekend": 1 if timestamp.weekday() >= 5 else 0,
        }

    def _apply_scenario(
        self,
        *,
        payload: dict[str, Any],
        scenario_cfg: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        enriched = payload.copy()
        enriched["transaction_id"] = self._random_id("TX")
        enriched["timestamp"] = self._normalize_timestamp(
            enriched.get("timestamp"),
            index=index,
        )

        amount_range = scenario_cfg.get("amount_multiplier", [1.0, 1.0])
        if len(amount_range) == 2:
            amount_multiplier = self.random.uniform(
                float(amount_range[0]),
                float(amount_range[1]),
            )
        else:
            amount_multiplier = 1.0
        enriched["amount"] = round(
            max(1.0, float(enriched["amount"]) * amount_multiplier),
            2,
        )

        risk_bias = float(scenario_cfg.get("high_risk_bias", 0.0))
        if self.random.random() < risk_bias:
            enriched["distance_from_home_km"] = round(
                float(enriched["distance_from_home_km"])
                + self.random.uniform(25.0, 320.0),
                3,
            )
            enriched["tx_count_last_1h"] = max(
                int(enriched["tx_count_last_1h"]),
                self.random.randint(5, 16),
            )
            enriched["tx_count_last_24h"] = max(
                int(enriched["tx_count_last_24h"]),
                self.random.randint(18, 72),
            )
            enriched["time_since_last_tx_min"] = round(
                self.random.uniform(2.0, 45.0),
                3,
            )
            enriched["transaction_status"] = "APPROVED"

        merchant_country_pool = scenario_cfg.get("merchant_country_pool")
        if merchant_country_pool:
            enriched["merchant_country"] = self.random.choice(merchant_country_pool)

        ip_country_pool = scenario_cfg.get("ip_country_pool")
        if ip_country_pool:
            enriched["ip_country"] = self.random.choice(ip_country_pool)

        device_type_pool = scenario_cfg.get("device_type_pool")
        if device_type_pool:
            enriched["device_type"] = self.random.choice(device_type_pool)

        os_pool = scenario_cfg.get("os_pool")
        if os_pool:
            enriched["os"] = self.random.choice(os_pool)

        if self.random.random() < float(scenario_cfg.get("force_new_device_ratio", 0.0)):
            enriched["is_new_device"] = 1
        if self.random.random() < float(scenario_cfg.get("force_new_merchant_ratio", 0.0)):
            enriched["is_new_merchant"] = 1

        night_ratio = float(scenario_cfg.get("night_transaction_ratio", 0.0))
        if self.random.random() < night_ratio:
            enriched["hour_of_day"] = self.random.choice([0, 1, 2, 3, 22, 23])

        avg_amount_last_30d = max(
            1.0,
            float(enriched.get("avg_amount_last_30d", enriched["amount"])),
        )
        enriched["amount_ratio_vs_avg"] = round(
            max(0.01, float(enriched["amount"]) / avg_amount_last_30d),
            4,
        )

        return enriched

    def _normalize_timestamp(self, value: Any, *, index: int) -> str:
        if isinstance(value, str) and value.strip():
            try:
                parsed = datetime.fromisoformat(value.replace("T", " "))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                shifted = parsed + timedelta(minutes=index % 15)
                return shifted.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass

        timestamp = datetime.now(tz=timezone.utc) - timedelta(minutes=index % 120)
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")

    def _random_id(self, prefix: str) -> str:
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        suffix = "".join(self.random.choice(alphabet) for _ in range(8))
        return f"{prefix}_{suffix}"

    @staticmethod
    def _to_int(value: Any, default: int | None = None) -> int | None:
        if value in (None, ""):
            return default
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        if value in (None, ""):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _bounded_binary(cls, value: Any, *, default: int = 0) -> int:
        parsed = cls._to_int(value, default)
        return 1 if parsed and int(parsed) > 0 else 0
