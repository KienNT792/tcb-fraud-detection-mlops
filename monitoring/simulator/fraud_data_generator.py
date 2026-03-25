from __future__ import annotations

import csv
import random
import subprocess
from pathlib import Path
from typing import Any

# Must mirror preprocess.py _YN_MAP and _STATUS_MAP exactly
_YN_MAP: dict[str, int] = {"Y": 1, "N": 0, "N/A": 0}
_STATUS_MAP: dict[str, int] = {"APPROVED": 1, "DECLINED": 0}


class FraudDataGenerator:
    def __init__(
        self,
        *,
        repo_root: Path,
        source_csv: str,
        regenerate_before_run: bool = False,
        java_main_class: str = "com.SyntheticTCBFraudDataGenerator",
        seed: int = 42,
    ) -> None:
        self.repo_root = repo_root
        self.source_path = (repo_root / source_csv).resolve()
        self.regenerate_before_run = regenerate_before_run
        self.java_main_class = java_main_class
        self.random = random.Random(seed)
        self._rows: list[dict[str, Any]] = []

    def prepare(self) -> None:
        if self.regenerate_before_run:
            self._try_generate_with_java()
        if not self.source_path.exists():
            raise FileNotFoundError(f"Simulation data not found: {self.source_path}")
        self._rows = self._load_rows()
        if not self._rows:
            raise ValueError(f"Simulation data is empty: {self.source_path}")

    def _try_generate_with_java(self) -> None:
        project_dir = self.repo_root / "data-generation"
        if not project_dir.exists():
            return
        # Best-effort data generation. If Java/Maven is unavailable, keep existing CSV.
        command = (
            "mvn -q -DskipTests package && "
            f"java -cp target/classes {self.java_main_class}"
        )
        subprocess.run(
            command,
            cwd=project_dir,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
        )

    def _load_rows(self) -> list[dict[str, Any]]:
        with open(self.source_path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            return [row for row in reader]

    def sample_payload(self, scenario_cfg: dict[str, Any]) -> dict[str, Any]:
        row = self.random.choice(self._rows)
        return self._to_payload(row=row, scenario_cfg=scenario_cfg)

    def _to_payload(self, *, row: dict[str, str], scenario_cfg: dict[str, Any]) -> dict[str, Any]:
        # Base mapping from generated CSV schema to serving request schema.
        payload: dict[str, Any] = {
            "transaction_id": row["transaction_id"],
            "timestamp": row["timestamp"],
            "customer_id": row["customer_id"],
            "amount": self._to_float(row.get("amount"), 100_000.0),
            "customer_tier": row.get("customer_tier", "MASS"),
            "card_type": row.get("card_type"),
            "card_tier": row.get("card_tier"),
            "card_bin": self._to_int(row.get("card_bin")),
            "currency": row.get("currency", "VND"),
            "account_age_days": self._to_int(row.get("account_age_days")),
            "merchant_name": row.get("merchant_name"),
            "mcc_code": self._to_int(row.get("mcc_code")),
            "merchant_category": row.get("merchant_category"),
            "merchant_city": row.get("merchant_city"),
            "merchant_country": row.get("merchant_country"),
            "device_type": row.get("device_type"),
            "os": row.get("os"),
            "ip_country": row.get("ip_country"),
            "distance_from_home_km": self._to_float(row.get("distance_from_home_km"), 0.0),
            # These 3 fields are encoded at train time by preprocess.clean_data().
            # Simulator MUST send integers, not raw strings, to match model input.
            "cvv_match": _YN_MAP.get(str(row.get("cvv_match", "N")), 0),
            "is_3d_secure": _YN_MAP.get(str(row.get("is_3d_secure", "N")), 0),
            "transaction_status": _STATUS_MAP.get(str(row.get("transaction_status", "APPROVED")), 1),
            "tx_count_last_1h": self._to_int(row.get("tx_count_last_1h"), 0),
            "tx_count_last_24h": self._to_int(row.get("tx_count_last_24h"), 1),
            "time_since_last_tx_min": self._to_float(row.get("time_since_last_tx_min"), 30.0),
            "avg_amount_last_30d": self._to_float(row.get("avg_amount_last_30d"), 200_000.0),
            "amount_ratio_vs_avg": self._to_float(row.get("amount_ratio_vs_avg"), 1.0),
            "is_new_device": self._to_int(row.get("is_new_device"), 0),
            "is_new_merchant": self._to_int(row.get("is_new_merchant"), 0),
            "hour_of_day": self._to_int(row.get("hour_of_day")),
            "is_weekend": self._to_int(row.get("is_weekend")),
        }

        amount_range = scenario_cfg.get("amount_multiplier", [1.0, 1.0])
        low_mult = float(amount_range[0])
        high_mult = float(amount_range[1])
        amount_mult = self.random.uniform(low_mult, high_mult)
        payload["amount"] = round(max(1.0, payload["amount"] * amount_mult), 2)

        risk_bias = float(scenario_cfg.get("high_risk_bias", 0.0))
        if self.random.random() < risk_bias:
            payload["merchant_country"] = self.random.choice(["SG", "US", "JP"])
            payload["ip_country"] = self.random.choice(["SG", "US", "JP"])
            payload["transaction_status"] = 1  # APPROVED encoded as int
            payload["distance_from_home_km"] = round(
                payload["distance_from_home_km"] + self.random.uniform(20.0, 300.0),
                2,
            )
            payload["tx_count_last_1h"] = max(payload["tx_count_last_1h"], self.random.randint(8, 30))
            payload["tx_count_last_24h"] = max(payload["tx_count_last_24h"], self.random.randint(20, 100))

        if self.random.random() < float(scenario_cfg.get("force_new_device_ratio", 0.0)):
            payload["is_new_device"] = 1
        if self.random.random() < float(scenario_cfg.get("force_new_merchant_ratio", 0.0)):
            payload["is_new_merchant"] = 1

        return payload

    @staticmethod
    def _to_int(value: str | None, default: int | None = None) -> int | None:
        if value is None or value == "":
            return default
        try:
            return int(float(value))
        except ValueError:
            return default

    @staticmethod
    def _to_float(value: str | None, default: float) -> float:
        if value is None or value == "":
            return default
        try:
            return float(value)
        except ValueError:
            return default
