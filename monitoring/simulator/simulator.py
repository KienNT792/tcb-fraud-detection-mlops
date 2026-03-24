from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

from monitoring.simulator.fraud_data_generator import FraudDataGenerator


@dataclass(slots=True)
class SimulationResult:
    scenario: str
    total_requests: int
    success_requests: int
    failed_requests: int
    elapsed_seconds: float
    avg_latency_ms: float


class FraudPredictionSimulator:
    def __init__(self, config_path: str = "monitoring/simulator/sim_config.yaml") -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.config_path = (self.repo_root / config_path).resolve()
        with open(self.config_path, encoding="utf-8") as fh:
            self.config: dict[str, Any] = yaml.safe_load(fh)

        data_cfg = self.config["data"]
        self.generator = FraudDataGenerator(
            repo_root=self.repo_root,
            source_csv=data_cfg["source_csv"],
            regenerate_before_run=bool(data_cfg.get("regenerate_before_run", False)),
            java_main_class=data_cfg.get("java_main_class", "com.SyntheticTCBFraudDataGenerator"),
        )
        self.generator.prepare()

    def check_health(self) -> None:
        health_url = self.config["api"]["health_url"]
        timeout_seconds = float(self.config["defaults"].get("timeout_seconds", 15))
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.get(health_url)
            resp.raise_for_status()

    def run(
        self,
        *,
        scenario: str | None = None,
        requests: int | None = None,
        rps: float | None = None,
    ) -> SimulationResult:
        defaults = self.config["defaults"]
        resolved_scenario = scenario or defaults["scenario"]
        scenario_cfg = self.config["scenarios"].get(resolved_scenario)
        if not scenario_cfg:
            raise ValueError(f"Unknown scenario: {resolved_scenario}")

        total_requests = int(requests or defaults["requests"])
        resolved_rps = float(rps or defaults["rps"])
        timeout_seconds = float(defaults.get("timeout_seconds", 15))
        prediction_url = self.config["api"]["prediction_url"]

        success = 0
        failed = 0
        latencies_ms: list[float] = []
        interval = 1.0 / max(0.1, resolved_rps)
        started = time.perf_counter()

        with httpx.Client(timeout=timeout_seconds) as client:
            for _ in range(total_requests):
                payload = self.generator.sample_payload(scenario_cfg)
                req_started = time.perf_counter()
                try:
                    resp = client.post(prediction_url, json=payload)
                    if resp.status_code < 400:
                        success += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
                finally:
                    latencies_ms.append((time.perf_counter() - req_started) * 1000)
                time.sleep(interval)

        elapsed = time.perf_counter() - started
        avg_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
        return SimulationResult(
            scenario=resolved_scenario,
            total_requests=total_requests,
            success_requests=success,
            failed_requests=failed,
            elapsed_seconds=elapsed,
            avg_latency_ms=avg_latency_ms,
        )

    def fetch_drift_status(self) -> dict[str, Any]:
        drift_url = self.config["api"]["drift_url"]
        timeout_seconds = float(self.config["defaults"].get("timeout_seconds", 15))
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.get(drift_url)
            resp.raise_for_status()
            return resp.json()
