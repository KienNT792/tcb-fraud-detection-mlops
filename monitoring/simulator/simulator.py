from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import yaml

from monitoring.simulator.fraud_data_generator import FraudDataGenerator

_DEFAULT_PORT_CANDIDATES = (8000, 80)
_PROMETHEUS_PORT = 9090
_METRIC_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)$"
)
_PROMETHEUS_QUERIES = {
    "prediction_samples_15m": "sum(increase(tcb_prediction_samples_total[15m]))",
    "prediction_requests_15m": "sum(increase(tcb_prediction_requests_total[15m]))",
    "fraud_rate_15m": (
        "sum(increase(tcb_prediction_fraud_total[15m])) "
        "/ clamp_min(sum(increase(tcb_prediction_samples_total[15m])), 1)"
    ),
    "drift_ratio": "max(tcb_drift_ratio)",
    "drift_features_alerting": "max(tcb_drift_features_alerting)",
    "drift_reference_samples": "max(tcb_drift_reference_samples)",
    "drift_current_samples": "max(tcb_drift_current_samples)",
}


@dataclass(slots=True)
class PhaseResult:
    name: str
    endpoint: str
    total_samples: int
    successful_samples: int
    failed_samples: int
    http_requests_sent: int
    successful_http_requests: int
    failed_http_requests: int
    fraud_predictions: int
    elapsed_seconds: float
    avg_latency_ms: float


@dataclass(slots=True)
class SimulationResult:
    scenario: str
    mode: str
    base_url: str
    batch_size: int
    rps: float
    total_requests: int
    success_requests: int
    failed_requests: int
    http_requests_sent: int
    successful_http_requests: int
    failed_http_requests: int
    fraud_predictions: int
    elapsed_seconds: float
    avg_latency_ms: float
    warmup_samples: int = 0
    warmup: dict[str, Any] = field(default_factory=dict)
    health: dict[str, Any] = field(default_factory=dict)
    metrics_before: dict[str, float] = field(default_factory=dict)
    metrics_after: dict[str, float] = field(default_factory=dict)
    metrics_delta: dict[str, float] = field(default_factory=dict)
    drift_before: dict[str, Any] = field(default_factory=dict)
    drift_after: dict[str, Any] = field(default_factory=dict)
    prometheus: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


class FraudPredictionSimulator:
    def __init__(
        self,
        config_path: str = "monitoring/simulator/sim_config.yaml",
        *,
        base_url: str | None = None,
        prometheus_url: str | None = None,
    ) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.config_path = (self.repo_root / config_path).resolve()
        with open(self.config_path, encoding="utf-8") as fh:
            self.config: dict[str, Any] = yaml.safe_load(fh)

        defaults = self.config.get("defaults", {})
        api_cfg = self.config.get("api", {})
        data_cfg = self.config.get("data", {})

        self.timeout_seconds = float(defaults.get("timeout_seconds", 20))
        self.port_candidates = tuple(
            int(port)
            for port in api_cfg.get("api_port_candidates", _DEFAULT_PORT_CANDIDATES)
        )
        configured_base_url = base_url or api_cfg.get("base_url")
        self.base_url = self._resolve_api_base_url(
            requested_url=configured_base_url,
            port_candidates=self.port_candidates,
        )
        self.prometheus_url = self._resolve_prometheus_url(
            explicit_url=prometheus_url,
            configured_url=api_cfg.get("prometheus_url"),
            prefer_base_host=base_url is not None,
        )

        self.health_url = f"{self.base_url}/health"
        self.prediction_url = f"{self.base_url}/predict"
        self.batch_prediction_url = f"{self.base_url}/predict/batch"
        self.drift_url = f"{self.base_url}/monitoring/drift"
        self.metrics_url = f"{self.base_url}/metrics"

        self.generator = FraudDataGenerator(
            repo_root=self.repo_root,
            source_csv=data_cfg.get("source_csv"),
            source_candidates=data_cfg.get("source_candidates"),
            fallback_to_synthetic=bool(data_cfg.get("fallback_to_synthetic", True)),
            seed=int(data_cfg.get("seed", 42)),
        )
        self.generator.prepare()

    def check_health(self) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.get(self.health_url)
            response.raise_for_status()
            return response.json()

    def run(
        self,
        *,
        scenario: str | None = None,
        requests: int | None = None,
        rps: float | None = None,
        batch_size: int | None = None,
        warmup_samples: int | None = None,
        mode: str | None = None,
        capture_metrics: bool = True,
        capture_drift: bool = True,
        capture_prometheus: bool = False,
        prometheus_wait_seconds: float | None = None,
    ) -> SimulationResult:
        defaults = self.config["defaults"]
        resolved_scenario = scenario or defaults["scenario"]
        scenario_cfg = self.config["scenarios"].get(resolved_scenario)
        if not scenario_cfg:
            raise ValueError(f"Unknown scenario: {resolved_scenario}")

        resolved_mode = (mode or defaults.get("mode", "batch")).lower()
        if resolved_mode not in {"single", "batch"}:
            raise ValueError("mode must be 'single' or 'batch'")

        resolved_batch_size = int(batch_size or defaults.get("batch_size", 1))
        if resolved_mode == "single":
            resolved_batch_size = 1
        if resolved_batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        resolved_rps = float(rps or defaults["rps"])
        resolved_requests = int(requests or defaults["requests"])
        resolved_warmup_samples = int(
            warmup_samples
            if warmup_samples is not None
            else defaults.get("warmup_samples", 0)
        )

        health_payload = self.check_health()
        issues: list[str] = []
        metrics_before = (
            self._safe_metrics_snapshot(issues, when="before")
            if capture_metrics
            else {}
        )
        drift_before = (
            self._safe_drift_status(issues, when="before")
            if capture_drift
            else {}
        )

        warmup_result: PhaseResult | None = None
        if resolved_warmup_samples > 0:
            warmup_result = self._run_phase(
                name="warmup",
                scenario_cfg=self.config["scenarios"]["normal"],
                total_samples=resolved_warmup_samples,
                rps=resolved_rps,
                batch_size=resolved_batch_size,
                mode=resolved_mode,
            )

        scenario_result = self._run_phase(
            name=resolved_scenario,
            scenario_cfg=scenario_cfg,
            total_samples=resolved_requests,
            rps=resolved_rps,
            batch_size=resolved_batch_size,
            mode=resolved_mode,
        )

        metrics_after = (
            self._safe_metrics_snapshot(issues, when="after")
            if capture_metrics
            else {}
        )
        drift_after = (
            self._safe_drift_status(issues, when="after")
            if capture_drift
            else {}
        )
        metrics_delta = self._metric_delta(metrics_before, metrics_after)

        prometheus_payload: dict[str, Any] = {}
        if capture_prometheus:
            wait_seconds = float(
                prometheus_wait_seconds
                if prometheus_wait_seconds is not None
                else defaults.get("prometheus_wait_seconds", 20)
            )
            prometheus_payload = self.fetch_prometheus_snapshot(
                wait_seconds=wait_seconds,
            )
            self._append_prometheus_alignment_issue(
                issues=issues,
                metrics_delta=metrics_delta,
                prometheus_payload=prometheus_payload,
            )

        return SimulationResult(
            scenario=resolved_scenario,
            mode=resolved_mode,
            base_url=self.base_url,
            batch_size=resolved_batch_size,
            rps=resolved_rps,
            total_requests=scenario_result.total_samples,
            success_requests=scenario_result.successful_samples,
            failed_requests=scenario_result.failed_samples,
            http_requests_sent=scenario_result.http_requests_sent,
            successful_http_requests=scenario_result.successful_http_requests,
            failed_http_requests=scenario_result.failed_http_requests,
            fraud_predictions=scenario_result.fraud_predictions,
            elapsed_seconds=scenario_result.elapsed_seconds,
            avg_latency_ms=scenario_result.avg_latency_ms,
            warmup_samples=resolved_warmup_samples,
            warmup=asdict(warmup_result) if warmup_result else {},
            health=health_payload,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            metrics_delta=metrics_delta,
            drift_before=drift_before,
            drift_after=drift_after,
            prometheus=prometheus_payload,
            issues=issues,
        )

    def fetch_drift_status(self) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.get(self.drift_url)
            response.raise_for_status()
            return response.json()

    def fetch_metrics_snapshot(self) -> dict[str, float]:
        metric_map = {
            "prediction_requests_total": "tcb_prediction_requests_total",
            "prediction_samples_total": "tcb_prediction_samples_total",
            "prediction_fraud_total": "tcb_prediction_fraud_total",
            "drift_baseline_ready": "tcb_drift_baseline_ready",
            "drift_reference_samples": "tcb_drift_reference_samples",
            "drift_current_samples": "tcb_drift_current_samples",
            "drift_features_total": "tcb_drift_features_total",
            "drift_features_alerting": "tcb_drift_features_alerting",
            "drift_ratio": "tcb_drift_ratio",
            "drift_overall_score": "tcb_drift_overall_score",
        }
        values = {key: 0.0 for key in metric_map}
        inverse = {metric_name: key for key, metric_name in metric_map.items()}

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.get(self.metrics_url)
            response.raise_for_status()
            for line in response.text.splitlines():
                match = _METRIC_LINE_RE.match(line.strip())
                if not match:
                    continue
                metric_name = match.group("name")
                key = inverse.get(metric_name)
                if key is None:
                    continue
                values[key] += float(match.group("value"))
        return values

    def fetch_prometheus_snapshot(self, *, wait_seconds: float = 0.0) -> dict[str, Any]:
        if not self.prometheus_url:
            return {
                "configured": False,
                "error": "prometheus_url_not_configured",
            }

        if wait_seconds > 0:
            time.sleep(wait_seconds)

        results: dict[str, Any] = {
            "configured": True,
            "url": self.prometheus_url,
            "wait_seconds": wait_seconds,
            "queries": {},
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            for name, query in _PROMETHEUS_QUERIES.items():
                try:
                    response = client.get(
                        f"{self.prometheus_url}/api/v1/query",
                        params={"query": query},
                    )
                    response.raise_for_status()
                    payload = response.json()
                    results["queries"][name] = {
                        "query": query,
                        "value": self._extract_prometheus_value(payload),
                    }
                except Exception as exc:  # pragma: no cover - runtime guard
                    results["queries"][name] = {
                        "query": query,
                        "error": str(exc),
                    }
        return results

    def _safe_metrics_snapshot(
        self,
        issues: list[str],
        *,
        when: str,
    ) -> dict[str, float]:
        try:
            return self.fetch_metrics_snapshot()
        except Exception as exc:
            issues.append(f"metrics_{when}_failed: {exc}")
            return {}

    def _safe_drift_status(
        self,
        issues: list[str],
        *,
        when: str,
    ) -> dict[str, Any]:
        try:
            return self.fetch_drift_status()
        except Exception as exc:
            issues.append(f"drift_{when}_failed: {exc}")
            return {"error": str(exc)}

    def _append_prometheus_alignment_issue(
        self,
        *,
        issues: list[str],
        metrics_delta: dict[str, float],
        prometheus_payload: dict[str, Any],
    ) -> None:
        prediction_delta = float(metrics_delta.get("prediction_samples_total", 0.0))
        if prediction_delta <= 0:
            return

        queries = prometheus_payload.get("queries", {})
        prom_prediction = queries.get("prediction_samples_15m", {}).get("value")
        prom_drift = queries.get("drift_current_samples", {}).get("value")

        if prom_prediction not in (None, 0.0) or prom_drift not in (None, 0.0, 1.0):
            return

        issues.append(
            "prometheus_not_following_api_metrics: API /metrics changed but "
            "Prometheus still shows the old series. If you are running the "
            "simulator on the VPS host, target http://127.0.0.1:8002 so "
            "traffic reaches the same stable FastAPI instance that Prometheus "
            "scrapes."
        )

    def _run_phase(
        self,
        *,
        name: str,
        scenario_cfg: dict[str, Any],
        total_samples: int,
        rps: float,
        batch_size: int,
        mode: str,
    ) -> PhaseResult:
        if total_samples <= 0:
            return PhaseResult(
                name=name,
                endpoint="/predict",
                total_samples=0,
                successful_samples=0,
                failed_samples=0,
                http_requests_sent=0,
                successful_http_requests=0,
                failed_http_requests=0,
                fraud_predictions=0,
                elapsed_seconds=0.0,
                avg_latency_ms=0.0,
            )

        if rps <= 0:
            raise ValueError("rps must be > 0")

        endpoint = "/predict/batch" if mode == "batch" else "/predict"
        samples_sent = 0
        successful_samples = 0
        failed_samples = 0
        successful_http_requests = 0
        failed_http_requests = 0
        fraud_predictions = 0
        latencies_ms: list[float] = []
        request_interval = 1.0 / max(rps, 0.1)

        started = time.perf_counter()
        with httpx.Client(timeout=self.timeout_seconds) as client:
            while samples_sent < total_samples:
                phase_started = time.perf_counter()
                current_batch_size = min(
                    batch_size if mode == "batch" else 1,
                    total_samples - samples_sent,
                )
                payloads = self.generator.sample_batch(
                    current_batch_size,
                    scenario_cfg,
                    starting_index=samples_sent,
                )
                try:
                    request_started = time.perf_counter()
                    if mode == "batch":
                        response = client.post(
                            self.batch_prediction_url,
                            json={"transactions": payloads},
                        )
                    else:
                        response = client.post(
                            self.prediction_url,
                            json=payloads[0],
                        )
                    latency_ms = (time.perf_counter() - request_started) * 1000
                    latencies_ms.append(latency_ms)
                    response.raise_for_status()
                    response_payload = response.json()

                    successful_http_requests += 1
                    successful_samples += current_batch_size
                    fraud_predictions += self._extract_fraud_predictions(
                        response_payload=response_payload,
                        mode=mode,
                    )
                except Exception:
                    failed_http_requests += 1
                    failed_samples += current_batch_size
                finally:
                    samples_sent += current_batch_size

                phase_elapsed = time.perf_counter() - phase_started
                if phase_elapsed < request_interval:
                    time.sleep(request_interval - phase_elapsed)

        elapsed = time.perf_counter() - started
        avg_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
        total_http_requests = successful_http_requests + failed_http_requests
        return PhaseResult(
            name=name,
            endpoint=endpoint,
            total_samples=total_samples,
            successful_samples=successful_samples,
            failed_samples=failed_samples,
            http_requests_sent=total_http_requests,
            successful_http_requests=successful_http_requests,
            failed_http_requests=failed_http_requests,
            fraud_predictions=fraud_predictions,
            elapsed_seconds=elapsed,
            avg_latency_ms=avg_latency_ms,
        )

    def _resolve_api_base_url(
        self,
        *,
        requested_url: str | None,
        port_candidates: tuple[int, ...],
    ) -> str:
        normalized_url = self._normalize_url(requested_url or "http://127.0.0.1:8000")
        parsed = urlsplit(normalized_url)
        base_candidate = self._base_from_parts(
            scheme=parsed.scheme,
            hostname=parsed.hostname or "127.0.0.1",
            port=parsed.port,
        )
        candidates: list[str] = []

        if parsed.port is None:
            for port in port_candidates:
                candidate = self._base_from_parts(
                    scheme=parsed.scheme,
                    hostname=parsed.hostname or "127.0.0.1",
                    port=port,
                )
                if candidate not in candidates:
                    candidates.append(candidate)
            if base_candidate not in candidates:
                candidates.append(base_candidate)
        else:
            candidates.append(base_candidate)

        errors: list[str] = []
        for candidate in candidates:
            try:
                with httpx.Client(timeout=min(self.timeout_seconds, 5.0)) as client:
                    response = client.get(f"{candidate}/health")
                    response.raise_for_status()
                    return candidate
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")

        raise RuntimeError(
            "Could not resolve a working API base URL from "
            f"{requested_url!r}. Tried: {' | '.join(errors)}"
        )

    def _resolve_prometheus_url(
        self,
        *,
        explicit_url: str | None,
        configured_url: str | None,
        prefer_base_host: bool,
    ) -> str | None:
        if explicit_url:
            return self._strip_trailing_slash(self._normalize_url(explicit_url))
        if configured_url and not prefer_base_host:
            return self._strip_trailing_slash(self._normalize_url(configured_url))

        parsed = urlsplit(self.base_url)
        hostname = parsed.hostname
        if not hostname:
            return None
        return self._base_from_parts(
            scheme=parsed.scheme,
            hostname=hostname,
            port=_PROMETHEUS_PORT,
        )

    @staticmethod
    def _extract_prometheus_value(payload: dict[str, Any]) -> float | None:
        result = payload.get("data", {}).get("result", [])
        if not result:
            return None
        value = result[0].get("value", [])
        if len(value) < 2:
            return None
        try:
            return float(value[1])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_fraud_predictions(
        *,
        response_payload: dict[str, Any],
        mode: str,
    ) -> int:
        if mode == "batch":
            return int(response_payload.get("fraud_detected", 0))
        return 1 if response_payload.get("is_fraud_pred") else 0

    @staticmethod
    def _metric_delta(
        before: dict[str, float],
        after: dict[str, float],
    ) -> dict[str, float]:
        if not before or not after:
            return {}
        keys = set(before) | set(after)
        return {
            key: round(float(after.get(key, 0.0)) - float(before.get(key, 0.0)), 6)
            for key in sorted(keys)
        }

    @staticmethod
    def _normalize_url(value: str) -> str:
        normalized = value.strip()
        if "://" not in normalized:
            normalized = f"http://{normalized}"
        parsed = urlsplit(normalized)
        return urlunsplit((parsed.scheme or "http", parsed.netloc, "", "", ""))

    @staticmethod
    def _base_from_parts(
        *,
        scheme: str,
        hostname: str,
        port: int | None,
    ) -> str:
        netloc = hostname if port is None else f"{hostname}:{port}"
        return FraudPredictionSimulator._strip_trailing_slash(
            urlunsplit((scheme or "http", netloc, "", "", ""))
        )

    @staticmethod
    def _strip_trailing_slash(value: str) -> str:
        return value[:-1] if value.endswith("/") else value
