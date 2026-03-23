from __future__ import annotations

import os

from monitoring.simulator.common import (
    DEFAULT_THRESHOLD,
    drift_payload,
    fetch_drift_snapshot,
    run_fixed_rate,
    wait_for_http_ready,
)


def main() -> None:
    wait_for_http_ready("http://127.0.0.1:8000/health")
    result = run_fixed_rate(
        name="feature_drift_5rps",
        payload_builder=drift_payload,
        duration_seconds=int(os.getenv("SIM_DRIFT_DURATION_SECONDS", "180")),
        rps=int(os.getenv("SIM_DRIFT_RPS", "5")),
    )
    snapshot = fetch_drift_snapshot()
    print(
        {
            "scenario": result.name,
            "drift_ratio": snapshot.get("drift_ratio"),
            "overall_score": snapshot.get("overall_score"),
            "threshold": float(os.getenv("SIM_DRIFT_THRESHOLD", str(DEFAULT_THRESHOLD))),
            "current_samples": snapshot.get("current_samples"),
            "features_alerting": snapshot.get("features_alerting"),
        }
    )


if __name__ == "__main__":
    main()
