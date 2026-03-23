from __future__ import annotations

import os

from monitoring.simulator.common import baseline_payload, run_fixed_rate, wait_for_http_ready


def main() -> None:
    wait_for_http_ready("http://127.0.0.1:8000/health")
    run_fixed_rate(
        name="baseline_5rps",
        payload_builder=baseline_payload,
        duration_seconds=int(os.getenv("SIM_BASELINE_DURATION_SECONDS", "120")),
        rps=int(os.getenv("SIM_BASELINE_RPS", "5")),
    )


if __name__ == "__main__":
    main()
