from __future__ import annotations

import os

from monitoring.simulator.common import baseline_payload, run_fixed_rate, wait_for_http_ready


def main() -> None:
    wait_for_http_ready("http://127.0.0.1:8000/health")
    run_fixed_rate(
        name="high_volume_burst",
        payload_builder=baseline_payload,
        duration_seconds=int(os.getenv("SIM_BURST_DURATION_SECONDS", "90")),
        rps=int(os.getenv("SIM_BURST_RPS", "40")),
    )


if __name__ == "__main__":
    main()
