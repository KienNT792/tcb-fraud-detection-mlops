from __future__ import annotations

import os

from monitoring.simulator.common import (
    CANDIDATE_URL,
    baseline_payload,
    run_fixed_rate,
    wait_for_http_ready,
)


def main() -> None:
    wait_for_http_ready(f"{CANDIDATE_URL}/health")
    run_fixed_rate(
        name="candidate_soak_test",
        payload_builder=baseline_payload,
        duration_seconds=int(os.getenv("SIM_CANDIDATE_SOAK_SECONDS", "120")),
        rps=int(os.getenv("SIM_CANDIDATE_SOAK_RPS", "5")),
        target_url=CANDIDATE_URL,
    )


if __name__ == "__main__":
    main()
