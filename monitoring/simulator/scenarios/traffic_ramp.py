from __future__ import annotations

import os

from monitoring.simulator.common import baseline_payload, run_phase_plan, wait_for_http_ready


def main() -> None:
    wait_for_http_ready("http://127.0.0.1:8000/health")
    base_duration = int(os.getenv("SIM_RAMP_PHASE_SECONDS", "60"))
    phases = [
        ("warmup", base_duration, 5, baseline_payload),
        ("normal", base_duration, 10, baseline_payload),
        ("busy_hour", base_duration, 20, baseline_payload),
        ("peak", base_duration, 35, baseline_payload),
    ]
    run_phase_plan(name="traffic_ramp", phases=phases)


if __name__ == "__main__":
    main()
