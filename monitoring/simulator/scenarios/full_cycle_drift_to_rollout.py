from __future__ import annotations

import os
import threading

from monitoring.simulator.common import (
    DEFAULT_THRESHOLD,
    baseline_payload,
    drift_payload,
    fetch_drift_snapshot,
    log_rollout_state,
    manual_progressive_rollout,
    manual_train_and_stage_candidate,
    rollout_duration_seconds,
    run_fixed_rate,
    wait_for_drift_threshold,
    wait_for_http_ready,
)


def main() -> None:
    wait_for_http_ready("http://127.0.0.1:8000/health")

    warmup_seconds = int(os.getenv("SIM_FULL_CYCLE_WARMUP_SECONDS", "60"))
    drift_rps = int(os.getenv("SIM_FULL_CYCLE_DRIFT_RPS", "10"))
    drift_threshold = float(
        os.getenv("SIM_FULL_CYCLE_DRIFT_THRESHOLD", str(DEFAULT_THRESHOLD))
    )
    drift_window_seconds = int(
        os.getenv("SIM_FULL_CYCLE_DRIFT_WINDOW_SECONDS", "180")
    )
    initial_percentage = int(
        os.getenv("SIM_INITIAL_CANARY_PERCENTAGE", "10")
    )
    step_percentage = int(os.getenv("SIM_FULL_CYCLE_STEP_PERCENTAGE", "10"))
    step_wait_seconds = int(os.getenv("SIM_FULL_CYCLE_STEP_SECONDS", "30"))
    rollout_rps = int(os.getenv("SIM_FULL_CYCLE_ROLLOUT_RPS", "5"))
    post_promotion_soak_seconds = int(
        os.getenv("SIM_FULL_CYCLE_POST_PROMOTION_SOAK_SECONDS", "30")
    )
    skip_drift_wait = (
        os.getenv("SIM_FULL_CYCLE_SKIP_DRIFT_WAIT", "false").lower() == "true"
    )
    require_training = os.getenv("SIM_REQUIRE_TRAIN", "false").lower() == "true"
    allow_preprocess = os.getenv("SIM_AUTO_PREPROCESS", "true").lower() == "true"

    run_fixed_rate(
        name="full_cycle:warmup",
        payload_builder=baseline_payload,
        duration_seconds=warmup_seconds,
        rps=5,
    )
    run_fixed_rate(
        name="full_cycle:drift_injection",
        payload_builder=drift_payload,
        duration_seconds=drift_window_seconds,
        rps=drift_rps,
    )
    if skip_drift_wait or drift_threshold <= 0:
        snapshot = fetch_drift_snapshot()
    else:
        snapshot = wait_for_drift_threshold(
            threshold=drift_threshold,
            max_wait_seconds=max(drift_window_seconds, 60),
            poll_seconds=5,
        )
    print({"drift_snapshot": snapshot})

    manual_train_and_stage_candidate(
        initial_percentage=initial_percentage,
        require_training=require_training,
        allow_preprocess=allow_preprocess,
    )
    log_rollout_state()

    traffic_duration = rollout_duration_seconds(
        initial_percentage=initial_percentage,
        step_percentage=step_percentage,
        step_wait_seconds=step_wait_seconds,
    ) + post_promotion_soak_seconds

    traffic_thread = threading.Thread(
        target=run_fixed_rate,
        kwargs={
            "name": "full_cycle:rollout_traffic",
            "payload_builder": baseline_payload,
            "duration_seconds": traffic_duration,
            "rps": rollout_rps,
            "log_rollout_every": max(rollout_rps * 10, 10),
        },
        daemon=True,
    )
    traffic_thread.start()
    manual_progressive_rollout(
        step_percentage=step_percentage,
        step_wait_seconds=step_wait_seconds,
        auto_promote=True,
    )
    traffic_thread.join()
    log_rollout_state()


if __name__ == "__main__":
    main()
