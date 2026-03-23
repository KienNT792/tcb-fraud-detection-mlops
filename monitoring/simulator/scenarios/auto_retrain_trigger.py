from __future__ import annotations

import logging
import os
import time

from monitoring.simulator.common import (
    DEFAULT_THRESHOLD,
    fetch_drift_snapshot,
    fetch_rollout_state,
    log_rollout_state,
    manual_progressive_rollout,
    manual_train_and_stage_candidate,
    wait_for_http_ready,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _should_trigger(
    snapshot: dict[str, object],
    *,
    threshold: float,
    force: bool,
) -> bool:
    if force or threshold <= 0:
        return True
    if not snapshot.get("ready"):
        return False
    return float(snapshot.get("drift_ratio", 0.0)) >= threshold


def main() -> None:
    wait_for_http_ready("http://127.0.0.1:8000/health")

    threshold = float(
        os.getenv("AUTO_RETRAIN_THRESHOLD", str(DEFAULT_THRESHOLD))
    )
    poll_seconds = float(os.getenv("AUTO_RETRAIN_POLL_SECONDS", "20"))
    cooldown_seconds = float(os.getenv("AUTO_RETRAIN_COOLDOWN_SECONDS", "300"))
    step_percentage = int(
        os.getenv(
            "AUTO_RETRAIN_STEP_PERCENTAGE",
            os.getenv("SIM_FULL_CYCLE_STEP_PERCENTAGE", "10"),
        )
    )
    step_wait_seconds = int(
        os.getenv(
            "AUTO_RETRAIN_STEP_SECONDS",
            os.getenv("SIM_FULL_CYCLE_STEP_SECONDS", "30"),
        )
    )
    initial_percentage = int(os.getenv("SIM_INITIAL_CANARY_PERCENTAGE", "10"))
    require_training = os.getenv("SIM_REQUIRE_TRAIN", "false").lower() == "true"
    allow_preprocess = os.getenv("SIM_AUTO_PREPROCESS", "true").lower() == "true"
    auto_promote = os.getenv("AUTO_RETRAIN_AUTO_PROMOTE", "true").lower() == "true"
    run_once = os.getenv("AUTO_RETRAIN_RUN_ONCE", "false").lower() == "true"
    force = os.getenv("AUTO_RETRAIN_FORCE", "false").lower() == "true"

    last_cycle_started_at = 0.0

    while True:
        snapshot = fetch_drift_snapshot()
        state = fetch_rollout_state()
        candidate_percentage = int(state["rollout"]["candidate_percentage"])
        drift_ratio = float(snapshot.get("drift_ratio", 0.0))
        ready = bool(snapshot.get("ready"))

        logger.info(
            "Auto retrain poll | ready=%s | drift_ratio=%.6f | threshold=%.3f | canary=%s%%",
            ready,
            drift_ratio,
            threshold,
            candidate_percentage,
        )

        can_start_cycle = (
            candidate_percentage == 0
            and (time.time() - last_cycle_started_at) >= cooldown_seconds
            and _should_trigger(snapshot, threshold=threshold, force=force)
        )

        if can_start_cycle:
            last_cycle_started_at = time.time()
            manual_train_and_stage_candidate(
                initial_percentage=initial_percentage,
                require_training=require_training,
                allow_preprocess=allow_preprocess,
            )
            log_rollout_state()
            manual_progressive_rollout(
                step_percentage=step_percentage,
                step_wait_seconds=step_wait_seconds,
                auto_promote=auto_promote,
            )
            log_rollout_state()

        if run_once:
            return

        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
