from __future__ import annotations

import os
from pprint import pprint

from monitoring.simulator.common import manual_train_and_stage_candidate, wait_for_http_ready


def main() -> None:
    wait_for_http_ready("http://127.0.0.1:8002/health")
    require_training = os.getenv("SIM_REQUIRE_TRAIN", "false").lower() == "true"
    allow_preprocess = os.getenv("SIM_AUTO_PREPROCESS", "true").lower() == "true"
    state = manual_train_and_stage_candidate(
        initial_percentage=int(os.getenv("SIM_INITIAL_CANARY_PERCENTAGE", "10")),
        require_training=require_training,
        allow_preprocess=allow_preprocess,
    )
    pprint(state)


if __name__ == "__main__":
    main()
