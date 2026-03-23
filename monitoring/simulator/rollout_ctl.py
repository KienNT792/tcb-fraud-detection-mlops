from __future__ import annotations

import argparse
import json
import logging

from monitoring.simulator.common import (
    fetch_rollout_state,
    promote_candidate_to_stable,
    set_canary_percentage,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Control canary rollout and rollback.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    set_cmd = subparsers.add_parser("set-canary", help="Set candidate traffic percentage.")
    set_cmd.add_argument("--percent", type=int, required=True, help="Candidate traffic percent (0-100).")

    subparsers.add_parser("rollback", help="Rollback all traffic to stable (0% candidate).")
    subparsers.add_parser("promote", help="Promote candidate to stable and reset canary to 0%.")
    subparsers.add_parser("status", help="Print rollout state.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "set-canary":
        state = set_canary_percentage(args.percent, reload=True)
    elif args.command == "rollback":
        state = set_canary_percentage(0, reload=True)
    elif args.command == "promote":
        state = promote_candidate_to_stable()
    else:
        state = fetch_rollout_state()

    print(json.dumps(state, indent=2))
    logger.info("Rollout command applied: %s", args.command)


if __name__ == "__main__":
    main()
