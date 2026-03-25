from __future__ import annotations

import argparse
import json

from monitoring.simulator.simulator import FraudPredictionSimulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fraud prediction simulator using data-generation dataset.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        choices=["normal", "slight_drift", "moderate_drift", "severe_drift"],
        help="Simulation scenario.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=None,
        help="Number of API requests to send.",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=None,
        help="Requests per second.",
    )
    parser.add_argument(
        "--check-drift",
        action="store_true",
        help="Fetch and print /monitoring/drift after simulation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    simulator = FraudPredictionSimulator()
    simulator.check_health()
    result = simulator.run(
        scenario=args.scenario,
        requests=args.requests,
        rps=args.rps,
    )
    payload = {
        "scenario": result.scenario,
        "total_requests": result.total_requests,
        "success_requests": result.success_requests,
        "failed_requests": result.failed_requests,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "avg_latency_ms": round(result.avg_latency_ms, 2),
    }
    if args.check_drift:
        payload["drift"] = simulator.fetch_drift_status()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
