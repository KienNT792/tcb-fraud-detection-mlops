from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from monitoring.simulator.simulator import FraudPredictionSimulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run fraud prediction traffic against the serving API and capture "
            "monitoring snapshots before and after the run."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Serving API base URL. If a host is passed without a port, the "
            "simulator will probe common ports and prefer :8000."
        ),
    )
    parser.add_argument(
        "--prometheus-url",
        default=None,
        help="Prometheus base URL. Defaults to the same host on port 9090.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Scenario name from sim_config.yaml (normal, slight_drift, moderate_drift, severe_drift).",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=None,
        help="Number of transactions in the measured phase.",
    )
    parser.add_argument(
        "--warmup-samples",
        type=int,
        default=None,
        help=(
            "Number of baseline transactions sent first to warm up prediction "
            "metrics and drift reference. Use >= 100 when drift is still missing."
        ),
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=None,
        help="HTTP requests per second. In batch mode, each request can carry many transactions.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Transactions per /predict/batch request. Ignored in single mode.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "batch"],
        default=None,
        help="Use /predict or /predict/batch.",
    )
    parser.add_argument(
        "--check-drift",
        action="store_true",
        help="Include drift snapshots before and after the run.",
    )
    parser.add_argument(
        "--check-metrics",
        action="store_true",
        help="Include API /metrics snapshots before and after the run.",
    )
    parser.add_argument(
        "--check-prometheus",
        action="store_true",
        help="Query Prometheus after the run for the dashboard metrics.",
    )
    parser.add_argument(
        "--prometheus-wait-seconds",
        type=float,
        default=None,
        help="Seconds to wait before querying Prometheus after the run.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    simulator = FraudPredictionSimulator(
        base_url=args.base_url,
        prometheus_url=args.prometheus_url,
    )
    result = simulator.run(
        scenario=args.scenario,
        requests=args.requests,
        rps=args.rps,
        batch_size=args.batch_size,
        warmup_samples=args.warmup_samples,
        mode=args.mode,
        capture_metrics=args.check_metrics,
        capture_drift=args.check_drift,
        capture_prometheus=args.check_prometheus,
        prometheus_wait_seconds=args.prometheus_wait_seconds,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
