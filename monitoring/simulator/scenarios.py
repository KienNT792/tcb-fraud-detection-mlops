from __future__ import annotations

from monitoring.simulator.simulator import FraudPredictionSimulator


def main() -> None:
    simulator = FraudPredictionSimulator()
    simulator.check_health()

    test_plan = [
        ("normal", 80, 2.0),
        ("slight_drift", 80, 2.0),
        ("moderate_drift", 100, 3.0),
        ("severe_drift", 120, 4.0),
    ]
    for name, requests, rps in test_plan:
        result = simulator.run(scenario=name, requests=requests, rps=rps)
        print(
            {
                "scenario": result.scenario,
                "total": result.total_requests,
                "success": result.success_requests,
                "failed": result.failed_requests,
                "avg_latency_ms": round(result.avg_latency_ms, 2),
            }
        )

    print({"drift": simulator.fetch_drift_status()})


if __name__ == "__main__":
    main()
