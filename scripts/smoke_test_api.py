from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


DEFAULT_PAYLOAD = {
    "transaction_id": "TX_SMOKE_001",
    "timestamp": "2026-03-14 10:23:00",
    "customer_id": "CUST_001",
    "amount": 350000,
    "customer_tier": "PRIORITY",
    "card_type": "VISA",
    "card_tier": "GOLD",
    "currency": "VND",
    "merchant_name": "Grab",
    "mcc_code": 4121,
    "merchant_category": "Transport",
    "merchant_city": "Ha Noi",
    "merchant_country": "VN",
    "device_type": "Mobile",
    "os": "iOS",
    "ip_country": "VN",
    "distance_from_home_km": 2.5,
    "cvv_match": "Y",
    "is_3d_secure": "Y",
    "transaction_status": "APPROVED",
    "tx_count_last_1h": 1,
    "tx_count_last_24h": 3,
    "time_since_last_tx_min": 120.0,
    "avg_amount_last_30d": 400000,
    "amount_ratio_vs_avg": 0.875,
    "is_new_device": 0,
    "is_new_merchant": 0,
    "card_bin": 411111,
    "account_age_days": 730,
    "is_weekend": 0,
    "hour_of_day": 10,
}


def wait_until_ready(url: str, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/ready", timeout=5) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError:
            time.sleep(2)
    raise TimeoutError(f"API did not become ready within {timeout}s")


def call_predict(url: str, api_key: str | None) -> dict:
    data = json.dumps(DEFAULT_PAYLOAD).encode("utf-8")
    request = urllib.request.Request(
        f"{url}/predict",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if api_key:
        request.add_header("x-api-key", api_key)

    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the fraud detection API.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--api-key", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wait_until_ready(args.url.rstrip("/"), args.timeout)
    result = call_predict(args.url.rstrip("/"), args.api_key)

    required_fields = {
        "transaction_id",
        "fraud_score",
        "is_fraud_pred",
        "threshold",
        "risk_level",
        "model_version",
        "prediction_timestamp",
    }
    missing = required_fields - set(result.keys())
    if missing:
        raise SystemExit(f"Smoke test failed. Missing fields: {sorted(missing)}")

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
