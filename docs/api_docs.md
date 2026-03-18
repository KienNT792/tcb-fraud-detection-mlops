# API Docs - TCB Fraud Detection

## Base endpoints

- `GET /`
- `GET /health`
- `GET /live`
- `GET /ready`
- `GET /metrics`
- `POST /predict`
- `POST /predict/batch`
- `GET /admin/rollout`
- `POST /admin/rollout/config`
- `POST /admin/rollout/advance`
- `POST /admin/rollout/pause`
- `POST /admin/rollout/promote`
- `POST /admin/rollout/rollback`
- `GET /admin/rollout/ui`

## Authentication

Neu bien moi truong `API_KEY` duoc set, client phai gui header:

```http
x-api-key: <your-api-key>
```

Neu `API_KEY` rong, API se cho phep goi prediction ma khong can auth. Cach nay giup local demo de dang hon, trong khi van ho tro basic production hardening.

## Example request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TX_001",
    "timestamp": "2026-03-14 10:23:00",
    "customer_id": "CUST_12345",
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
    "hour_of_day": 10
  }'
```

## Example response

```json
{
  "transaction_id": "TX_001",
  "fraud_score": 0.0231,
  "is_fraud_pred": false,
  "threshold": 0.6788,
  "risk_level": "LOW",
  "model_version": "local-demo",
  "prediction_timestamp": "2026-03-14T10:23:01+00:00",
  "served_by": "stable",
  "rollout_candidate_percent": 10
}
```

## Monitoring

- `GET /metrics` expose Prometheus metrics cho request rate, latency va prediction distribution.
- `GET /live` duoc dung cho liveness probe.
- `GET /ready` duoc dung cho readiness probe.
- `GET /health` tra ve metadata cua model dang load.

## Canary rollout

- Admin UI: `GET /admin/rollout/ui`
- Admin JSON status: `GET /admin/rollout`
- Cau hinh rollout cho model candidate duoc luu qua `ROLLOUT_CONFIG_PATH`
- Support schedule auto-advance bang `ROLLOUT_AUTO_ADVANCE=true` va `ROLLOUT_STEP_INTERVAL_MINUTES`
- Batch response tra them `traffic_distribution`, con tung item se co `served_by`, `threshold`, `model_version`
- Khi deploy nhieu replica tren Kubernetes, `ROLLOUT_CONFIG_PATH` va candidate model artifact nen duoc mount tu shared volume/object storage. Neu moi pod giu file local rieng, rollout state se khong dong bo giua cac replica.

## Streaming

Kafka topics de xuat:

- input topic: `transactions.raw`
- output topic: `transactions.scored`

Demo scripts:

- `python -m streaming.producer --count 10`
- `python -m streaming.scoring_consumer --once`
