# Fraud Simulator

Simulator nay mo phong traffic prediction cho du an fraud detection va su dung du lieu tu thu muc `data-generation`.

## Nguon du lieu

- Mac dinh doc CSV: `data-generation/tcb_credit_fraud_dataset.csv`
- Co the bat `regenerate_before_run: true` trong `sim_config.yaml` de thu generate lai du lieu bang Java (best effort).

## Lenh nhanh

```bash
# Chay scenario mac dinh
python -m monitoring.simulator.run_simulation

# Chay moderate drift, 120 requests, 3 rps
python -m monitoring.simulator.run_simulation --scenario moderate_drift --requests 120 --rps 3

# Chay va in drift status
python -m monitoring.simulator.run_simulation --scenario severe_drift --check-drift

# Chay bo scenario mau
python -m monitoring.simulator.scenarios
```

## Scenario

- `normal`: replay data gan voi phan phoi goc.
- `slight_drift`: drift nhe ve amount + hanh vi.
- `moderate_drift`: drift vua.
- `severe_drift`: drift manh de test alert.

## Yeu cau

- API dang chay tai `http://127.0.0.1:8000`
- Endpoint bat buoc: `/health`, `/predict`, `/monitoring/drift`
