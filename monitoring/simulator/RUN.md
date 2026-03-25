# Simulator Run Guide

## Why the dashboard was showing `0`

The old simulator had two problems:

1. It depended on `data-generation/tcb_credit_fraud_dataset.csv`, but that file is not present in this repo.
2. It only sent a small number of direct `/predict` calls, so Prometheus and the drift monitor often stayed near zero.

The rewritten simulator fixes that by:

- auto-detecting the API base URL and preferring `:8000` when you pass only a host;
- using the available `modelv2/raw/synthetic_credit_fraud_v2.csv` dataset when present;
- falling back to synthetic payloads if no CSV exists;
- warming up the API with at least `120` samples before the measured run;
- supporting `/predict/batch` so metrics rise quickly.

## Quick run

From the repo root:

```bash
./scripts/run_simulator.sh
```

That targets local defaults:

- API: `http://127.0.0.1:8000`
- Prometheus: `http://127.0.0.1:9090`
- Scenario: `moderate_drift`
- Warmup: `120` samples
- Measured phase: `300` samples

## Run against the VPS

The live API is exposed on port `8000`, and Prometheus is on `9090`.

```bash
./scripts/run_simulator.sh \
  --base-url http://35.222.198.13/
```

If you omit `--prometheus-url`, the simulator will derive `http://35.222.198.13:9090` automatically from the same host.

## Run on the VPS host for Grafana/Prometheus validation

If you SSH into the VPS and want the Grafana panels fed by the exact target that
Prometheus scrapes, run against the stable FastAPI port on localhost instead of
the public load balancer:

```bash
SIMULATOR_BASE_URL=http://127.0.0.1:8002 \
SIMULATOR_PROMETHEUS_URL=http://127.0.0.1:9090 \
./scripts/run_simulator.sh
```

Use this mode when:

- `/metrics` on the public URL changes, but Prometheus queries still stay at `0`
- you want Grafana panels to move immediately on the same `fastapi-stable` job

You can also use environment variables:

```bash
SIMULATOR_BASE_URL=http://35.222.198.13/ \
SIMULATOR_PROMETHEUS_URL=http://35.222.198.13:9090 \
./scripts/run_simulator.sh
```

## Useful overrides

Run a lighter baseline:

```bash
./scripts/run_simulator.sh \
  --scenario normal \
  --warmup-samples 100 \
  --requests 120 \
  --batch-size 10 \
  --rps 2
```

Run heavier drift:

```bash
./scripts/run_simulator.sh \
  --base-url http://35.222.198.13/ \
  --prometheus-url http://35.222.198.13:9090 \
  --scenario severe_drift \
  --warmup-samples 120 \
  --requests 400 \
  --batch-size 20 \
  --rps 3
```

Use direct single-request mode instead of `/predict/batch`:

```bash
./scripts/run_simulator.sh \
  --mode single \
  --batch-size 1 \
  --requests 80
```

## What to expect in the output

The JSON result includes:

- `health`: current model health from `/health`
- `warmup`: summary of the warmup phase
- `metrics_before`, `metrics_after`, `metrics_delta`: snapshots from `/metrics`
- `drift_before`, `drift_after`: snapshots from `/monitoring/drift`
- `prometheus`: dashboard-facing Prometheus queries after the run

## Drift note

If drift is still `0`, check these two conditions:

1. `warmup_samples` must be at least `100`, because the API uses a live warmup window when no reference parquet is deployed.
2. Prometheus needs a scrape cycle before dashboard panels reflect the new samples, so the simulator waits before querying Prometheus.
