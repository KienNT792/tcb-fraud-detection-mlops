# TCB Fraud Detection MLOps

Tai lieu hien co trong repo:

- Tong quan va hien trang project: `.README.md`
- Requirements mo rong va roadmap: `docs/extended_requirements.md`
- API docs: `docs/api_docs.md`

Trang thai hien tai:

- Da co core ML pipeline va FastAPI serving
- Da bo sung Docker, docker-compose, Airflow DAG, MLflow flow, Prometheus/Grafana, Kafka demo, Kubernetes manifests va GitHub Actions baseline
- Phan deploy cloud se bo sung va tinh chinh sau khi co thong tin VPS Google Cloud

Quickstart local:

1. Cai dependencies theo `ml_pipeline/requirements.txt` va `serving_api/requirements.txt`
2. Chay `python scripts/bootstrap_demo_artifacts.py --if-missing`
3. Chay `make run-api` hoac `make compose-up`
4. Kiem tra `http://localhost:8000/docs`, `http://localhost:8000/metrics`, `http://localhost:8080` (Airflow), `http://localhost:5000` (MLflow), `http://localhost:9090` (Prometheus), `http://localhost:3000` (Grafana)
