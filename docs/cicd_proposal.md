# De Xuat Luong CI/CD Cho TCB Fraud Detection

Tai lieu nay mo ta luong CI/CD duoc de xuat cho branch `dev/ver2` de dong goi, trien khai va van hanh dong bo he thong MLOps tren Google Cloud VPS.

## 1. Thanh phan chinh

- Jenkins: dieu phoi CI/CD, nhan webhook tu GitHub va chay `Jenkinsfile`.
- Airflow: lap lich pipeline `preprocess -> train -> evaluate` qua DAG `dags/fraud_pipeline.py`.
- MLflow: quan ly experiment, metric, artifact va version model.
- MinIO: object storage noi bo cho artifact, model va du lieu da xu ly.
- FastAPI: phuc vu model qua API `/predict`, `/predict/batch`, `/health`, `/metrics`.
- Prometheus: thu thap metric he thong va metric ung dung.
- Grafana: dashboard cho API, container va VPS.
- Docker Compose: khoi tao toan bo stack tren cung mot Docker network tren VPS.

## 2. Luong CI

Khi developer push code len GitHub:

1. Jenkins nhan webhook va checkout source code.
2. Jenkins tao virtualenv, cai dependency cho `ml_pipeline` va `serving_api`.
3. Jenkins chay lint voi `flake8`.
4. Jenkins chay test:
   - `ml_pipeline/tests/test_preprocess.py`
   - `serving_api/tests/`
5. Neu test dat coverage yeu cau, Jenkins build image `tcb-fraud-fastapi:<branch-build>`.
6. Jenkins luu `build/image-tag.txt` lam artifact cho buoc deploy.

## 3. Luong CD

Sau khi CI thanh cong:

1. Jenkins SSH vao Google Cloud VPS.
2. VPS cap nhat source code tu branch dich (`main` hoac `dev/ver2`).
3. Jenkins chay `docker compose up -d --build` de cap nhat cac service:
   - FastAPI serving
   - MLflow server
   - MinIO
   - Airflow
   - Prometheus
   - Grafana
4. Jenkins chay health check sau deploy voi cac endpoint:
   - `http://localhost:${FASTAPI_PORT}/health`
   - `http://localhost:${MLFLOW_PORT}`
   - `http://localhost:${AIRFLOW_PORT}/health`
   - `http://localhost:${GRAFANA_PORT}/api/health`

## 4. Luong MLOps Sau Trien Khai

1. Airflow dinh ky kich hoat DAG huan luyen.
2. Pipeline training ghi metric va artifact vao MLflow.
3. MLflow su dung MinIO lam artifact store thong qua S3-compatible endpoint.
4. Model va artifact moi duoc dong bo vao thu muc `models/` va `data/processed/` de FastAPI co the phuc vu ngay.
5. Prometheus scrape:
   - FastAPI `/metrics`
   - `node-exporter` cho CPU, RAM, disk tren VPS
   - `cadvisor` cho metric container
6. Grafana doc Prometheus datasource de hien thi dashboard van hanh va monitoring model.

## 5. Mapping Voi Repo Hien Tai

- `Jenkinsfile`: mo ta pipeline CI/CD chinh.
- `docker-compose.yml`: stack deployment tren VPS.
- `serving_api/Dockerfile`: Docker build da duoc sua de build dung tu repo root.
- `serving_api/app/main.py`: bo sung `/metrics` cho Prometheus.
- `dags/fraud_pipeline.py`: DAG retraining dinh ky.
- `monitoring/prometheus/prometheus.yml`: job scrape cho API, node-exporter va cadvisor.
- `monitoring/grafana/provisioning/*`: cau hinh datasource/dashboard cho Grafana.

## 6. Bien Moi Truong Can Co Tren VPS

Tao file `.env` tu `.env.example` va cap nhat cac gia tri toi thieu:

- `MINIO_ROOT_USER`
- `MINIO_ROOT_PASSWORD`
- `MINIO_BUCKET`
- `GRAFANA_ADMIN_USER`
- `GRAFANA_ADMIN_PASSWORD`
- `AIRFLOW_UID`
- `IMAGE_TAG`
- `FASTAPI_PORT`
- `MINIO_API_PORT`
- `MINIO_CONSOLE_PORT`
- `MLFLOW_PORT`
- `AIRFLOW_PORT`
- `PROMETHEUS_PORT`
- `GRAFANA_PORT`
- `CADVISOR_PORT`

Ngoai ra Jenkins can duoc cau hinh them:

- Credential `gcp-vps-ssh`
- Env `DEPLOY_USER`
- Env `DEPLOY_HOST`
- Git webhook tu GitHub sang Jenkins

## 7. Ghi Chu Trien Khai

- Compose hien dung mot VPS va mot Docker network, phu hop giai doan demo hoac pilot.
- Neu chuyen sang production tai luong cao, nen tach artifact store, metadata database va scheduler thanh cac thanh phan rieng.
- Can pin version image va bo sung secret manager truoc khi dua vao moi truong production that su.
