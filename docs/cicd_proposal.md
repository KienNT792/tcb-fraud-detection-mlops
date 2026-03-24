# Đề Xuất Luồng CI/CD Cho TCB Fraud Detection

Tài liệu mô tả luồng CI/CD sử dụng **GitHub Actions** để đóng gói, triển khai và vận hành hệ thống MLOps trên Google Cloud VPS.

## 1. Thành phần chính

- **GitHub Actions**: điều phối CI/CD, trigger tự động khi push/PR.
- **Airflow**: lập lịch pipeline `check_model_quality >> preprocess >> train >> evaluate >> stage_candidate >> verify_candidate`.
- **MLflow**: quản lý experiment, metric, artifact và version model.
- **MinIO**: object storage nội bộ cho artifact, model và dữ liệu đã xử lý.
- **FastAPI**: phục vụ model qua API `/predict`, `/predict/batch`, `/health`, `/metrics`.
- **Prometheus**: thu thập metric hệ thống và metric ứng dụng.
- **Grafana**: dashboard cho API, container và VPS.
- **Docker Compose**: khởi tạo toàn bộ stack trên cùng một Docker network trên VPS.

## 2. Luồng CI

Khi developer push code hoặc tạo PR:

1. GitHub Actions checkout source code.
2. Setup Python 3.10, cài dependency cho `ml_pipeline` và `serving_api`.
3. Chạy lint với `flake8`.
4. Chạy test với coverage gate `--cov-fail-under=80`:
   - `ml_pipeline/tests/test_preprocess.py` (preprocessing)
   - `ml_pipeline/tests/test_model.py` (train + evaluate + inference)
   - `serving_api/tests/` (API endpoints + model loader)
5. Build và push Docker image `tungb12ok/tcb-detect-credit:<sha>` lên Docker Hub.

## 3. Luồng CD

Sau khi CI thành công (chỉ trên branch `main` hoặc `dev/ver2`):

1. GitHub Actions SSH vào Google Cloud VPS.
2. VPS cập nhật source code từ branch đích.
3. VPS `docker login`, `docker compose pull`, rồi `docker compose up -d --no-build` để cập nhật các service:
   - FastAPI Stable + Candidate
   - Nginx Load Balancer
   - MLflow server
   - MinIO
   - Airflow
   - Prometheus + Grafana
4. Health check sau deploy với các endpoint:
   - `http://localhost:${FASTAPI_PORT}/health`
   - `http://localhost:${MLFLOW_PORT}`
   - `http://localhost:${AIRFLOW_PORT}/health`
   - `http://localhost:${GRAFANA_PORT}/api/health`

## 4. Luồng MLOps Sau Triển Khai

1. Airflow định kỳ kích hoạt DAG huấn luyện (02:00 UTC hàng ngày).
2. Pipeline training ghi metric và artifact vào MLflow.
3. MLflow sử dụng MinIO làm artifact store thông qua S3-compatible endpoint.
4. Nếu model mới PASS evaluation → `stage_candidate` copy artifacts → `verify_candidate` poll health.
5. Candidate FastAPI auto-reload model qua manifest-based hot-reload.
6. Prometheus scrape:
   - FastAPI `/metrics`
   - `node-exporter` cho CPU, RAM, disk trên VPS
   - `cadvisor` cho metric container
   - `nginx-exporter` cho loadbalancer metrics
7. Grafana đọc Prometheus datasource để hiển thị dashboard vận hành và monitoring model.

## 5. Mapping Với Repo Hiện Tại

- `.github/workflows/ci-cd-pipeline.yml`: pipeline CI/CD chính.
- `docker-compose.yml`: stack deployment trên VPS.
- `serving_api/Dockerfile`: Docker build đa giai đoạn từ repo root.
- `serving_api/app/main.py`: bổ sung `/metrics` cho Prometheus.
- `dags/fraud_pipeline.py`: DAG retraining định kỳ + canary staging.
- `monitoring/prometheus/prometheus.yml`: job scrape cho API, node-exporter và cadvisor.
- `monitoring/grafana/provisioning/*`: cấu hình datasource/dashboard cho Grafana.

## 6. Biến Môi Trường Cần Có Trên VPS

Tạo file `.env` từ `.env.example` và cập nhật các giá trị tối thiểu:

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

GitHub Actions cần cấu hình Secrets:

- `GCP_DEPLOY_HOST` — IP của GCP VPS, có thể để ở Secrets hoặc Variables
- `GCP_DEPLOY_USER` — SSH username (vd: `ubuntu`), có thể để ở Secrets hoặc Variables
- `DEPLOY_PATH` — thư mục deploy trên VPS, có thể để ở Secrets hoặc Variables
- `SSH_DEPLOY_KEY` — SSH private key dùng để Actions SSH vào VPS
- `DOCKERHUB_TOKEN` — Docker Hub access token dùng để push image và pull trên VPS

Tham khảo thêm hướng dẫn chuẩn bị VPS và cài public key tại `docs/cloud_ssh_setup.md`.
