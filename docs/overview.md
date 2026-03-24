# Project Overview

Tài liệu này tóm tắt nhanh kiến trúc, luồng vận hành và cấu hình triển khai chính của repo `tcb-fraud-detection-mlops`.

## 1. Mục tiêu hệ thống

Repo này triển khai một hệ thống MLOps cho bài toán phát hiện gian lận giao dịch, gồm 3 phần chính:

- `ml_pipeline/`: tiền xử lý, train, evaluate, model registry
- `serving_api/`: FastAPI phục vụ inference và metrics
- `monitoring/`, `dags/`, `docker-compose.yml`: quan sát hệ thống, orchestration và deploy stack

## 2. Thành phần runtime chính

Khi deploy bằng Docker Compose, stack hiện tại gồm:

- `fastapi-stable`: API phục vụ model stable
- `fastapi-candidate`: API phục vụ model candidate
- `loadbalancer`: Nginx điều phối traffic
- `mlflow`: tracking, registry và model stage
- `minio`: artifact store tương thích S3
- `airflow`: chạy DAG retraining
- `prometheus`, `grafana`: monitoring
- `node-exporter`, `cadvisor`, `nginx-exporter`: metrics hạ tầng và container

## 3. Luồng CI/CD hiện tại

Workflow chính: `.github/workflows/ci-cd-pipeline.yml`

### CI

Chạy trên `push` vào `main` và `pull_request`:

- checkout source
- setup Python 3.10
- cài dependency
- chạy lint
- chạy test với coverage gate `>= 80%`
- build và push Docker image của FastAPI lên Docker Hub

### CD

Chỉ chạy khi `push` lên branch deploy:

- validate config deploy
- GitHub Actions SSH vào VPS
- VPS clone/fetch/pull chính repo này
- VPS `docker login` vào Docker Hub
- VPS `docker compose pull`
- tạo `.env` từ `.env.example` nếu chưa có
- chạy `docker compose up -d --no-build`
- chạy health check cho FastAPI, MLflow, Airflow, Grafana

## 4. Các key và config cần có

### GitHub Actions -> VPS

- `SSH_DEPLOY_KEY`:
  private SSH key dùng để GitHub Actions SSH vào VPS
- `DOCKERHUB_TOKEN`:
  Docker Hub access token để push image trong CI và pull image trên VPS
- `GCP_DEPLOY_HOST`:
  IP hoặc domain của VPS
- `GCP_DEPLOY_USER`:
  user SSH trên VPS

Ghi nhớ:

- `SSH_DEPLOY_KEY` phải là private key, thường bắt đầu bằng `-----BEGIN OPENSSH PRIVATE KEY-----`
- public key tương ứng phải nằm trong `~/.ssh/authorized_keys` trên VPS
- image deploy hiện được hardcode là `tungb12ok/tcb-detect-credit`

## 5. Cấu hình môi trường trên VPS

Workflow deploy hiện kỳ vọng VPS đã có:

- `git`
- `curl`
- `docker`
- `docker compose`

Thư mục deploy mặc định:

- `$HOME/tcb-fraud-detection-mlops`

Biến môi trường runtime được đọc từ file `.env` trong thư mục deploy. Nếu file chưa tồn tại, workflow sẽ copy từ `.env.example`, nhưng bạn vẫn phải SSH vào VPS để thay bằng secret thật.

## 6. Tài liệu liên quan

- `docs/cloud_ssh_setup.md`: cách cài SSH key, token và chuẩn bị VPS
- `docs/cicd_proposal.md`: mô tả chi tiết hơn về luồng CI/CD
- `docs/api_docs.md`: tài liệu API

## 7. Trạng thái triển khai nên nhớ

- `SSH_DEPLOY_KEY` phục vụ kết nối SSH tới VPS
- `GCP_DEPLOY_HOST` và `GCP_DEPLOY_USER` có thể để trong `Secrets` hoặc `Variables`
- workflow hiện tự dùng URL public của chính repo đang chạy, không cần cấu hình `GIT_REPO_URL`
