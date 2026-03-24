# Cloud SSH Setup For GitHub Actions Deploy

Tài liệu này mô tả phần chuẩn bị trên VPS để workflow [`ci-cd-pipeline.yml`](../.github/workflows/ci-cd-pipeline.yml) có thể deploy bằng `SSH_DEPLOY_KEY`.

## 1. Secrets cần có trên GitHub Actions

- `GCP_DEPLOY_HOST`: IP hoặc DNS của VPS
- `GCP_DEPLOY_USER`: user dùng để SSH vào VPS
- `SSH_DEPLOY_KEY`: private key dùng cho GitHub Actions SSH vào VPS
- `GIT_REPO_URL`: URL clone repo trên VPS

## 2. Phân biệt public key và private key

Nếu giá trị của bạn bắt đầu bằng `ssh-ed25519 AAAA...` thì đó là public key, không phải private key.

- `SSH_DEPLOY_KEY` trên GitHub Actions phải là private key, thường bắt đầu bằng `-----BEGIN OPENSSH PRIVATE KEY-----`
- Public key tương ứng mới là thứ cần thêm vào file `~/.ssh/authorized_keys` trên VPS

## 3. Cài đặt tối thiểu trên VPS

Ví dụ cho Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates docker.io docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
sudo mkdir -p /opt/tcb-fraud-detection-mlops
sudo chown -R "$USER":"$USER" /opt/tcb-fraud-detection-mlops
```

Đăng xuất và đăng nhập lại sau khi thêm user vào group `docker`, rồi kiểm tra:

```bash
docker --version
docker compose version
git --version
curl --version
```

## 4. Cài public key lên VPS

Thêm public key tương ứng với `SSH_DEPLOY_KEY` vào VPS:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
echo 'ssh-ed25519 AAAAC3Nz...' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Nếu bạn đang dùng đúng public key mà bạn gửi (`ssh-ed25519 AAAAC3Nz...`) thì key đó cần nằm trong `~/.ssh/authorized_keys` của user `GCP_DEPLOY_USER` trên VPS.

## 5. Chuẩn bị project trên VPS

Workflow sẽ tự tạo thư mục deploy nếu chưa có:

- `/opt/tcb-fraud-detection-mlops`

Workflow cũng sẽ tự tạo `.env` từ `.env.example` ở lần chạy đầu nếu file `.env` chưa tồn tại. Sau đó bạn vẫn phải SSH vào VPS để cập nhật secret thật trong file `.env`.

Các biến tối thiểu hiện repo đang dùng:

- `MINIO_ROOT_USER`
- `MINIO_ROOT_PASSWORD`
- `MINIO_BUCKET`
- `GRAFANA_ADMIN_USER`
- `GRAFANA_ADMIN_PASSWORD`
- `AIRFLOW_UID`
- `FASTAPI_PORT`
- `MINIO_API_PORT`
- `MINIO_CONSOLE_PORT`
- `MLFLOW_PORT`
- `AIRFLOW_PORT`
- `PROMETHEUS_PORT`
- `GRAFANA_PORT`
- `CADVISOR_PORT`

## 6. Firewall / network

Ít nhất phải mở cổng SSH cho GitHub Actions SSH vào VPS:

- `22/tcp` hoặc cổng SSH tùy bạn cấu hình

Nếu bạn muốn truy cập trực tiếp từ bên ngoài, cần mở thêm các cổng phù hợp:

- `8000` cho FastAPI qua load balancer
- `5000` cho MLflow
- `8080` cho Airflow
- `3000` cho Grafana
- `9000` và `9001` cho MinIO
- `9090` cho Prometheus

## 7. Kiểm tra trước khi chạy workflow

SSH thử từ máy cá nhân vào VPS bằng đúng key pair đó trước:

```bash
ssh <deploy-user>@<deploy-host>
```

Nếu bước này chưa thành công thì GitHub Actions cũng sẽ chưa deploy được.
