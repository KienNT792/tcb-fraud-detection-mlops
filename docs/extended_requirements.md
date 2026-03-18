# Extended Requirements - TCB Fraud Detection MLOps

## 1. Mục tiêu mở rộng

Project TCB Fraud Detection cần được phát triển từ một repo model + API thành một hệ thống MLOps hoàn chỉnh, có khả năng:

- Tự động hóa toàn bộ vòng đời ML từ ingest dữ liệu đến deploy và monitoring
- Theo dõi, quản lý và truy vết model xuyên suốt vòng đời vận hành
- Hỗ trợ triển khai thực tế với Docker, Kubernetes và CI/CD
- Hỗ trợ cả batch pipeline và near real-time scoring
- Dễ tái lập, dễ bảo trì và dễ mở rộng

## 2. Functional Requirements

### 2.1 Data Ingestion va Data Pipeline

Hệ thống phải:

- Nhận dữ liệu giao dịch từ file batch hoặc stream
- Hỗ trợ ingest realtime qua Kafka
- Validate schema đầu vào trước khi đưa vào pipeline
- Tách vùng dữ liệu thành `raw`, `validated`, `processed`, `feature-ready`
- Có versioning cho dataset và artifact dữ liệu
- Có log đầy đủ ở từng bước xử lý
- Cho phép rerun từng bước độc lập khi pipeline fail

Thành phần công nghệ đề xuất:

- Apache Airflow cho orchestration
- Great Expectations hoặc công cụ tương đương cho data validation
- DVC hoặc lakeFS/MinIO cho data versioning và artifact storage

Acceptance criteria tối thiểu:

- Phát hiện thiếu cột bắt buộc
- Phát hiện sai kiểu dữ liệu
- Phát hiện duplicate bất thường
- Ghi log input/output và trạng thái cho từng stage

### 2.2 Model Training Pipeline

Pipeline training phải:

- Được orchestration bằng Airflow DAG
- Hỗ trợ chạy theo schedule hoặc trigger thủ công
- Log toàn bộ experiment vào MLflow
- Lưu `hyperparameters`, `metrics`, `model artifact`, `feature importance`, `optimal threshold`, `model version`
- Có bước so sánh champion/challenger
- Chỉ promote model mới nếu đạt ngưỡng tối thiểu

Acceptance criteria đề xuất:

- `Recall >= 0.95`
- `PR-AUC >= baseline`
- `F1 >= baseline - tolerance`
- Không có schema mismatch giữa training feature và inference feature

### 2.3 Model Registry va Model Lifecycle

Hệ thống phải dùng MLflow Model Registry để:

- Quản lý version model
- Chuyển trạng thái `Staging`, `Production`, `Archived`
- Lưu metadata cho từng model version
- Rollback về version trước đó khi production gặp lỗi

Model production phải truy vết được:

- `dataset version`
- `code version / git commit`
- `mlflow run_id`
- `training timestamp`
- `actor` hoặc `trigger channel`

### 2.4 Serving va Realtime Inference

Serving layer phải:

- Containerize bằng Docker
- Chạy nhiều replica trên Kubernetes
- Có `Service` và `Ingress` hoặc Load Balancer
- Hỗ trợ autoscaling theo CPU, memory hoặc request rate
- Có endpoint `health`, `readiness`, `liveness`
- Có request logging và response metadata cho monitoring
- Hỗ trợ nhận transaction từ REST API và Kafka consumer

Kết quả scoring phải gồm:

- `fraud_score`
- `is_fraud_pred`
- `threshold`
- `risk_level`
- `model_version`
- `prediction_timestamp`

### 2.5 Monitoring va Alerting

Hệ thống phải có monitoring cho cả hạ tầng và chất lượng model.

Monitoring hệ thống:

- CPU
- RAM
- request latency
- throughput
- error rate

Công nghệ đề xuất:

- Prometheus
- Grafana

Monitoring ML:

- data drift
- prediction drift
- feature distribution shift
- tỷ lệ positive prediction
- model performance degradation theo thời gian

Công nghệ đề xuất:

- Evidently AI
- MLflow metrics logging
- custom monitoring jobs

Alerting phải hỗ trợ:

- latency vượt ngưỡng
- API lỗi tăng cao
- data drift vượt threshold
- model quality giảm

Kênh cảnh báo có thể dùng:

- email
- Slack
- webhook

### 2.6 Orchestration

Apache Airflow phải orchestrate các job chính:

- `data_validation_task`
- `preprocessing_task`
- `feature_engineering_task`
- `training_task`
- `evaluation_task`
- `register_model_task`
- `deploy_task`
- `monitoring_task`

Yêu cầu với DAG:

- Có retry policy
- Có dependency rõ ràng giữa các task
- Có logging và tracking trạng thái task
- Hỗ trợ schedule

Lịch chạy đề xuất:

- training hằng ngày
- monitoring drift mỗi giờ
- batch scoring mỗi 15 phút

### 2.7 CI/CD

CI phải tự động thực hiện khi có pull request hoặc push vào branch chính:

- chạy unit tests
- chạy lint và format checks
- build Docker image
- validate Airflow DAG
- kiểm tra file cấu hình Kubernetes
- chạy integration test cho API

CD phải:

- push Docker image lên registry
- deploy lên staging
- chạy smoke test
- nếu pass thì promote lên production

Công cụ đề xuất:

- GitHub Actions cho test và code quality
- Jenkins hoặc GitHub Actions cho build, deploy và rollback

## 3. Non-Functional Requirements

### 3.1 Scalability

- API inference phải scale ngang được
- Kubernetes phải hỗ trợ tối thiểu 2-3 replicas cho serving service
- Load Balancer phải phân phối đều request
- Kafka phải hỗ trợ mở rộng partition khi volume tăng

### 3.2 Reliability

- Mục tiêu uptime API: `>= 99%`
- Container phải tự restart khi crash
- Deployment phải hỗ trợ rollback
- Airflow tasks phải có retry
- Serving không được load model lại ở mỗi request

### 3.3 Performance

- Latency cho single request nên `< 300 ms`
- Batch prediction phải hỗ trợ nhiều giao dịch trong một request
- Kafka consumer phải xử lý liên tục với độ trễ thấp

### 3.4 Security

- API cần authentication, ví dụ API key hoặc JWT
- Secret phải được quản lý qua Kubernetes Secret, Jenkins Credentials hoặc `.env` chỉ dùng local
- Không commit credential vào repo
- Giới hạn quyền truy cập vào model registry và deployment environment

### 3.5 Reproducibility

Mỗi training run phải lưu được:

- `code version`
- `data version`
- `config version`
- `model artifact`

Docker environment phải đảm bảo đồng nhất giữa local, staging và production.

### 3.6 Maintainability

- Có README và tài liệu triển khai rõ ràng
- Có Makefile hoặc script để chạy local nhanh
- Module tách lớp rõ ràng, dễ mở rộng
- Có test coverage ở mức chấp nhận được cho pipeline và serving

## 4. Công nghệ bắt buộc de xuat

Các thành phần nền tảng nên được đưa vào phạm vi chính thức của project:

- Docker cho container hóa toàn bộ service chính
- Kubernetes cho deploy, scale và self-healing
- MLflow cho tracking, artifact logging và model registry
- Airflow cho orchestration và scheduling
- Kafka cho transaction streaming và near real-time scoring
- Jenkins hoặc GitHub Actions cho CI/CD
- Prometheus, Grafana, Evidently AI cho monitoring
- MinIO hoặc object storage tương đương cho model, report và dataset artifact

## 5. Kiến trúc mở rộng de xuat

Kiến trúc mục tiêu gồm các thành phần sau:

- Kafka nhận transaction stream
- Airflow orchestrate validation, preprocessing, train, evaluate, register, deploy
- MLflow quản lý experiment và model registry
- FastAPI phục vụ online inference
- Kafka consumer thực hiện stream scoring
- Docker đóng gói các service
- Kubernetes triển khai và scale service
- Ingress hoặc Load Balancer route traffic
- Prometheus + Grafana giám sát hạ tầng
- Evidently AI giám sát drift và chất lượng model
- MinIO hoặc object storage lưu artifacts
- Jenkins hoặc GitHub Actions điều phối CI/CD

## 6. Deliverables cho do an

Deliverables nên có để project đạt mức MLOps hoàn chỉnh:

- Source code đầy đủ
- Dockerfile cho từng service chính
- `docker-compose.yml` chạy local
- Airflow DAG hoàn chỉnh
- MLflow tracking + model registry hoạt động
- Kafka demo pipeline cho realtime transaction
- Kubernetes manifests hoặc Helm chart
- CI/CD pipeline bằng Jenkins hoặc GitHub Actions
- Dashboard monitoring
- README hướng dẫn chạy local, train, serve, deploy, monitor
- Demo end-to-end từ ingest -> train -> register -> deploy -> predict -> monitor

## 7. Roadmap de xuat

### Phase 1

- Hoàn thiện Docker cho training, serving, MLflow, Airflow
- Hoàn thiện README, Makefile, local run
- Hoàn thiện Airflow DAG
- Tích hợp MLflow tracking đầy đủ

### Phase 2

- Viết GitHub Actions hoặc Jenkins pipeline
- Deploy FastAPI bằng Kubernetes
- Thêm Service, Ingress, autoscaling
- Bổ sung integration test và smoke test

### Phase 3

- Tích hợp Kafka cho streaming inference
- Tích hợp Evidently AI monitoring
- Bổ sung dashboard Prometheus/Grafana
- Hoàn thiện rollback và champion/challenger promotion workflow

## 8. Doan requirement tong hop cho bao cao

Project không chỉ dừng ở mức xây dựng model phát hiện gian lận và API dự đoán, mà cần được phát triển thành một hệ thống MLOps hoàn chỉnh. Hệ thống phải hỗ trợ tự động hóa toàn bộ vòng đời machine learning từ ingest dữ liệu, tiền xử lý, huấn luyện, đánh giá, đăng ký model, triển khai, giám sát đến tái huấn luyện. Apache Airflow được sử dụng để orchestration các pipeline batch và training workflow. MLflow được sử dụng cho experiment tracking, artifact logging và model registry. Toàn bộ service phải được container hóa bằng Docker và triển khai trên Kubernetes để hỗ trợ scale ngang, self-healing và rolling update. Hệ thống cần có CI/CD pipeline bằng Jenkins hoặc GitHub Actions để tự động kiểm thử, build image và triển khai. Với bài toán giao dịch realtime, Kafka được sử dụng làm message broker để tiếp nhận luồng transaction và hỗ trợ scoring gần thời gian thực. Ngoài ra, hệ thống cần có monitoring cho cả hạ tầng và chất lượng model thông qua Prometheus, Grafana và Evidently AI, đồng thời sử dụng Ingress hoặc Load Balancer để phân phối traffic tới nhiều replica của serving service. Các thành phần này giúp project đáp ứng tốt hơn các yêu cầu của một hệ thống MLOps thực tế: tự động hóa, tái lập, quan sát được, dễ mở rộng và sẵn sàng vận hành.

## 9. Doi chieu voi repo hien tai

Dựa trên codebase hiện có, trạng thái hiện tại của repo so với target architecture như sau:

- Da co:
  - `ml_pipeline/src/preprocess.py`
  - `ml_pipeline/src/train.py`
  - `ml_pipeline/src/evaluate.py`
  - `ml_pipeline/src/inference.py`
  - `serving_api/app/main.py`
  - MLflow logging ở mức experiment tracking
  - artifact model và evaluation report đã commit
  - unit test cho preprocessing và serving API

- Chua hoan thien:
  - Airflow DAG đang trống
  - monitoring drift job đang trống
  - GitHub Actions workflow đang trống
  - `docker-compose.yml`, `Makefile`, `README.md`, `docs/api_docs.md` đang trống
  - chưa có Kubernetes manifests hoặc Helm chart
  - chưa có Kafka producer/consumer
  - chưa có MLflow Model Registry flow
  - chưa có auth cho API
  - chưa có Prometheus metrics export trong API
  - chưa có pipeline deploy staging/prod

## 10. Ghi chu cho phase deploy

Phần triển khai cloud se duoc chot sau khi co thong tin VPS Google Cloud, bao gom:

- IP hoac domain
- OS image
- CPU, RAM, disk
- cach mo port
- registry se su dung
- phuong an chay Docker Compose hay Kubernetes
- secret va bien moi truong can cap

Tai thoi diem co thong tin ha tang, tai lieu nay se duoc cap nhat thanh deployment plan cu the.
