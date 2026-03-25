# Monitoring Stack: Prometheus → Grafana → AlertManager

Tài liệu này mô tả kiến trúc, cấu hình và luồng hoạt động của hệ thống giám sát (Monitoring Stack) trong dự án **TCB Fraud Detection MLOps**.

---

## 1. Tổng quan kiến trúc

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Metrics Sources                             │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  fastapi-stable  │  │ fastapi-candidate│  │  node-exporter   │  │
│  │   :8000/metrics  │  │  :8000/metrics   │  │   :9100/metrics  │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  │
│           │                     │                      │            │
│  ┌────────┴─────────┐  ┌────────┴─────────┐            │            │
│  │  nginx-exporter  │  │    cAdvisor      │            │            │
│  │   :9113/metrics  │  │  :8080/metrics   │            │            │
│  └────────┬─────────┘  └────────┬─────────┘            │            │
└───────────┼─────────────────────┼────────────────────-─┼────────────┘
            │                     │                      │
            └─────────────────────┼──────────────────────┘
                                  │ scrape (15s)
                                  ▼
                       ┌──────────────────┐
                       │   Prometheus     │
                       │     :9090        │
                       │  + alerts.yml    │
                       └────────┬─────────┘
                    ┌───────────┴───────────┐
                    │ evaluate rules        │ expose metrics
                    ▼                       ▼
          ┌──────────────────┐    ┌──────────────────┐
          │  AlertManager    │    │     Grafana       │
          │     :9093        │    │     :3000         │
          └────────┬─────────┘    └──────────────────┘
                   │
         ┌─────────┴──────────┐
         │ auto_rollback=true │
         ▼                    ▼
  ┌─────────────────┐  ┌─────────────────┐
  │    Rollback     │  │  default-null   │
  │  Webhook :8085  │  │  (silent sink)  │
  └─────────────────┘  └─────────────────┘
```

---

## 2. Prometheus

**File cấu hình:** [`monitoring/prometheus/prometheus.yml`](../monitoring/prometheus/prometheus.yml)

### 2.1 Thông số toàn cục

| Tham số              | Giá trị | Mô tả                                      |
|----------------------|---------|--------------------------------------------|
| `scrape_interval`    | `15s`   | Tần suất thu thập metrics từ các target    |
| `evaluation_interval`| `15s`   | Tần suất đánh giá các alerting rules       |

### 2.2 Scrape Targets

| Job Name           | Target                   | Mô tả                                         |
|--------------------|--------------------------|-----------------------------------------------|
| `prometheus`       | `prometheus:9090`        | Self-monitoring của Prometheus                |
| `fastapi-stable`   | `fastapi-stable:8000`    | Serving API phiên bản ổn định (production)    |
| `fastapi-candidate`| `fastapi-candidate:8000` | Serving API phiên bản thử nghiệm (canary)     |
| `loadbalancer`     | `nginx-exporter:9113`    | Metrics Nginx load balancer qua nginx-exporter|
| `node-exporter`    | `node-exporter:9100`     | Metrics tài nguyên host (CPU, RAM, Disk)      |
| `cadvisor`         | `cadvisor:8080`          | Metrics tài nguyên từng Docker container      |

### 2.3 Custom Metrics (Serving API)

Các metrics tùy chỉnh do FastAPI serving API (`tcb_*`) expose tại `/metrics`:

| Metric Name                        | Loại     | Mô tả                                           |
|------------------------------------|----------|-------------------------------------------------|
| `tcb_http_requests_total`          | Counter  | Tổng số HTTP request, nhãn `status` (2xx/5xx)  |
| `tcb_http_request_duration_seconds`| Histogram| Phân phối latency theo từng request             |
| `tcb_model_loaded`                 | Gauge    | Trạng thái model (`1` = loaded, `0` = unloaded) |
| `tcb_prediction_samples_total`     | Counter  | Tổng số prediction request                     |
| `tcb_prediction_fraud_total`       | Counter  | Tổng số prediction phân loại là fraud           |
| `tcb_drift_ratio`                  | Gauge    | Tỉ lệ feature drift hiện tại (từ Evidently AI)  |

---

## 3. Alerting Rules

**File cấu hình:** [`monitoring/prometheus/alerts.yml`](../monitoring/prometheus/alerts.yml)

Các rules được tổ chức thành 2 nhóm:

### 3.1 Nhóm `fraud-api-alerts` — Giám sát Production API

| Alert Name               | Biểu thức PromQL                                                          | Ngưỡng    | Thời gian | Severity   | Mô tả                                                      |
|--------------------------|---------------------------------------------------------------------------|-----------|-----------|------------|------------------------------------------------------------|
| `FraudApiDown`           | `up{job="fastapi-stable"} == 0`                                           | —         | 2 phút    | `critical` | Stable API không phản hồi scrape từ Prometheus             |
| `FraudApiHighErrorRate`  | `rate(5xx) / rate(all) > 0.05`                                            | > 5%      | 5 phút    | `warning`  | Tỉ lệ lỗi 5xx vượt ngưỡng 5%                               |
| `FraudApiHighLatencyP95` | `histogram_quantile(0.95, ...) > 1.0`                                     | > 1 giây  | 10 phút   | `warning`  | Latency P95 vượt 1 giây                                    |
| `FraudModelNotLoaded`    | `min(tcb_model_loaded) < 1`                                               | —         | 3 phút    | `critical` | Ít nhất 1 instance API báo model chưa được load            |
| `FraudDriftRatioHigh`    | `max(tcb_drift_ratio) > 0.2`                                              | > 0.20    | 10 phút   | `warning`  | Data drift ratio vượt ngưỡng 20%                           |

### 3.2 Nhóm `fraud-canary-alerts` — Giám sát Canary Deployment

| Alert Name                      | Biểu thức PromQL                                                                                  | Thời gian | Severity  | Mô tả                                                                       |
|---------------------------------|---------------------------------------------------------------------------------------------------|-----------|-----------|-----------------------------------------------------------------------------|
| `CandidateModelBehaviorRegression` | Fraud rate của candidate < 70% fraud rate của stable trong khi có đủ traffic                  | 15 phút   | `warning` | Candidate model phát hiện fraud thấp hơn đáng kể so với stable, kích hoạt auto-rollback |

> **Auto-rollback trigger:** Alert `CandidateModelBehaviorRegression` được gắn nhãn `auto_rollback="true"`, AlertManager sẽ route alert này đến webhook rollback thay vì silent sink.

---

## 4. AlertManager

**File cấu hình:** [`monitoring/alertmanager/alertmanager.yml`](../monitoring/alertmanager/alertmanager.yml)

### 4.1 Cấu hình toàn cục

| Tham số           | Giá trị | Mô tả                                              |
|-------------------|---------|----------------------------------------------------|
| `resolve_timeout` | `5m`    | Thời gian chờ trước khi đánh dấu alert đã giải quyết |
| `group_wait`      | `30s`   | Thời gian chờ gom nhóm alert trước khi gửi lần đầu |
| `group_interval`  | `5m`    | Khoảng thời gian gửi lại nếu nhóm có alert mới     |
| `repeat_interval` | `30m`   | Khoảng thời gian lặp lại thông báo nếu chưa resolved|

### 4.2 Routing Logic

```
Route: group_by=[alertname]
│
├── matchers: auto_rollback="true"
│     └── receiver: rollback-webhook
│           └── POST http://rollback-automation:8085/alertmanager
│
└── (default)
      └── receiver: default-null  (silent, không gửi thông báo)
```

### 4.3 Receivers

| Receiver           | Loại    | Endpoint                                   | Mô tả                                                       |
|--------------------|---------|--------------------------------------------|-------------------------------------------------------------|
| `default-null`     | —       | —                                          | Sink rỗng, nuốt tất cả alert không khớp rule nào             |
| `rollback-webhook` | Webhook | `http://rollback-automation:8085/alertmanager` | Kích hoạt auto-rollback qua dịch vụ `rollback-automation`   |

### 4.4 Auto-Rollback Service

**File:** [`monitoring/automation/alertmanager_rollback_receiver.py`](../monitoring/automation/alertmanager_rollback_receiver.py)

Một FastAPI microservice nhỏ lắng nghe webhook từ AlertManager. Khi nhận payload alert với `auto_rollback="true"`, service này sẽ thực hiện hành động rollback canary — chuyển load balancer trở lại 100% stable instance.

---

## 5. Grafana

**Dashboard:** [`monitoring/grafana/dashboards/tcb-mlops-overview.json`](../monitoring/grafana/dashboards/tcb-mlops-overview.json)

**Datasource:** [`monitoring/grafana/provisioning/datasources/datasource.yml`](../monitoring/grafana/provisioning/datasources/datasource.yml)

### 5.1 Provisioning tự động

Grafana được cấu hình provisioning qua thư mục `monitoring/grafana/provisioning/`:
- **Datasources:** Tự động kết nối đến Prometheus tại `http://prometheus:9090`.
- **Dashboards:** Tự động import dashboard `tcb-mlops-overview.json` khi container khởi động.

### 5.2 Nội dung Dashboard `tcb-mlops-overview`

Dashboard cung cấp cái nhìn tổng quan về toàn bộ hệ thống MLOps, bao gồm các panel:

| Panel                              | Nguồn dữ liệu                        | Mô tả                                         |
|------------------------------------|--------------------------------------|-----------------------------------------------|
| Request Rate (stable vs candidate) | `tcb_http_requests_total`            | Throughput theo thời gian thực của 2 phiên bản|
| Error Rate (5xx)                   | `tcb_http_requests_total{status=~"5.."}` | Tỉ lệ lỗi HTTP                            |
| Latency P50 / P95 / P99            | `tcb_http_request_duration_seconds`  | Phân phối độ trễ theo percentile              |
| Fraud Prediction Rate              | `tcb_prediction_fraud_total`         | Xu hướng phát hiện fraud theo thời gian       |
| Model Load Status                  | `tcb_model_loaded`                   | Trạng thái model trên từng instance           |
| Data Drift Ratio                   | `tcb_drift_ratio`                    | Feature drift ratio từ Evidently AI           |
| CPU / Memory Usage                 | `node-exporter`, `cadvisor`          | Tài nguyên host và container                  |
| Nginx Connections                  | `nginx-exporter`                     | Trạng thái load balancer                      |

---

## 6. Luồng hoạt động end-to-end

```
1. Mỗi 15 giây, Prometheus scrape metrics từ tất cả targets.
2. Prometheus đánh giá alerting rules trong alerts.yml.
3a. Nếu có alert → gửi đến AlertManager:9093.
3b. Metrics được Grafana kéo về hiển thị trực quan trên dashboard.
4. AlertManager nhóm và route alert:
   - auto_rollback=true → POST webhook → rollback-automation:8085 → kích hoạt rollback.
   - Các alert khác → default-null (silent).
5. Rollback automation service xử lý và chuyển traffic 100% về stable.
```

---

## 7. Truy cập các dịch vụ

| Dịch vụ      | URL                     | Credentials mặc định |
|--------------|-------------------------|----------------------|
| Prometheus   | http://localhost:9090   | —                    |
| Grafana      | http://localhost:3000   | `admin` / `admin`    |
| AlertManager | http://localhost:9093   | —                    |

> **Lưu ý:** Đối với môi trường production, thay đổi credentials mặc định của Grafana và cấu hình TLS cho tất cả các endpoint.

---

## 8. Cách thêm alert rule mới

1. Mở file [`monitoring/prometheus/alerts.yml`](../monitoring/prometheus/alerts.yml).
2. Thêm rule mới vào group phù hợp (hoặc tạo group mới).
3. Định nghĩa `expr` bằng PromQL, `for` (thời gian sustain), `labels.severity`, và `annotations`.
4. Nếu alert cần kích hoạt auto-rollback, thêm label `auto_rollback: "true"`.
5. Reload cấu hình Prometheus (không cần restart):
   ```bash
   curl -X POST http://localhost:9090/-/reload
   ```
6. Kiểm tra alert trong tab **Alerts** tại Prometheus UI.
