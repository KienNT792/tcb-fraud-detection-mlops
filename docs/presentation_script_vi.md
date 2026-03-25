# Script Thuyết Trình Hệ Thống TCB Fraud Detection MLOps

Mục tiêu: script nói 5-7 phút, tập trung vào kiến trúc hệ thống và luồng vận hành end-to-end.

## Slide 1 - Mở đầu

Xin chào thầy/cô và các bạn. Hôm nay em sẽ trình bày đề tài "TCB Fraud Detection - End-to-End MLOps Pipeline".

Đây là một hệ thống phát hiện gian lận giao dịch thẻ tín dụng theo thời gian thực, nhưng điểm em muốn nhấn mạnh không chỉ là model machine learning, mà là toàn bộ vòng đời vận hành của model trong môi trường gần với production.

Hệ thống bao gồm các phần: xử lý dữ liệu, huấn luyện model, đánh giá, đăng ký version, serving qua API, canary deployment, monitoring và orchestration tự động.

## Slide 2 - Bài toán và mục tiêu

Bộ dữ liệu của đề tài gồm 100.000 giao dịch, trong đó giao dịch fraud chỉ chiếm khoảng 2,84%. Đây là bài toán mất cân bằng lớp rất rõ, nên nếu dùng accuracy thì sẽ gây hiểu nhầm. Vì vậy, hệ thống ưu tiên các metric phù hợp hơn như PR-AUC, recall và F1-score.

Mục tiêu của đề tài là trả lời câu hỏi: làm sao để không chỉ train được một model phát hiện fraud, mà còn có thể đưa model đó vào vận hành, theo dõi nó, cập nhật nó an toàn và rollback khi cần thiết.

## Slide 3 - Kiến trúc tổng thể

Về tổng thể, hệ thống được chia thành 4 lớp chính.

Lớp thứ nhất là dữ liệu và machine learning pipeline. Dữ liệu raw được quản lý bằng DVC, sau đó đi qua preprocessing, training và evaluation.

Lớp thứ hai là model registry và artifact storage. MLflow dùng để track thực nghiệm, lưu metrics, lưu model version; MinIO đóng vai trò artifact store theo chuẩn S3.

Lớp thứ ba là serving. Model đang production được phục vụ bởi FastAPI stable. Model mới sẽ được đưa lên FastAPI candidate. Nginx dùng làm load balancer để chia traffic giữa stable và candidate theo cơ chế canary.

Lớp thứ tư là observability và orchestration. Prometheus thu thập metrics, Grafana hiển thị dashboard, Evidently AI theo dõi drift, và Airflow điều phối quá trình retraining theo lịch.

Nếu tóm gọn trong một câu, kiến trúc này cho phép hệ thống đi từ dữ liệu đến model, từ model đến API, và từ API đến monitoring một cách liên tục.

## Slide 4 - Pipeline machine learning

Đi sâu vào phần machine learning, luồng xử lý dữ liệu gồm các bước: load dữ liệu, validate schema, làm sạch, chia tập theo thời gian để tránh data leakage, tạo feature, sau đó lưu artifact xử lý.

Model được huấn luyện bằng XGBoost. Trong bài toán này, việc mất cân bằng dữ liệu được xử lý bằng `scale_pos_weight`. Toàn bộ tham số và kết quả huấn luyện được log lên MLflow.

Sau training là evaluation. Ở đây, hệ thống không dùng một metric duy nhất, mà kết hợp threshold tuning, SHAP explainability và fairness theo nhóm khách hàng. Điều này quan trọng vì trong bài toán tài chính, model không chỉ cần chính xác mà còn cần dễ giải thích và kiểm soát rủi ro.

## Slide 5 - Model registry và runtime bundle

Sau khi model đạt chất lượng yêu cầu, model sẽ được đưa vào MLflow Model Registry. Mỗi version có thể đi qua các stage như Staging và Production.

Hệ thống không chỉ lưu file model, mà đóng gói thành runtime bundle. Runtime bundle này bao gồm model, feature metadata và các artifact cần thiết cho suy luận. Nhờ vậy, khi deploy sang serving API, candidate service có thể nạp đúng bộ artifact đã được version hóa.

Ý nghĩa của cách làm này là giảm rủi ro "train một nơi, deploy một nẻo", vì mọi thứ cần cho inference đã được đóng gói rõ ràng.

## Slide 6 - Serving và canary deployment

Ở tầng serving, hệ thống có hai FastAPI service: `fastapi-stable` và `fastapi-candidate`.

`fastapi-stable` phục vụ model đang được xem là ổn định trong production. `fastapi-candidate` phục vụ model mới vừa được train hoặc vừa được stage lên để thử nghiệm.

Nginx load balancer dùng cơ chế split traffic. Ví dụ, 90% request sẽ đi vào stable, 10% request sẽ đi vào candidate. Cách này giúp chúng ta kiểm chứng model mới dưới traffic thật mà không phải thay thế toàn bộ hệ thống ngay lập tức.

Hệ thống cũng có các endpoint chính như `/predict`, `/predict/batch`, `/health`, `/metrics`, và `/monitoring/drift`.

Nếu candidate hoạt động tốt, chúng ta có thể promote candidate thành stable. Nếu candidate có dấu hiệu bất thường, có thể rollback về stable. Đây là điểm rất quan trọng trong MLOps, vì deployment của model cần ưu tiên an toàn hơn là chỉ deploy nhanh.

## Slide 7 - Monitoring và observability

Monitoring trong hệ thống này gồm 3 lớp.

Lớp thứ nhất là metrics ở tầng API. FastAPI expose `/metrics`, trong đó có các metric như tổng số request, request latency, trạng thái model đã load hay chưa, số lượng prediction, tỉ lệ giao dịch bị đánh dấu fraud, và các chỉ số drift.

Lớp thứ hai là Prometheus. Prometheus định kỳ scrape metrics từ FastAPI stable, FastAPI candidate, nginx exporter, node exporter, cadvisor, và chính nó. Sau khi thu thập, Prometheus lưu dữ liệu time-series và đánh giá các alert rule.

Lớp thứ ba là Grafana. Grafana không đi lấy metrics trực tiếp từ service, mà query Prometheus để hiển thị dashboard. Ngắn gọn là: service phát metrics, Prometheus thu thập và lưu, Grafana query Prometheus để vẽ biểu đồ.

Ở góc nhìn nghiệp vụ, dashboard cho phép theo dõi request rate, latency, error rate, prediction volume, fraud rate, drift ratio, và tài nguyên hệ thống. Nhờ đó, nhóm vận hành có thể nhìn thấy sức khỏe của model và hạ tầng trong cùng một nơi.

## Slide 8 - Drift detection và alerting

Ngoài monitoring hạ tầng, hệ thống còn theo dõi data drift bằng Evidently AI theo cơ chế sliding window.

Ý tưởng ở đây là so sánh phân phối dữ liệu hiện tại với dữ liệu tham chiếu. Nếu dữ liệu vào bắt đầu lệch đáng kể so với baseline, hệ thống sẽ tăng drift score và có thể kích hoạt cảnh báo.

Bên cạnh đó, Prometheus có các alert rule cho những tình huống như:

- API bị down
- tỉ lệ lỗi 5xx quá cao
- latency P95 vượt ngưỡng
- model không được load
- drift ratio vượt ngưỡng
- candidate có hành vi kém hơn stable trong quá trình canary

Ý nghĩa của alerting là biến monitoring từ mức "chỉ để xem" thành mức "có thể ra quyết định vận hành".

## Slide 9 - Airflow và tự động hóa vòng đời model

Phần orchestration được thực hiện bằng Airflow với DAG `fraud_detection_training_pipeline`, chạy hằng ngày lúc 02:00 UTC.

Lượt chạy tổng quát sẽ là: kiểm tra chất lượng model hiện tại, preprocessing, training, evaluation, stage candidate, và verify candidate.

Như vậy, hệ thống không phải là một demo train model thủ công. Đây là một pipeline có lịch chạy, có registry, có deployment logic và có monitoring sau deployment.

Đó là tinh thần chính của MLOps: biến machine learning thành một hệ thống có thể vận hành lặp lại và kiểm soát được.

## Slide 10 - Giá trị của hệ thống

Nếu em phải tóm tắt giá trị của đề tài trong 3 ý, em sẽ nói như sau.

Ý thứ nhất, đây là hệ thống end-to-end, bao quát từ dữ liệu đến production thay vì chỉ dừng lại ở một notebook train model.

Ý thứ hai, hệ thống ưu tiên tính an toàn khi deploy model thông qua canary deployment, monitoring và rollback.

Ý thứ ba, hệ thống ưu tiên khả năng quan sát và vận hành, thông qua MLflow, Prometheus, Grafana, Evidently AI và Airflow.

## Slide 11 - Kết luận

Tổng kết lại, đề tài này giải quyết bài toán fraud detection theo hướng MLOps hoàn chỉnh.

Luồng chính của hệ thống là:

Dữ liệu được version hóa và xử lý -> model được train và evaluate -> model được đăng ký trên MLflow -> model được đưa lên stable hoặc candidate qua FastAPI -> traffic được chia qua Nginx -> hệ thống được giám sát bằng Prometheus và Grafana -> và quá trình retraining được điều phối bằng Airflow.

Nếu machine learning giúp tạo ra model, thì MLOps giúp biến model đó thành một sản phẩm có thể vận hành được. Đây cũng là thông điệp chính mà em muốn trình bày trong đề tài này.

Em xin hết. Cảm ơn thầy/cô và các bạn đã lắng nghe.

## Ghi chú để trình bày tự tin hơn

- Khi nói về Prometheus và Grafana, có thể nhớ câu then chốt: "Prometheus scrape, Grafana query."
- Khi nói về canary, có thể nhớ câu then chốt: "Không đổi toàn bộ traffic ngay lập tức, mà chia nhỏ để kiểm chứng an toàn."
- Khi nói về MLOps, có thể nhớ câu then chốt: "Giá trị của đề tài nằm ở khả năng vận hành model, không chỉ nằm ở việc huấn luyện model."

## Nếu cần demo nhanh

- Mở MLflow để nói về training run, metrics và model registry.
- Mở Grafana để nói về request rate, latency và drift.
- Mở Prometheus để cho thấy alert rules hoặc query một metric như `max(tcb_drift_ratio)`.
- Mở API `/health` hoặc `/metrics` để kết nối phần serving với phần monitoring.
