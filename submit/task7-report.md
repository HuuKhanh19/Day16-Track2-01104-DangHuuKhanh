# Báo cáo Task 7 - Quan sát tài nguyên, network và chi phí trên GCP

1. Hạ tầng được triển khai trên project `track2-day16-01104` với VM `ai-gpu-node` loại `e2-medium` (2 vCPU, 4 GB RAM); các operation từ lúc tạo tài nguyên đầu tiên đến khi Terraform ghi state hoàn tất mất khoảng 3 phút 27 giây.
2. Startup script bắt đầu lúc 14:04:12 UTC ngày 14/08/2026 và báo môi trường LightGBM sẵn sàng lúc 14:05:37 UTC, tương đương khoảng 1 phút 25 giây chạy bootstrap và khoảng 1 phút 53 giây kể từ khi VM được tạo.
3. Dataset có 284.807 giao dịch, 30 đặc trưng và 492 dòng fraud; thời gian đọc dữ liệu là 1,85 giây, còn thời gian huấn luyện `LGBMClassifier` là 8,59 giây.
4. Model đạt AUC-ROC 0,9428, accuracy 0,99949 và F1-score 0,8497; do dữ liệu mất cân bằng nên AUC-ROC, precision, recall và F1 có ý nghĩa đánh giá cao hơn việc chỉ nhìn accuracy. `best_iteration` là `null` vì benchmark không sử dụng early stopping.
5. Precision 0,8632 nghĩa là khoảng 86,32% các giao dịch được model cảnh báo fraud là fraud thật, còn recall 0,8367 nghĩa là model phát hiện khoảng 83,67% fraud và bỏ sót khoảng 16,33% số fraud thực tế.
6. Inference một dòng có latency trung bình 0,98 ms; với batch 1.000 dòng, thời gian dự đoán là 13,70 ms và throughput đạt khoảng 73.004 dòng/giây, cho thấy batch inference tận dụng xử lý vector hóa tốt hơn dự đoán từng dòng riêng lẻ.
7. Khi huấn luyện, tiến trình `python3` sử dụng gần 200% CPU trên máy 2 vCPU, trong khi RAM chỉ dùng khoảng 861 MiB trên tổng 3,8 GiB và còn khoảng 3,0 GiB khả dụng; vì vậy CPU là bottleneck rõ hơn RAM.
8. Ảnh GCP Observability ghi nhận đỉnh CPU và network đúng thời điểm benchmark; Memory Utilization trên dashboard không có dữ liệu vì VM chưa cài Ops Agent, nên mức sử dụng RAM được đối chiếu bằng kết quả `free -h` trên VM.
9. Billing Report đã ghi nhận usage cost 4.771 VND cho project `track2-day16-01104`; khoản savings/free trial bù trừ -4.771 VND nên tổng thanh toán hiện tại là 0 VND. Các thành phần có khả năng đóng góp chi phí gồm Compute Engine `e2-medium`, Cloud NAT và External HTTP Load Balancer.
