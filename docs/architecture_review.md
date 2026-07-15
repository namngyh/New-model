# Audit kiến trúc ban đầu

Ngày audit: 2026-07-15.

Workspace ban đầu chỉ có `VNINDEX_Daily.csv`; remote `main` chỉ có một README từ commit khởi tạo. Vì vậy không có pipeline cũ để bảo tồn hoặc mô hình cũ để dùng làm benchmark.

## Phát hiện dữ liệu

- CSV có 12 token ở nhiều dòng dù header chỉ có 6 cột hữu ích. Nguyên nhân là dấu phẩy hàng nghìn trong OHLCV không được quote.
- Không thể đọc đúng bằng `pandas.read_csv` mặc định; cần parser tái dựng 5 trường OHLCV và kiểm tra quan hệ High/Low.
- Parser audit nhận diện 6.306 quan sát hợp lệ từ 2000-07-28 đến 2026-07-13, đủ OHLCV và một bản ghi trùng chính xác.
- Không nội suy qua ngày không giao dịch.

## Hợp đồng thiết kế

1. Mỗi target lưu `target_end_date_h`; train chỉ chứa nhãn kết thúc trước boundary kế tiếp.
2. Scaler, HMM, EGARCH, chọn biến và calibration chỉ fit trên train/validation phù hợp.
3. HMM xuất xác suất bằng forward recursion `P(S_t | F_t)`, không dùng smoothed probability.
4. Test là đoạn cuối thời gian và không tham gia chọn tham số.
5. Dự báo điểm, phân loại regime và mô phỏng rủi ro được đánh giá ở các bảng riêng.
6. Mọi fallback được ghi vào `reports/diagnostics/run_diagnostics.json`.

