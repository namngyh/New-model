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

## Mở rộng experimental ngày 2026-07-15

- A0 được khóa và archive trước khi sửa; không xóa các module đang hoạt động.
- `point_forecast.py` thêm validation-gated convex center với minimum-improvement và one-standard-error shrinkage.
- `conformal.py` thêm global/rolling/volatility/regime/joint sequential conformal; score chỉ mature khi `origin + horizon` đã quan sát được.
- `simulation.py` thêm proportional/equal/Neyman stratification, adaptive stopping, antithetic và control-variate utilities.
- `importance_sampling.py` dùng transition/shock proposals, log likelihood ratio, ESS và stratified importance sampling; không clipping mặc định.
- `bootstrap.py` phân biệt inner residual bootstrap với outer stationary-block record bootstrap và cung cấp full-mode checkpoint interface.
- `jackknife.py` thêm month/quarter/regime-episode/large-drawdown deletion cùng fixed-model feature-importance influence.
- `tail_head.py` thử ba weighted tail classifiers và hai auxiliary hierarchical heads với sigmoid calibration; production gate dựa hoàn toàn trên validation.
- Vì nghiệm thu chỉ đạt 7/9 tiêu chí, `default.yaml` vẫn giữ baseline và toàn bộ nâng cấp nằm ở `experimental.yaml`.
