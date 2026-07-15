# Ma trận nghiệm thu

Run nghiệm thu: `configs/default.yaml`, dữ liệu đến 2026-07-13, 10.000 path, seed chính 55. Trạng thái chỉ được coi là đạt khi file hoặc test tương ứng tồn tại.

| Nhóm | Trạng thái | Bằng chứng |
|---|---|---|
| Dữ liệu | Đạt, có 42 cảnh báo OHLC nguồn | `reports/tables/data_quality.csv`, `reports/diagnostics/ohlc_violations.csv` |
| Leakage | Đạt | `tests/test_leakage.py`, `tests/test_splits.py`; purge bằng ngày kết thúc target |
| Filtered HMM | Đạt | 5 state, seed 55; `artifacts/metadata/hmm_diagnostics.json` |
| EGARCH Student-t | Đạt, không fallback | `artifacts/metadata/egarch_diagnostics.json`, residual diagnostics |
| RF và calibration | Đạt về triển khai; chất lượng Bear/Stress yếu | per-horizon, per-class, calibration, RF tuning và lambda selection |
| Simulation | Đạt | CSV/JSON/NPZ, 10.000 hybrid paths, fan chart |
| Bootstrap và jackknife | Đạt ở quick record-level mode | statistical tests, 24 jackknife metric/block combinations |
| Baseline và ablation | Đạt; mô hình chính không thắng | model, volatility comparison và ablation |
| Báo cáo | Đạt | 36 PNG, 5 SVG, `model_report.md`, `executive_summary.md`, `README.md` |
| Tái lập | Đạt | default config, data hash, library versions, seeds và model artifacts |

Kết luận acceptance không đồng nghĩa model quality đạt mục tiêu thương mại. Random walk with drift có RMSE thấp hơn mô hình chính ở cả 6 horizon; h=20 Bear/Stress recall bằng 0 và interval 95% bị under-cover. Các failure này được giữ nguyên trong báo cáo.
