# Ma trận nghiệm thu calibration và tail simulation

Run nghiệm thu: `configs/experimental.yaml`, dữ liệu đến 2026-07-13, seed 55. Pipeline chạy đủ 6 horizon, 400 trees, 2.000 outer-bootstrap replications, 20.000 importance paths và adaptive Monte Carlo đến giới hạn 40.000 paths. `configs/full.yaml` không được chạy và không được tuyên bố là full result.

Baseline A0 được lưu tại:

- `reports/archive/20260715T145822+0700/`
- `artifacts/archive/20260715T145822+0700/`

| Tiêu chí | Giá trị thực | Trạng thái | Bằng chứng |
|---|---:|---|---|
| h20 RMSE ≤ 0,05733 | 0,057330365 | Không đạt rất sát | `reports/tables/before_after_metrics.csv` |
| Coverage 95% trong 92,5%–97,5% | 95,2623% | Đạt | `reports/tables/interval_metrics.csv` |
| VaR 95% exceedance trong 3%–7% | 4,7377% | Đạt | `reports/tables/tail_risk_metrics.csv` |
| Coverage improvement có block-bootstrap CI > 0 | CI [5,75%; 12,18%] | Đạt | `reports/tables/outer_bootstrap_summary.csv` |
| Importance variance reduction ≥ 30% | 1,383× | Đạt | `reports/tables/importance_sampling_sensitivity.csv` |
| Importance ESS/N ≥ 20% | 80,2385% | Đạt | cùng bảng trên |
| Không làm interval score xấu đáng kể | giảm 13,62% | Đạt | `reports/tables/before_after_metrics.csv` |
| RMSE improvement có outer-bootstrap CI | new-minus-old CI [-0,01615; -0,00434] | Đạt | `reports/tables/outer_bootstrap_summary.csv` |
| Tail-head validation gates | ROC-AUC 0,512; AP 0,210 < prevalence 0,222 | Không đạt | `reports/tables/tail_head_experiment.csv` |

Tổng cộng đạt **7/9** tiêu chí. Hai failure được giữ nguyên, không làm tròn RMSE để đổi trạng thái và không đưa tail head vào production.

## Quyết định promotion

- `configs/default.yaml` giữ A0 làm baseline production.
- A1–A9 nằm trong `configs/experimental.yaml`.
- Baseline-gated center chọn alpha 0 ở cả 6 horizon. Đây là fallback protection, không phải bằng chứng ML vượt random-walk drift.
- Sequential conformal h20 chọn `regime_stratified` hoàn toàn trên validation.
- Adaptive Monte Carlo chạm 40.000 paths; batch cuối ổn định nhưng chưa đủ ba batch liên tiếp, nên stopping reason là `maximum_paths`.
- Outer bootstrap chạy quick OOS stationary-block mode. Generic full-refit/checkpoint interface đã được triển khai nhưng full mode không được chạy trong nghiệm thu này.
- Tail head bị loại bởi validation gates; multiclass production head vẫn giữ nguyên với Bear/Stress recall bằng 0.

## Kiểm thử

Leakage/maturity, finite-sample conformal, stratum fallback, likelihood ratio, ESS, known-distribution estimator, adaptive stopping, deterministic seed, stratified weighting, stationary bootstrap và delete-block jackknife đều có unit test. Kiểm tra cuối: **37 passed**, Ruff đạt, Black đạt và `git diff --check` đạt.
