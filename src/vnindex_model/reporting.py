"""Vietnamese technical and executive reports derived from saved metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _markdown(frame: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_reports(context: dict, root: str | Path = ".") -> None:
    root = Path(root)
    quality = context["quality"]
    summary = context["latest_summary"]
    comparison: pd.DataFrame = context["model_comparison"]
    per_horizon: pd.DataFrame = context["per_horizon_metrics"]
    per_class: pd.DataFrame = context["per_class_metrics"]
    calibration: pd.DataFrame = context["calibration_metrics"]
    tail: pd.DataFrame = context["tail_risk_metrics"]
    statistical: pd.DataFrame = context["statistical_tests"]
    interval: pd.DataFrame = context["interval_metrics"]
    ablation: pd.DataFrame = context["ablation"]
    jackknife: pd.DataFrame = context["jackknife"]
    seeds: pd.DataFrame = context["seed_stability"]
    importance: pd.DataFrame = context["feature_importance"]
    volatility: pd.DataFrame = context["volatility_model_comparison"]
    thresholds: pd.DataFrame = context["regime_threshold_selection"]
    monte_carlo: pd.DataFrame = context["monte_carlo_risk_summary"]
    features: pd.DataFrame = context["features"]
    best = comparison.loc[comparison.groupby("horizon")["rmse_return"].idxmin(), ["horizon", "model", "rmse_return"]]
    model_mask = comparison["model"].str.startswith(("rf_", "soft_gated", "full_"))
    best_baselines = comparison.loc[~model_mask].loc[
        comparison.loc[~model_mask].groupby("horizon")["rmse_return"].idxmin()
    ]
    evidence = []
    for row in per_horizon.itertuples():
        baseline_row = best_baselines[best_baselines["horizon"] == row.horizon].iloc[0]
        test_row = statistical[
            (statistical["horizon"] == row.horizon) & (statistical["baseline"] == baseline_row["model"])
        ]
        evidence.append(
            row.rmse_return < baseline_row["rmse_return"]
            and len(test_row) == 1
            and test_row.iloc[0]["dm_pvalue"] < 0.05
            and test_row.iloc[0]["bootstrap_ci_upper"] < 0
        )
    baseline20 = best_baselines[best_baselines["horizon"] == 20]
    if evidence and all(evidence):
        superiority = "Mô hình chính tốt hơn baseline mạnh nhất ở mọi horizon theo RMSE, DM/HAC và block-bootstrap CI"
    else:
        detail = (
            f"; ở h=20, {baseline20.iloc[0]['model']} có RMSE {baseline20.iloc[0]['rmse_return']:.6f}"
            if len(baseline20)
            else ""
        )
        superiority = f"Chưa có đủ bằng chứng để kết luận mô hình chính tốt hơn baseline mạnh nhất{detail}"
    significant = statistical[(statistical["dm_pvalue"] < 0.05) & (statistical["mean_loss_difference"] < 0)]
    returns = features["log_return"].dropna()
    h20 = per_horizon[per_horizon["horizon"] == 20].iloc[0]
    per_class20 = per_class[per_class["horizon"] == 20].set_index("class")
    calibration20 = calibration[
        (calibration["horizon"] == 20) & calibration["method"].astype(str).str.startswith("selected_test_")
    ].iloc[0]
    tail20 = tail[tail["horizon"] == 20].iloc[0]
    thresholds20 = thresholds[(thresholds["horizon"] == 20) & thresholds["selected"]].iloc[0]
    interval_best = interval.iloc[interval["coverage_error"].abs().argmin()]
    top_features = ", ".join(importance.head(3)["feature"].astype(str))
    transition = np.asarray(context["hmm_diagnostics"]["transition_matrix"])
    volatility_selected = volatility[(volatility["split"] == "validation") & volatility["selected_on_validation"]].iloc[
        0
    ]
    ablation20 = ablation[ablation["horizon"] == 20]
    best_ablation = ablation20.loc[ablation20["brier_score"].idxmin()]
    widest_jackknife = (
        jackknife.assign(width=jackknife["ci_upper"] - jackknife["ci_lower"])
        .sort_values("width", ascending=False)
        .iloc[0]
    )
    forecast = context["forecast"]
    actual_comments = {
        "01_data_splits": f"Train kết thúc {context['split_dates']['train_boundary']}, test bắt đầu sau boundary {context['split_dates']['test_boundary']}; test là đoạn cuối và không tham gia tuning.",
        "02_log_returns": f"Log-return trung bình {returns.mean():.4%}/phiên, độ lệch chuẩn {returns.std():.4%}; cực trị quan sát [{returns.min():.2%}, {returns.max():.2%}].",
        "03_return_distribution": f"Return có skewness {returns.skew():.3f} và excess kurtosis {returns.kurt():.2f}; EGARCH ước lượng ν={context['egarch_diagnostics']['nu']:.2f}, nên Student-t phản ánh tail dày hơn Gaussian.",
        "04_rolling_volatility": f"Rolling volatility 20 phiên hiện tại là {features['rolling_volatility_20'].iloc[-1]:.3%}, so với median lịch sử {features['rolling_volatility_20'].median():.3%}.",
        "05_historical_drawdown": f"Drawdown sâu nhất trong mẫu là {features['current_drawdown'].min():.2%}; drawdown tại ngày cuối là {features['current_drawdown'].iloc[-1]:.2%}.",
        "06_hmm_regimes": f"Filtered HMM chọn {context['hmm_diagnostics']['selected_states']} state, seed {context['hmm_diagnostics']['selected_seed']}; nhãn kinh tế được xếp từ return, volatility và drawdown train, không theo số state.",
        "07_filtered_probabilities": f"Entropy regime ngày cuối là {context['forecast'][['probability_bull', 'probability_sideway', 'probability_bear', 'probability_stress']].iloc[0].clip(lower=1e-12).pipe(lambda x: float(-(x * np.log(x)).sum())):.3f}; xác suất được forward-filter, không smoothed.",
        "08_transition_matrix": f"Xác suất tự chuyển lớn nhất là {np.diag(transition).max():.2%}, tương ứng expected duration xấp xỉ {max(context['hmm_diagnostics']['expected_duration']):.1f} phiên.",
        "09_egarch_volatility": f"EGARCH Student-t hội tụ={context['egarch_diagnostics']['converged']}; ν={context['egarch_diagnostics']['nu']:.2f}. Validation QLIKE chọn {volatility_selected['model']} ({volatility_selected['qlike']:.4f}).",
        "10_residual_diagnostics": f"Residual diagnostic cho Ljung–Box p={context['egarch_diagnostics']['ljung_box_pvalue']:.3g}; squared-residual p={context['egarch_diagnostics']['squared_ljung_box_pvalue']:.3g}.",
        "11_residual_acf": f"Squared-residual Ljung–Box p={context['egarch_diagnostics']['squared_ljung_box_pvalue']:.3g}; giá trị thấp sẽ là bằng chứng volatility dynamics còn sót lại.",
        "12_feature_importance": f"Ba feature impurity-importance cao nhất của return forest h=20 là {top_features}; importance không đồng nghĩa quan hệ nhân quả.",
        "13_grouped_permutation_importance": f"Grouped permutation được tính trên test với 3 lần lặp; top feature riêng lẻ có permutation delta {importance.iloc[0]['permutation_importance']:.6f} theo negative MAE.",
        "14_partial_dependence": f"Đường binned effect dùng feature {importance.iloc[0]['feature']}; đây là quan hệ dự báo có điều kiện, không phải tác động nhân quả.",
        "15_confusion_matrix": f"H=20 chọn λ hướng={thresholds20['lambda_bull']:.2f}, λ stress={thresholds20['lambda_stress']:.2f}; test có {int(per_class20.loc['Bear', 'support'])} Bear và {int(per_class20.loc['Stress', 'support'])} Stress, recall={h20['recall_bear']:.2%}/{h20['recall_stress']:.2%}.",
        "16_reliability_diagram": f"Calibration h=20 chọn {h20['calibration']}; test ECE={h20['ece']:.4f}, slope={calibration20.get('calibration_slope', np.nan):.3f}, intercept={calibration20.get('calibration_intercept', np.nan):.3f}.",
        "17_actual_predicted_return": f"H=20 RMSE return={h20['rmse_return']:.6f}, correlation={h20['return_correlation']:.3f}, directional accuracy={h20['directional_accuracy']:.2%}.",
        "18_actual_predicted_price": f"H=20 MAE price={h20['mae_price']:.2f}, RMSE price={h20['rmse_price']:.2f}, sMAPE={h20['smape_price']:.2%}.",
        "19_rolling_error": f"Mean signed error h=20={h20['mean_signed_error']:.4%}; rolling MAE cho thấy sai số tập trung theo giai đoạn thay vì ổn định hoàn toàn.",
        "20_model_comparison": f"Ở h=20, RMSE thấp nhất thuộc {best[best['horizon'] == 20].iloc[0]['model']} ({best[best['horizon'] == 20].iloc[0]['rmse_return']:.6f}). {superiority}.",
        "21_interval_coverage_width": f"Mức {interval_best['level']:.0%} gần nominal nhất nhưng coverage={interval_best['coverage']:.2%}, error={interval_best['coverage_error']:+.2%}, width={interval_best['average_width']:.4f}.",
        "22_diebold_mariano": f"Có {len(significant)}/{len(statistical)} phép DM thuận lợi và p<5% trước hiệu chỉnh multiple comparisons; điều này không đủ để thắng baseline mạnh nhất.",
        "23_ablation_study": f"Ablation h=20 có Brier thấp nhất ở {best_ablation['component']} ({best_ablation['brier_score']:.4f}); jackknife chỉ đo ổn định, không đổi point forecast.",
        "24_monte_carlo_paths": f"Hình chỉ vẽ tối đa 100 path trong {summary['number_of_paths']:,} path; mỗi path là kịch bản có điều kiện, không phải dự báo chắc chắn độc lập.",
        "25_fan_chart": f"Median terminal {summary['median_terminal_close']:.2f}; interval 95% [{forecast['lower_95'].iloc[-1]:.2f}, {forecast['upper_95'].iloc[-1]:.2f}], P(return âm)={summary['probability_negative_return']:.2%}.",
        "26_forecast_quantiles": f"Terminal mean {summary['expected_terminal_close']:.2f} và median {summary['median_terminal_close']:.2f}; chênh {summary['expected_terminal_close'] - summary['median_terminal_close']:.2f} điểm phản ánh bất đối xứng nhỏ.",
        "27_terminal_price_distribution": f"Terminal mean/median là {summary['expected_terminal_close']:.2f}/{summary['median_terminal_close']:.2f}; độ rộng interval 95% là {forecast['upper_95'].iloc[-1] - forecast['lower_95'].iloc[-1]:.2f} điểm.",
        "28_terminal_return_distribution": f"Expected/median terminal return là {summary['expected_return']:.2%}/{summary['median_return']:.2%}; P(tăng)={summary['probability_positive_return']:.2%}.",
        "29_maximum_drawdown_distribution": f"Expected/median MDD là {summary['expected_maximum_drawdown']:.2%}/{summary['median_maximum_drawdown']:.2%}; quantile xấu 5%={summary['maximum_drawdown_quantiles']['0.05']:.2%}.",
        "30_var_expected_shortfall": f"VaR 95%={summary['var_95']:.2%}, ES 95%={summary['expected_shortfall_95']:.2%}; backtest h=20 có exceedance 95%={tail20['95_var_exceedance_rate']:.2%} (Kupiec p={tail20['95_kupiec_pvalue']:.3g}).",
        "31_drawdown_probabilities": f"P(MDD vượt 3/5/7/10%) lần lượt {summary['drawdown_probabilities']['0.03']:.2%}, {summary['drawdown_probabilities']['0.05']:.2%}, {summary['drawdown_probabilities']['0.07']:.2%}, {summary['drawdown_probabilities']['0.1']:.2%}.",
        "32_forecast_regime_probabilities": f"Ở bước 20, Bull/Sideway/Bear/Stress lần lượt {forecast['probability_bull'].iloc[-1]:.2%}/{forecast['probability_sideway'].iloc[-1]:.2%}/{forecast['probability_bear'].iloc[-1]:.2%}/{forecast['probability_stress'].iloc[-1]:.2%}.",
        "33_simulation_comparison": f"ES95 giữa Student-t/bootstrap/hybrid dao động từ {monte_carlo['expected_shortfall_95'].min():.2%} đến {monte_carlo['expected_shortfall_95'].max():.2%}; hybrid dùng student weight {summary['student_weight']:.2f}.",
        "34_jackknife_sensitivity": f"Block nhạy nhất là {widest_jackknife['metric']} với CI [{widest_jackknife['ci_lower']:.4f}, {widest_jackknife['ci_upper']:.4f}]; run dùng record-level quick jackknife.",
        "35_seed_stability": f"RMSE h=20 qua {len(seeds)} seed nằm trong [{seeds['rmse_return'].min():.6f}, {seeds['rmse_return'].max():.6f}], std={seeds['rmse_return'].std():.6f}.",
        "36_latest_forecast": f"Origin {summary['forecast_origin']}; median return {summary['median_return']:.2%}, P(tăng/giảm)={summary['probability_positive_return']:.2%}/{summary['probability_negative_return']:.2%}, P(MDD>5%)={summary['drawdown_probabilities']['0.05']:.2%}.",
    }
    figures = context["figure_names"]
    commentary = []
    for index, name in enumerate(figures, start=1):
        text = actual_comments[name]
        commentary.extend([f"### Hình {index}: {name}", "", f"![{name}](figures/{name}.png)", "", text, ""])
    best_markdown = _markdown(best)
    commentary_text = "\n".join(commentary)
    report = f"""# Báo cáo mô hình VN-Index Filtered HMM – EGARCH – Regime-Aware Random Forest

## 1. Tóm tắt

Pipeline dự báo riêng lợi suất, mức điểm, trạng thái và phân phối rủi ro ở các horizon {context["horizons"]}. Dữ liệu thật kết thúc ngày {quality["end_date"]}. Kết luận so sánh: **{superiority.lower()}**.

## 2. Mục tiêu và phạm vi

Mục tiêu vận hành là 20 phiên. Dự báo điểm, phân loại trạng thái và mô phỏng tail risk là ba nhiệm vụ khác nhau; metric của chúng không được thay thế cho nhau.

## 3. Dữ liệu và chất lượng

- Nguồn: `{quality["source_file"]}`; hash `{quality["sha256"]}`.
- {quality["rows_loaded"]:,} phiên từ {quality["start_date"]} đến {quality["end_date"]}.
- Xóa {quality["duplicate_rows_removed"]} bản ghi trùng chính xác; vi phạm OHLC: {quality["ohlc_constraint_violations"]}.
- Không nội suy close qua phiên thiếu. Estimated trading dates tương lai chỉ dùng ngày làm việc gần đúng, chưa loại ngày nghỉ HOSE.

## 4. Cơ sở lý thuyết và kiến trúc

Return horizon: `R(t,h)=log(P(t+h)/P(t))`; giá suy ra `P_hat=P(t) exp(R_hat)`. Target regime dùng ngưỡng biến động quá khứ và Stress override theo forward maximum drawdown. Lambda được chọn theo class sufficiency và độ ổn định phân phối giữa train/validation; test không tham gia.

Filtered HMM được fit trên train và tính đúng `P(S_t | F_t)` bằng forward recursion; không gọi posterior smoothed trên toàn chuỗi. EGARCH(1,1) Student-t dùng shock đã chuẩn hóa phương sai. RF gồm global, HMM-feature, HMM+EGARCH và soft-gated experts. Calibration sigmoid/isotonic/none được chọn trên validation theo thời gian.

Hybrid simulation lấy shock từ hỗn hợp empirical regime-conditioned và standardized Student-t. Regime transition ở bước j là `q_ik ∝ P_ik^(1-eta_j) p_RF,k^eta_j`, sau đó chuẩn hóa.

## 5. Feature engineering và target

Feature gồm return/momentum, SMA/EMA/MACD, volatility/tail, drawdown, OHLC range, volume và lịch. Rolling feature chỉ nhìn hiện tại/quá khứ, không backfill. Feature variance/correlation được kiểm soát chỉ trên train. Mỗi horizon lưu `target_end_date_h` để purge đúng.

## 6. Validation methodology

Train 60%, validation 20%, untouched test cuối 20%; purge theo ngày kết thúc nhãn và embargo {context["embargo"]} phiên. Test không dùng để chọn K, RF, calibration hay threshold. Block bootstrap và HAC/DM đo bất định so sánh. Jackknife run này ở chế độ `{context["jackknife_mode"]}` trên OOS records, không phải bộ sinh path.

## 7. Kết quả dự báo điểm

{best_markdown}

R² được báo cáo nhưng không được diễn giải như hiệu quả giao dịch. Bảng đầy đủ nằm ở `tables/model_comparison.csv` và `tables/per_horizon_metrics.csv`.

## 8. Phân loại, calibration và tail risk

Confusion matrix, per-class recall, Brier, log loss và ECE được báo cáo riêng. Khả năng dự báo điểm tốt không bảo đảm recall Bear/Stress tốt. VaR/ES là thống kê có điều kiện theo mô hình, không phải mức lỗ tối đa.

## 9. Kiểm định thống kê và ablation

{superiority}. DM dùng HAC theo horizon; block bootstrap CI dùng các block liên tiếp. Nhiều phép so sánh làm tăng false discovery nên p-value cần được đọc thận trọng. Ablation tách HMM, EGARCH, soft gate, calibration và các simulation method; jackknife chỉ bổ sung đo ổn định, không cải thiện point forecast theo cơ chế.

## 10. Forecast mới nhất

- Origin: {summary["forecast_origin"]}; close cuối: {summary["last_observed_close"]:.2f}.
- Terminal mean/median: {summary["expected_terminal_close"]:.2f} / {summary["median_terminal_close"]:.2f}.
- Xác suất tăng/giảm: {_percent(summary["probability_positive_return"])} / {_percent(summary["probability_negative_return"])}.
- VaR 95%/99%: {_percent(summary["var_95"])} / {_percent(summary["var_99"])}.
- ES 95%/99%: {_percent(summary["expected_shortfall_95"])} / {_percent(summary["expected_shortfall_99"])}.
- Expected maximum drawdown: {_percent(summary["expected_maximum_drawdown"])}; P(MDD ≤ -5%): {_percent(summary["drawdown_probabilities"]["0.05"])}.

## 11. Phân tích 36 biểu đồ

{commentary_text}

## 12. Khả năng ứng dụng và hạn chế

Kết quả phù hợp cho stress testing và tham khảo phân phối có điều kiện, không phải khuyến nghị đầu tư. Giới hạn gồm structural break, proxy lịch giao dịch, sai số ước lượng HMM/EGARCH, sparse Stress class, calibration drift và giả định shock lịch sử còn đại diện. Không đánh đồng accuracy trạng thái với lợi nhuận chiến lược.

## 13. Hướng phát triển

Mở rộng nested model-refit bootstrap, calendar HOSE chính thức, covariate vĩ mô có timestamp kiểm chứng, conformal intervals theo regime và đánh giá live forward không tái sử dụng test.
"""
    (root / "reports/model_report.md").write_text(report, encoding="utf-8")
    executive = f"""# Tóm tắt điều hành

Mô hình ước lượng mức VN-Index, khả năng tăng/giảm, trạng thái thị trường và khoảng rủi ro cho 20 phiên sau {summary["forecast_origin"]}. Đây là phân phối kịch bản có điều kiện, không phải dự đoán chắc chắn hay khuyến nghị đầu tư.

Median terminal là **{summary["median_terminal_close"]:.2f}**, tương ứng median return **{_percent(summary["median_return"])}**. Xác suất tăng là **{_percent(summary["probability_positive_return"])}**, giảm là **{_percent(summary["probability_negative_return"])}**. VaR 95% là **{_percent(summary["var_95"])}**, ES 95% là **{_percent(summary["expected_shortfall_95"])}**, và xác suất maximum drawdown vượt 5% là **{_percent(summary["drawdown_probabilities"]["0.05"])}**.

So với baseline: {superiority.lower()}. Một mô hình có RMSE tốt hơn vẫn có thể nhận diện Bear/Stress kém hoặc tạo interval quá rộng. Kết quả cần được cập nhật khi có dữ liệu mới và không nên dùng đơn lẻ để quyết định giao dịch.
"""
    (root / "reports/executive_summary.md").write_text(executive, encoding="utf-8")
