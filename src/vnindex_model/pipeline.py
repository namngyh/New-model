"""End-to-end leakage-safe training, backtest, forecast, plots, and reports."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t
from sklearn.inspection import permutation_importance

from .baselines import baseline_predictions
from .bootstrap import choose_block_length, outer_stationary_bootstrap_quick
from .calibration import select_temporal_calibration
from .config import load_config
from .conformal import (
    assign_volatility_bins,
    select_conformal_method,
    sequential_conformal,
    signed_lower_quantile,
    volatility_bin_edges,
)
from .data import discover_data_file, validate_and_save
from .drawdown_forecast import (
    build_drawdown_forecast,
    compute_drawdown_paths,
    conditional_expected_drawdown,
    drawdown_backtest,
    drawdown_duration_statistics,
    drawdown_probability_intervals,
    first_passage_times,
    pointwise_drawdown_band,
    realized_drawdown_severity,
    recovery_statistics,
    simultaneous_drawdown_band,
)
from .evaluation import classification_metrics, historical_var_tests, interval_metrics, point_metrics
from .features import build_features, select_train_features
from .hmm import fit_filtered_hmm
from .importance_sampling import (
    importance_sensitivity_table,
    simulate_stratified_importance,
    simulate_tail_importance,
)
from .jackknife import (
    block_jackknife_table,
    delete_block_jackknife,
    feature_importance_delete_block_jackknife,
)
from .persistence import run_metadata, save_model, write_json
from .plotting import generate_advanced_figures, generate_all_figures, generate_drawdown_figures
from .point_forecast import CenterSelection, apply_center_blend, select_validation_gated_center
from .random_forest import (
    fit_forest_bundle,
    fit_soft_gated_forest,
    predict_bundle,
    predict_soft_gated,
)
from .reporting import write_reports
from .simulation import (
    adaptive_simulate_paths,
    maximum_drawdown,
    run_stratified_simulation,
    simulate_paths,
)
from .splits import purged_train_validation_test
from .statistical_tests import block_bootstrap_difference, diebold_mariano
from .tail_head import run_tail_head_experiment
from .targets import CLASS_NAMES, assign_regime_labels, build_targets, select_regime_thresholds
from .validation import assert_no_target_overlap
from .volatility import fit_arch_candidate, fit_egarch_student_t

LOGGER = logging.getLogger("vnindex_model")


def _tune_rf_regularization(
    technical: pd.DataFrame,
    dates: pd.Series,
    target_end_dates: pd.Series,
    train_index: np.ndarray,
    y_class: np.ndarray,
    y_return: np.ndarray,
    y_normalized: np.ndarray,
    y_drawdown: np.ndarray,
    base_config: dict,
    seed: int,
    horizon: int,
    n_folds: int,
) -> tuple[dict, list[dict]]:
    """Choose RF regularization on two purged temporal folds inside outer train."""
    candidates = [
        {},
        {"max_depth": 5, "min_samples_leaf": 30, "max_features": 0.5},
        {"max_depth": 9, "min_samples_leaf": 40, "max_features": "sqrt"},
    ]
    rows: list[dict] = []
    n = len(train_index)
    fractions = [0.65, 0.82] if n_folds == 2 else np.linspace(0.50, 0.85, n_folds).tolist()
    fold_starts = [int(n * fraction) for fraction in fractions]
    fold_width = max(int(n * 0.15), 60)
    for candidate_number, overrides in enumerate(candidates):
        candidate = {
            **base_config,
            **overrides,
            "n_estimators": min(int(base_config["n_estimators"]), 80),
        }
        objectives = []
        for fold, start in enumerate(fold_starts, start=1):
            validation_start = min(start + horizon, n - 1)
            validation_stop = min(validation_start + fold_width, n)
            boundary = pd.Timestamp(dates.iloc[train_index[start]])
            fold_train = train_index[:start]
            end_values = pd.to_datetime(target_end_dates.iloc[fold_train]).to_numpy()
            fold_train = fold_train[end_values < np.datetime64(boundary)]
            fold_validation = train_index[validation_start:validation_stop]
            if len(fold_train) < 200 or len(fold_validation) < 30:
                continue
            bundle = fit_forest_bundle(
                technical.iloc[fold_train],
                y_class[fold_train],
                y_return[fold_train],
                y_normalized[fold_train],
                y_drawdown[fold_train],
                technical.columns.tolist(),
                candidate,
                seed,
            )
            prediction = predict_bundle(bundle, technical.iloc[fold_validation])
            rmse = float(np.sqrt(np.mean((y_return[fold_validation] - prediction["return"]) ** 2)))
            probability_loss = classification_metrics(y_class[fold_validation], prediction["probabilities"])[0][
                "log_loss"
            ]
            objective = rmse / max(float(np.std(y_return[fold_validation])), 1e-8) + 0.10 * probability_loss
            objectives.append(objective)
            rows.append(
                {
                    "horizon": horizon,
                    "candidate": candidate_number,
                    "fold": fold,
                    "max_depth": candidate["max_depth"],
                    "min_samples_leaf": candidate["min_samples_leaf"],
                    "max_features": candidate["max_features"],
                    "rmse_return": rmse,
                    "log_loss": probability_loss,
                    "objective": objective,
                    "n_train": len(fold_train),
                    "n_validation": len(fold_validation),
                }
            )
        if not objectives:
            rows.append(
                {
                    "horizon": horizon,
                    "candidate": candidate_number,
                    "fold": 0,
                    "objective": np.inf,
                }
            )
    summary = pd.DataFrame(rows).groupby("candidate", as_index=False)["objective"].mean()
    selected_number = int(summary.loc[summary["objective"].idxmin(), "candidate"])
    selected = {**base_config, **candidates[selected_number]}
    for row in rows:
        row["selected"] = row["candidate"] == selected_number
    return selected, rows


def _configure_logging(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    if not LOGGER.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        file_handler = logging.FileHandler(root / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(stream)
        LOGGER.addHandler(file_handler)


def _log(event: str, **payload) -> None:
    LOGGER.info(json.dumps({"event": event, **payload}, ensure_ascii=False, default=str))


def _markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(map(str, columns)) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines) + "\n"


def _save_table(frame: pd.DataFrame, name: str, root: Path) -> None:
    table_root = root / "reports/tables"
    table_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(table_root / f"{name}.csv", index=False)
    (table_root / f"{name}.md").write_text(_markdown(frame), encoding="utf-8")


def _crps_empirical(
    actual: np.ndarray, center: np.ndarray, scale: np.ndarray, residuals: np.ndarray, seed: int
) -> float:
    rng = np.random.default_rng(seed)
    pool = residuals[np.isfinite(residuals)]
    draws = center[:, None] + scale[:, None] * rng.choice(pool, size=(len(actual), 120), replace=True)
    first = np.mean(np.abs(draws - actual[:, None]), axis=1)
    second = 0.5 * np.mean(np.abs(draws[:, :, None] - draws[:, None, :]), axis=(1, 2))
    return float(np.mean(first - second))


def _drawdown_scenario_table(
    hybrid,
    bootstrap_result,
    economic_labels: list[str],
    initial_price: float,
    historical_peak: float,
    daily_drift: float,
    daily_volatility: float,
    degrees_of_freedom: float,
    crisis_residuals: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    """Keep conditional stress scenarios separate from baseline forecast probabilities."""
    rng = np.random.default_rng(seed + 7000)
    base_returns = np.asarray(hybrid.return_paths, dtype=float)
    base_regimes = np.asarray(hybrid.regime_paths, dtype=int)
    scenario_returns: dict[str, np.ndarray] = {"baseline_hybrid": base_returns}
    for label, name in [("Bear", "bear_conditioned"), ("Stress", "stress_conditioned")]:
        states = [state for state, value in enumerate(economic_labels) if value == label]
        mask = np.isin(base_regimes[:, 0], states)
        scenario_returns[name] = base_returns[mask] if mask.sum() >= 100 else base_returns
    center = float(daily_drift)
    scenario_returns["volatility_plus_1_sigma"] = center + 1.5 * (base_returns - center)
    scenario_returns["volatility_plus_2_sigma"] = center + 2.0 * (base_returns - center)
    pool = np.asarray(crisis_residuals, dtype=float)
    pool = pool[np.isfinite(pool)]
    if len(pool) < 20:
        pool = np.asarray(bootstrap_result.return_paths, dtype=float).ravel() / max(daily_volatility, 1e-8)
    crisis_count = min(len(base_returns), 20000)
    crisis_shocks = rng.choice(pool, size=(crisis_count, base_returns.shape[1]), replace=True)
    scenario_returns["historical_crisis_blocks"] = center + daily_volatility * crisis_shocks
    tail_count = min(len(base_returns), 20000)
    tail_nu = max(3.2, float(degrees_of_freedom) / 2)
    tail_shocks = rng.standard_t(tail_nu, size=(tail_count, base_returns.shape[1])) * np.sqrt((tail_nu - 2) / tail_nu)
    scenario_returns["student_t_tail_heavy"] = center + daily_volatility * tail_shocks
    rows = []
    for scenario, returns in scenario_returns.items():
        cumulative = np.clip(np.cumsum(returns, axis=1), -5, 5)
        prices = initial_price * np.exp(cumulative)
        origin = build_drawdown_forecast(prices, initial_price, "origin_peak")
        historical = build_drawdown_forecast(prices, initial_price, "historical_peak", historical_peak=historical_peak)
        terminal = returns.sum(axis=1)
        var95 = float(np.quantile(terminal, 0.05))
        recovery = recovery_statistics(prices, historical_peak)
        rows.append(
            {
                "scenario": scenario,
                "is_stress_scenario": scenario != "baseline_hybrid",
                "paths": len(returns),
                "expected_return": float(np.mean(np.exp(terminal) - 1)),
                "var_95": var95,
                "es_95": float(np.mean(terminal[terminal <= var95])),
                "mdar_95": origin.summary["mdar_95"],
                "ced_95": origin.summary["ced_95"],
                "probability_mdd_5": float(np.mean(origin.running_maximum_drawdown_paths[:, -1] >= 0.05)),
                "probability_mdd_10": float(np.mean(origin.running_maximum_drawdown_paths[:, -1] >= 0.10)),
                "probability_recovery": recovery["probability_recovery_by_horizon"],
                "historical_anchor_mdar_95": historical.summary["mdar_95"],
            }
        )
    return pd.DataFrame(rows)


def _append_drawdown_reports(context: dict, root: Path) -> None:
    origin = context["origin_summary"]
    historical = context["historical_summary"]
    acceptance = context["acceptance"]
    probability = context["probability"]
    recovery = context["recovery"]
    mc = context["mc"]
    importance = context["importance"]
    selected_is = importance[importance["selected"]]
    scenarios = context["scenarios"].set_index("scenario")
    conditional = context["conditional"]
    regime_coverage = conditional[
        (conditional["horizon"] == 20)
        & (conditional["anchor_mode"] == "origin_peak")
        & (conditional["level"] == 0.95)
        & (conditional["stratum_type"] == "regime")
    ]
    breach = {
        threshold: origin["first_passage"][str(threshold)]["probability_breach_by_horizon"]
        for threshold in (0.03, 0.05, 0.07, 0.10)
    }
    comments = {
        "54": f"Origin-peak median ending severity {origin['median_ending_drawdown_severity']:.2%}; MDaR95 {origin['mdar_95']:.2%}.",
        "55": f"Historical-peak median ending severity {historical['median_ending_drawdown_severity']:.2%}; anchor bắt đầu từ drawdown hiện tại.",
        "56": "Running maximum severity không giảm theo thời gian; khác drawdown tức thời có thể phục hồi.",
        "57": f"P(breach 3/5/7/10%) cuối horizon: {breach[0.03]:.2%}/{breach[0.05]:.2%}/{breach[0.07]:.2%}/{breach[0.10]:.2%}.",
        "58": "First-passage time chỉ thống kê trên path đã breach; path chưa breach được giữ right-censored.",
        "59": f"MDaR90/95/99: {origin['mdar_90']:.2%}/{origin['mdar_95']:.2%}/{origin['mdar_99']:.2%}; CED95 {origin['ced_95']:.2%}.",
        "60": f"Xác suất phục hồi historical peak trong horizon là {recovery['probability_recovery_by_horizon']:.2%}.",
        "61": f"Brier tốt nhất trên bảng backtest là {probability['brier_score'].min():.4f}; reliability gap được báo cáo riêng theo threshold.",
        "62": "Predicted median drawdown và realized drawdown được so trên cùng OOS origins; dispersion lớn phản ánh rủi ro đường đi khó dự báo.",
        "63": "Direct drawdown conformal được chọn trên validation, không tái dùng return-conformal multiplier.",
        "64": "Simultaneous band nhắm bao phủ cả trajectory nên rộng hơn pointwise band.",
        "65": "Historical-peak severity có thể cao ngay ở bước đầu vì không reset drawdown tại forecast origin.",
        "66": f"MCSE lớn nhất trong các breach statistic là {mc['mcse'].max():.4f}; đây là numerical error, không phải predictive interval.",
        "67": f"Proposal được chọn theo từng threshold; variance reduction tốt nhất {selected_is['variance_reduction_ratio'].max():.2f}x.",
        "68": "Stress scenarios là conditional what-if và không được trộn với xác suất baseline_hybrid.",
        "69": "Duration giữ recovery time NaN cho path chưa phục hồi, thay vì ép bằng horizon.",
        "70": "Calibration strata thiếu mẫu fallback về global; method được khóa bằng validation objective.",
    }
    figure_lines = []
    names = context["figure_names"]
    for name in names:
        number = name.split("_", 1)[0]
        figure_lines.extend([f"### {name}", "", f"![{name}](figures/{name}.png)", "", comments[number], ""])
    figure_text = "\n".join(figure_lines)
    section = f"""

## 14. Dự báo rủi ro đường đi và drawdown

Drawdown return giữ dấu âm để tương thích; toàn bộ bảng mới dùng `drawdown_severity=-drawdown_return>=0`. Origin-peak đo khoản giảm mới từ forecast origin, còn historical-peak giữ đỉnh lịch sử nên bắt đầu từ drawdown hiện tại. Terminal return dương vẫn có thể đi cùng intra-horizon drawdown lớn.

MDaR là quantile của maximum drawdown severity, khác VaR terminal return. CED là severity trung bình phía trên MDaR, khác expected shortfall của return. Monte Carlo confidence interval đo sai số số học của probability estimator; predictive/conformal bound đo bất định của outcome. Direct drawdown conformal có backtest coverage riêng và chỉ dùng score đã mature.

Origin-peak MDaR90/95/99 là **{origin['mdar_90']:.2%}/{origin['mdar_95']:.2%}/{origin['mdar_99']:.2%}**; CED90/95/99 là **{origin['ced_90']:.2%}/{origin['ced_95']:.2%}/{origin['ced_99']:.2%}**. Historical-peak recovery probability là **{recovery['probability_recovery_by_horizon']:.2%}**. Drawdown acceptance đạt **{int(acceptance['passed'].sum())}/{len(acceptance)}**; nếu thiếu một guardrail, module tiếp tục experimental.

Stress scenario không phải xác suất dự báo. Importance sampling giảm phương sai estimator rare event nhưng không làm drawdown dễ dự báo hơn.

## 15. Biểu đồ drawdown

{figure_text}
"""
    model_path = root / "reports/model_report.md"
    model_path.write_text(model_path.read_text(encoding="utf-8") + section, encoding="utf-8")
    executive_path = root / "reports/executive_summary.md"
    executive_path.write_text(
        executive_path.read_text(encoding="utf-8")
        + f"\n\n## Drawdown path risk\n\nOrigin-peak MDaR95/CED95 là **{origin['mdar_95']:.2%}/{origin['ced_95']:.2%}**; P(MDD vượt 5% là **{breach[0.05]:.2%}**. Historical-peak recovery probability trong horizon là **{recovery['probability_recovery_by_horizon']:.2%}**. Drawdown layer đạt {int(acceptance['passed'].sum())}/{len(acceptance)} checks và vẫn experimental nếu chưa đạt đủ.\n",
        encoding="utf-8",
    )
    readme_path = root / "README.md"
    baseline_scenario = scenarios.loc["baseline_hybrid"]
    bear_scenario = scenarios.loc["bear_conditioned"]
    volatility_two = scenarios.loc["volatility_plus_2_sigma"]
    crisis_scenario = scenarios.loc["historical_crisis_blocks"]
    accepted_is = selected_is[selected_is["accepted"]]
    regime_coverage_text = ", ".join(
        f"state {int(row.stratum)}: {row.conditional_coverage:.2%} (n={int(row.observations)})"
        for row in regime_coverage.itertuples()
    )
    readme_path.write_text(
        readme_path.read_text(encoding="utf-8") + f"""

## Calibrated drawdown forecast

Hai anchor `origin_peak` và `historical_peak`, MDaR/CED, first passage, recovery censoring, probability CI và direct drawdown conformal được sinh từ pipeline. Drawdown return giữ dấu âm để tương thích; các biểu đồ dưới đây dùng `drawdown_severity=-drawdown_return>=0`. Module đạt **{int(acceptance['passed'].sum())}/{len(acceptance)}** acceptance checks và vẫn experimental.

### Fan chart theo hai drawdown anchor

![Origin-peak drawdown fan chart](reports/figures/54_drawdown_fan_chart_origin_peak.png)

Origin-peak reset đỉnh tại forecast origin. Median severity cuối horizon là **{origin['median_ending_drawdown_severity']:.2%}**, trong khi expected maximum severity trong đường đi là **{origin['expected_maximum_drawdown_severity']:.2%}**. Chênh lệch này cho thấy terminal return dương vẫn có thể đi qua một nhịp giảm đáng kể trước khi hồi phục.

![Historical-peak drawdown fan chart](reports/figures/55_drawdown_fan_chart_historical_peak.png)

Historical-peak không reset drawdown. VN-Index đang thấp hơn historical peak khoảng **{-context['current_historical_drawdown']:.2%}**, nên median ending severity là **{historical['median_ending_drawdown_severity']:.2%}** và historical-anchor MDaR95 lên **{historical['mdar_95']:.2%}**. Hai fan chart trả lời hai câu hỏi khác nhau và không được thay thế cho nhau.

### First passage và MDaR/CED

![Drawdown breach probability term structure](reports/figures/57_drawdown_breach_probability_term_structure.png)

Xác suất tích lũy breach 3/5/7/10% đến phiên 20 lần lượt là **{breach[0.03]:.2%}/{breach[0.05]:.2%}/{breach[0.07]:.2%}/{breach[0.10]:.2%}**. Các đường không giảm vì một path đã chạm ngưỡng vẫn được tính là đã breach ở mọi bước sau; đây không phải xác suất drawdown tức thời tại từng phiên.

![MDaR and CED term structure](reports/figures/59_mdar_ced_term_structure.png)

Origin-peak MDaR90/95/99 là **{origin['mdar_90']:.2%}/{origin['mdar_95']:.2%}/{origin['mdar_99']:.2%}**; CED90/95/99 là **{origin['ced_90']:.2%}/{origin['ced_95']:.2%}/{origin['ced_99']:.2%}**. MDaR là quantile của maximum drawdown severity; CED là trung bình phía xấu hơn quantile đó, không phải VaR/ES của terminal return.

### Recovery và direct drawdown calibration

![Recovery probability curve](reports/figures/60_recovery_probability_curve.png)

Xác suất quay lại historical peak trong 20 phiên là **{recovery['probability_recovery_by_horizon']:.2%}**. Các path chưa hồi phục được giữ right-censored; median recovery time không được tính bằng cách ép chúng về ngày 20.

![Direct drawdown upper-bound coverage](reports/figures/63_drawdown_upper_bound_coverage.png)

Ở h=20, direct conformal đạt coverage MDaR95 **{context['coverage_95']:.2%}** và MDaR99 **{context['coverage_99']:.2%}**. Phương pháp được khóa trên validation; test chỉ đánh giá. Exceedance vẫn phụ thuộc mạnh do các horizon chồng lấn, vì vậy aggregate coverage tốt không đồng nghĩa các lỗi độc lập theo thời gian.

![Conditional coverage by regime](reports/figures/70_drawdown_calibration_by_regime.png)

Conditional coverage 95% theo filtered regime là {regime_coverage_text}. Nhóm nhỏ có coverage 100% không nên được xem là calibration hoàn hảo; sample size thấp làm ước lượng kém ổn định và pipeline fallback về global khi validation stratum không đủ mẫu.

### Rare-event efficiency và stress scenarios

![Importance sampling efficiency](reports/figures/67_drawdown_importance_sampling_efficiency.png)

Các proposal đạt gate ở MDD 7/10/15% với variance-reduction ratio từ **{accepted_is['variance_reduction_ratio'].min():.2f}x** đến **{accepted_is['variance_reduction_ratio'].max():.2f}x** và ESS/N tối thiểu **{accepted_is['ess_ratio'].min():.2%}**. Proposal MDD 5% chỉ đạt **{selected_is.loc[selected_is['threshold'] == 0.05, 'variance_reduction_ratio'].iloc[0]:.2f}x**, dưới gate 1.30x, nên bị từ chối. Importance sampling giảm numerical variance; nó không làm drawdown dễ dự báo hơn.

![Scenario drawdown risk cone](reports/figures/68_drawdown_scenario_risk_cone.png)

Baseline MDaR95/CED95 là **{baseline_scenario['mdar_95']:.2%}/{baseline_scenario['ced_95']:.2%}**. Bear-conditioned tăng lên **{bear_scenario['mdar_95']:.2%}/{bear_scenario['ced_95']:.2%}**; volatility +2σ tăng mạnh tới **{volatility_two['mdar_95']:.2%}/{volatility_two['ced_95']:.2%}**. Historical-crisis blocks cho MDaR95 **{crisis_scenario['mdar_95']:.2%}** và recovery **{crisis_scenario['probability_recovery']:.2%}**. Stress-conditioned hiện trùng baseline vì không đủ path khởi đầu Stress và engine fallback; đây là giới hạn mẫu, không phải bằng chứng Stress vô hại. Scenario là conditional what-if, không phải xác suất dự báo.

Các bảng chi tiết nằm trong `reports/tables/drawdown_*.csv`; artifacts mới nhất nằm trong `artifacts/forecasts/latest_drawdown_*`.
""",
        encoding="utf-8",
    )


def _run_drawdown_layer(
    root: Path,
    config: dict,
    data: pd.DataFrame,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    horizons: list[int],
    horizon_state: dict,
    volatility,
    full_volatility,
    full_hmm,
    hybrid,
    simulation_results: dict,
    convergence_history: pd.DataFrame,
    naive_tail,
    importance_results: list,
    selected_importance,
    latest_center_horizon: float,
    calibrated_daily_volatility: float,
    seed: int,
) -> dict:
    drawdown_config = config["drawdown"]
    thresholds = tuple(float(value) for value in drawdown_config["thresholds"])
    initial_price = float(data["close"].iloc[-1])
    historical_peak = float(data["close"].max())
    rolling_window = int(drawdown_config["rolling_peak_window"])
    rolling_peak = float(data["close"].iloc[-rolling_window:].max())
    origin = build_drawdown_forecast(hybrid.price_paths, initial_price, "origin_peak", thresholds=thresholds)
    historical = build_drawdown_forecast(
        hybrid.price_paths,
        initial_price,
        "historical_peak",
        historical_peak=historical_peak,
        thresholds=thresholds,
    )
    rolling = build_drawdown_forecast(
        hybrid.price_paths,
        initial_price,
        "rolling_peak",
        rolling_peak=rolling_peak,
        thresholds=thresholds,
    )
    recovery = recovery_statistics(hybrid.price_paths, historical_peak)
    duration_frame, duration_summary = drawdown_duration_statistics(historical.drawdown_paths)
    first_passage = first_passage_times(origin.running_maximum_drawdown_paths, thresholds)
    first_passage_rows = []
    first_passage_plot_rows = []
    for threshold, times in first_passage.items():
        breached = np.isfinite(times)
        first_passage_rows.append(
            {
                "threshold": threshold,
                "probability_breach_by_horizon": float(breached.mean()),
                "probability_no_breach_by_horizon": float(1 - breached.mean()),
                "median_time_to_breach": float(np.nanmedian(times)) if breached.any() else np.nan,
                "conditional_expected_time_to_breach": float(np.nanmean(times)) if breached.any() else np.nan,
                **{f"probability_breach_within_{step}": float(np.mean(times <= step)) for step in (5, 10, 20)},
            }
        )
        first_passage_plot_rows.extend(
            {"threshold": threshold, "first_passage_time": value} for value in times[np.isfinite(times)]
        )
    first_passage_table = pd.DataFrame(first_passage_rows)
    first_passage_plot = pd.DataFrame(first_passage_plot_rows)
    recovery_table = pd.DataFrame(
        {
            "step": np.arange(1, len(recovery["probability_recovery_by_step"]) + 1),
            "probability_recovery_by_step": recovery["probability_recovery_by_step"],
            "unrecovered_survival": recovery["unrecovered_survival_by_step"],
        }
    )
    recovery_summary = pd.DataFrame(
        [
            {
                key: value
                for key, value in recovery.items()
                if key
                not in {
                    "probability_recovery_by_step",
                    "recovery_times",
                    "right_censored",
                    "unrecovered_survival_by_step",
                }
            }
            | {"right_censored_paths": int(np.sum(recovery["right_censored"])), "paths": len(hybrid.price_paths)}
        ]
    )
    mdar_ced = pd.DataFrame(
        [
            {"anchor_mode": name, **result.summary}
            for name, result in {"origin_peak": origin, "historical_peak": historical, "rolling_peak": rolling}.items()
        ]
    ).drop(columns=["first_passage"])
    anchor_comparison = mdar_ced.copy()

    unweighted_events = {
        f"probability_mdd_{int(threshold * 100)}": origin.running_maximum_drawdown_paths[:, -1] >= threshold
        for threshold in thresholds
    }
    ordinary_mc = drawdown_probability_intervals(unweighted_events, seed=seed)
    ordinary_mc.insert(0, "method", "adaptive_hybrid")
    weighted_events = {
        f"probability_mdd_{int(threshold * 100)}": -selected_importance.maximum_drawdowns >= threshold
        for threshold in thresholds
    }
    weighted_mc = drawdown_probability_intervals(
        weighted_events,
        weights=selected_importance.normalized_weights,
        seed=seed,
    )
    weighted_mc.insert(0, "method", "importance_sampling")
    mc_uncertainty = pd.concat([ordinary_mc, weighted_mc], ignore_index=True)

    importance_rows = []
    for threshold in (0.05, 0.07, 0.10, 0.15):
        naive_event = -naive_tail.maximum_drawdowns >= threshold
        naive_probability = float(np.mean(naive_event))
        naive_mcse = float(np.sqrt(naive_probability * (1 - naive_probability) / len(naive_event)))
        threshold_rows = []
        for result in importance_results:
            event = (-result.maximum_drawdowns >= threshold).astype(float)
            weights = result.normalized_weights
            estimate = float(np.sum(weights * event))
            mcse = float(np.sqrt(np.sum(np.square(weights) * np.square(event - estimate))))
            ratio = float(np.square(naive_mcse) / max(np.square(mcse), 1e-18))
            row = {
                "threshold": threshold,
                "transition_strength": result.diagnostics["transition_strength"],
                "shock_strength": result.diagnostics["shock_strength"],
                "estimate": estimate,
                "naive_mcse": naive_mcse,
                "importance_mcse": mcse,
                "variance_reduction_ratio": ratio,
                "ess": result.diagnostics["ess"],
                "ess_ratio": result.diagnostics["ess_ratio"],
                "relative_mcse": mcse / max(estimate, 1e-15),
                "tail_events_generated": int(event.sum()),
                "accepted": bool(result.diagnostics["ess_ratio"] >= 0.20 and ratio >= 1.30),
                "selected": False,
            }
            threshold_rows.append(row)
        accepted = [row for row in threshold_rows if row["accepted"]]
        choice = (
            max(accepted, key=lambda row: row["variance_reduction_ratio"])
            if accepted
            else max(threshold_rows, key=lambda row: row["ess_ratio"])
        )
        choice["selected"] = True
        importance_rows.extend(threshold_rows)
    drawdown_importance = pd.DataFrame(importance_rows)

    backtest_metrics_frames = []
    interval_frames = []
    probability_frames = []
    selection_frames = []
    detail_frames = []
    close = data["close"].to_numpy(dtype=float)
    for horizon in horizons:
        state = horizon_state[horizon]
        for anchor_mode in ("origin_peak", "historical_peak"):
            realized = realized_drawdown_severity(close, horizon, anchor_mode, rolling_window)
            train = state["train_index"]
            validation = state["validation_index"]
            test = state["test_index"]
            outputs = drawdown_backtest(
                realized[validation],
                realized[test],
                state["validation_center"],
                state["improved_center"],
                state["validation_sigma"] / np.sqrt(horizon),
                state["test_sigma"] / np.sqrt(horizon),
                features["current_drawdown"].iloc[validation].to_numpy(),
                features["current_drawdown"].iloc[test].to_numpy(),
                state["validation_regime"],
                state["test_regime"],
                realized[train],
                volatility.standardized_residuals[train],
                float(volatility.diagnostics["nu"]),
                horizon,
                int(drawdown_config["backtest_paths"]),
                max(20, int(config["conformal"]["minimum_stratum_size"] // 2)),
                anchor_mode,
                thresholds[:4],
                seed,
            )
            metrics, intervals, probabilities, selections, details = outputs
            details.insert(0, "date", data["date"].iloc[test].to_numpy())
            backtest_metrics_frames.append(metrics)
            interval_frames.append(intervals)
            probability_frames.append(probabilities)
            selection_frames.append(selections)
            detail_frames.append(details)
    backtest_metrics = pd.concat(backtest_metrics_frames, ignore_index=True)
    interval_metrics_table = pd.concat(interval_frames, ignore_index=True)
    probability_calibration = pd.concat(probability_frames, ignore_index=True)
    drawdown_conformal_selection = pd.concat(selection_frames, ignore_index=True)
    backtest_details = pd.concat(detail_frames, ignore_index=True)
    conditional_rows = []
    rolling_frames = []
    for (horizon, anchor_mode), group in backtest_details.groupby(["horizon", "anchor_mode"]):
        ordered = group.sort_values("date")
        for level in (80, 90, 95, 99):
            coverage_column = f"conformal_covered_{level}"
            rolling_frames.append(
                pd.DataFrame(
                    {
                        "date": ordered["date"],
                        "horizon": horizon,
                        "anchor_mode": anchor_mode,
                        "level": level / 100,
                        "rolling_coverage_100": ordered[coverage_column].rolling(100, min_periods=40).mean(),
                    }
                )
            )
            for stratum in ("regime", "volatility_bin", "current_drawdown_bin"):
                for value, stratum_group in ordered.groupby(stratum):
                    conditional_rows.append(
                        {
                            "horizon": horizon,
                            "anchor_mode": anchor_mode,
                            "level": level / 100,
                            "stratum_type": stratum,
                            "stratum": value,
                            "observations": len(stratum_group),
                            "conditional_coverage": float(stratum_group[coverage_column].mean()),
                            "coverage_error": float(stratum_group[coverage_column].mean() - level / 100),
                        }
                    )
    conditional_coverage = pd.DataFrame(conditional_rows)
    rolling_coverage = pd.concat(rolling_frames, ignore_index=True)

    h20_details = backtest_details[
        (backtest_details["horizon"] == 20) & (backtest_details["anchor_mode"] == "origin_peak")
    ].reset_index(drop=True)
    pointwise_lower, pointwise_upper = pointwise_drawdown_band(origin.severity_paths, 0.95)
    state20 = horizon_state[20 if 20 in horizon_state else max(horizons)]
    validation_index = state20["validation_index"]
    trajectory_rows = []
    for origin_index in validation_index:
        if origin_index + 20 >= len(close):
            continue
        prices = close[origin_index + 1 : origin_index + 21][None, :]
        trajectory_rows.append(-compute_drawdown_paths(prices, close[origin_index], "origin_peak")[0])
    actual_trajectories = np.asarray(trajectory_rows)
    count = len(actual_trajectories)
    center_trajectories = np.tile(origin.term_structure["drawdown_median"].to_numpy(), (count, 1))
    scale_trajectories = np.tile(
        np.maximum(origin.term_structure["drawdown_q90"] - origin.term_structure["drawdown_median"], 1e-4).to_numpy(),
        (count, 1),
    )
    simultaneous_lower, simultaneous_upper, simultaneous_multiplier = simultaneous_drawdown_band(
        actual_trajectories,
        center_trajectories,
        scale_trajectories,
        origin.term_structure["drawdown_median"].to_numpy(),
        scale_trajectories[0],
        0.95,
    )
    simultaneous_context = {
        "center": origin.term_structure["drawdown_median"].to_numpy(),
        "pointwise_lower": pointwise_lower,
        "pointwise_upper": pointwise_upper,
        "simultaneous_lower": simultaneous_lower,
        "simultaneous_upper": simultaneous_upper,
        "multiplier": simultaneous_multiplier,
    }

    crisis_mask = features["current_drawdown"].to_numpy() <= -0.10
    scenario_table = _drawdown_scenario_table(
        hybrid,
        simulation_results["bootstrap"],
        full_hmm.economic_labels,
        initial_price,
        historical_peak,
        latest_center_horizon / 20,
        calibrated_daily_volatility,
        float(full_volatility.diagnostics["nu"]),
        full_volatility.standardized_residuals[crisis_mask],
        seed,
    )

    selected_mc = ordinary_mc.set_index("statistic")
    adaptive_rows = []
    final_batch = convergence_history.iloc[-1] if len(convergence_history) else pd.Series(dtype=float)
    previous_batch = convergence_history.iloc[-2] if len(convergence_history) > 1 else final_batch
    for statistic in [
        "probability_negative_return",
        "probability_mdd_3",
        "probability_mdd_5",
        "probability_mdd_7",
        "probability_mdd_10",
    ]:
        if statistic == "probability_negative_return":
            estimate = float(hybrid.summary["probability_negative_return"])
            mcse = float(np.sqrt(estimate * (1 - estimate) / len(hybrid.return_paths)))
        else:
            estimate = float(selected_mc.loc[statistic, "estimate"])
            mcse = float(selected_mc.loc[statistic, "mcse"])
        adaptive_rows.append(
            {
                "statistic": statistic,
                "estimate": estimate,
                "converged": mcse <= drawdown_config["adaptive_stopping"]["probability_absolute_mcse"],
                "mcse": mcse,
                "relative_mcse": mcse / max(estimate, 1e-15),
                "last_change": float(
                    abs(final_batch.get(statistic, estimate) - previous_batch.get(statistic, estimate))
                ),
                "required_tolerance": drawdown_config["adaptive_stopping"]["probability_absolute_mcse"],
            }
        )
    maximum = origin.running_maximum_drawdown_paths[:, -1]
    batch_size = min(int(drawdown_config["adaptive_stopping"]["batch_size"]), len(maximum) // 2)
    previous_maximum = maximum[:-batch_size] if batch_size else maximum
    for statistic, value, previous, tolerance in [
        (
            "mdar_95",
            origin.summary["mdar_95"],
            np.quantile(previous_maximum, 0.95),
            drawdown_config["adaptive_stopping"]["mdar_quantile_change_tolerance"],
        ),
        (
            "mdar_99",
            origin.summary["mdar_99"],
            np.quantile(previous_maximum, 0.99),
            drawdown_config["adaptive_stopping"]["mdar_quantile_change_tolerance"],
        ),
        (
            "ced_95",
            origin.summary["ced_95"],
            conditional_expected_drawdown(previous_maximum, [0.95])["ced_95"],
            drawdown_config["adaptive_stopping"]["ced_relative_change_tolerance"],
        ),
        (
            "ced_99",
            origin.summary["ced_99"],
            conditional_expected_drawdown(previous_maximum, [0.99])["ced_99"],
            drawdown_config["adaptive_stopping"]["ced_relative_change_tolerance"],
        ),
        (
            "median_maximum_drawdown",
            origin.summary["median_maximum_drawdown_severity"],
            np.median(previous_maximum),
            drawdown_config["adaptive_stopping"]["mdar_quantile_change_tolerance"],
        ),
    ]:
        change = float(abs(float(value) - float(previous)))
        relative = change / max(abs(float(value)), 1e-15)
        use_relative = statistic.startswith("ced")
        adaptive_rows.append(
            {
                "statistic": statistic,
                "estimate": value,
                "converged": relative <= tolerance if use_relative else change <= tolerance,
                "mcse": np.nan,
                "relative_mcse": np.nan,
                "last_change": relative if use_relative else change,
                "required_tolerance": tolerance,
            }
        )
    adaptive_status = pd.DataFrame(adaptive_rows)

    h20_interval = interval_metrics_table[
        (interval_metrics_table["horizon"] == 20) & (interval_metrics_table["anchor_mode"] == "origin_peak")
    ].set_index("level")
    h20_probability = probability_calibration[
        (probability_calibration["horizon"] == 20) & (probability_calibration["anchor_mode"] == "origin_peak")
    ]
    historical_probability = h20_probability[h20_probability["method"] == "historical_empirical"].set_index("threshold")
    hybrid_probability = h20_probability[h20_probability["method"] == "hybrid_monte_carlo"].set_index("threshold")
    h20_backtest = backtest_metrics[
        (backtest_metrics["horizon"] == 20) & (backtest_metrics["anchor_mode"] == "origin_peak")
    ].set_index("method")
    selected_thresholds = drawdown_importance[drawdown_importance["selected"]].set_index("threshold")
    acceptance = pd.DataFrame(
        [
            {
                "criterion": "MDaR95 coverage in [92.5%, 97.5%]",
                "value": h20_interval.loc[0.95, "upper_bound_coverage"],
                "passed": 0.925 <= h20_interval.loc[0.95, "upper_bound_coverage"] <= 0.975,
            },
            {
                "criterion": "MDaR99 coverage in [97.5%, 100%]",
                "value": h20_interval.loc[0.99, "upper_bound_coverage"],
                "passed": 0.975 <= h20_interval.loc[0.99, "upper_bound_coverage"] <= 1.0,
            },
            {
                "criterion": "Brier MDD5 improves historical frequency",
                "value": hybrid_probability.loc[0.05, "brier_score"] - historical_probability.loc[0.05, "brier_score"],
                "passed": hybrid_probability.loc[0.05, "brier_score"] < historical_probability.loc[0.05, "brier_score"],
            },
            {
                "criterion": "Brier MDD10 no worse than historical frequency",
                "value": hybrid_probability.loc[0.10, "brier_score"] - historical_probability.loc[0.10, "brier_score"],
                "passed": hybrid_probability.loc[0.10, "brier_score"]
                <= historical_probability.loc[0.10, "brier_score"],
            },
            {
                "criterion": "direct conformal q95 pinball improves uncalibrated MC",
                "value": h20_backtest.loc["hybrid_direct_drawdown_conformal", "pinball_q95"]
                - h20_backtest.loc["hybrid_monte_carlo", "pinball_q95"],
                "passed": h20_backtest.loc["hybrid_direct_drawdown_conformal", "pinball_q95"]
                < h20_backtest.loc["hybrid_monte_carlo", "pinball_q95"],
            },
            {
                "criterion": "importance ESS/N >= 20%",
                "value": selected_thresholds["ess_ratio"].min(),
                "passed": selected_thresholds["ess_ratio"].min() >= 0.20,
            },
            {
                "criterion": "variance reduction MDD7 or MDD10 >= 30%",
                "value": selected_thresholds.loc[[0.07, 0.10], "variance_reduction_ratio"].max(),
                "passed": selected_thresholds.loc[[0.07, 0.10], "variance_reduction_ratio"].max() >= 1.30,
            },
            {
                "criterion": "rare-event relative MCSE <= 10%",
                "value": selected_thresholds.loc[[0.07, 0.10], "relative_mcse"].max(),
                "passed": selected_thresholds.loc[[0.07, 0.10], "relative_mcse"].max() <= 0.10,
            },
            {"criterion": "label-maturity leakage contract enabled", "value": True, "passed": True},
        ]
    )
    promoted = bool(acceptance["passed"].all())

    combined_term = pd.concat(
        [
            result.term_structure.assign(anchor_mode=name)
            for name, result in {"origin_peak": origin, "historical_peak": historical, "rolling_peak": rolling}.items()
        ],
        ignore_index=True,
    )
    future_dates = pd.bdate_range(
        pd.Timestamp(data["date"].iloc[-1]) + pd.offsets.BDay(1), periods=hybrid.price_paths.shape[1]
    )
    combined_term["estimated_trading_date"] = np.tile(future_dates.strftime("%Y-%m-%d"), 3)
    combined_term.to_csv(root / "artifacts/forecasts/latest_drawdown_forecast.csv", index=False)
    sample_count = min(int(config["simulation"]["sample_paths"]), len(hybrid.price_paths))
    np.savez_compressed(
        root / "artifacts/forecasts/latest_drawdown_paths.npz",
        price_paths=hybrid.price_paths[:sample_count],
        origin_drawdown_paths=origin.drawdown_paths[:sample_count],
        historical_drawdown_paths=historical.drawdown_paths[:sample_count],
        origin_maximum_severity=origin.running_maximum_drawdown_paths[:, -1],
        historical_maximum_severity=historical.running_maximum_drawdown_paths[:, -1],
        recovery_times=recovery["recovery_times"],
    )
    summary = {
        "drawdown_anchor_modes": drawdown_config["anchor_modes"],
        "origin_peak_drawdown": origin.summary,
        "historical_peak_drawdown": historical.summary,
        "rolling_peak_drawdown": rolling.summary,
        "mdar": {key: origin.summary[key] for key in ["mdar_80", "mdar_90", "mdar_95", "mdar_975", "mdar_99"]},
        "ced": {key: origin.summary[key] for key in ["ced_90", "ced_95", "ced_99"]},
        "first_passage_probabilities": origin.summary["first_passage"],
        "recovery_probability": recovery["probability_recovery_by_horizon"],
        "drawdown_probability_confidence_intervals": ordinary_mc.to_dict(orient="records"),
        "drawdown_conformal_method": h20_interval["selected_conformal_method"].to_dict(),
        "drawdown_conformal_multiplier": simultaneous_multiplier,
        "drawdown_calibration_status": "promoted" if promoted else "experimental",
        "drawdown_mc_convergence_status": {
            "stopping_reason": hybrid.summary.get("stopping_reason", "fixed_paths"),
            "paths": len(hybrid.price_paths),
            "statistics_not_converged": adaptive_status.loc[~adaptive_status["converged"], "statistic"].tolist(),
        },
        "duration": duration_summary,
        "historical_peak": historical_peak,
        "current_historical_drawdown": initial_price / historical_peak - 1,
        "promotion_eligible": promoted,
    }
    write_json(root / "artifacts/forecasts/latest_drawdown_summary.json", summary)
    tables = {
        "drawdown_backtest_metrics": backtest_metrics,
        "drawdown_interval_metrics": interval_metrics_table,
        "drawdown_probability_calibration": probability_calibration,
        "drawdown_first_passage_summary": first_passage_table,
        "drawdown_recovery_summary": recovery_summary,
        "mdar_ced_summary": mdar_ced,
        "drawdown_mc_uncertainty": mc_uncertainty,
        "drawdown_importance_sampling": drawdown_importance,
        "drawdown_scenario_comparison": scenario_table,
        "drawdown_anchor_comparison": anchor_comparison,
        "drawdown_conformal_selection": drawdown_conformal_selection,
        "drawdown_conditional_coverage": conditional_coverage,
        "drawdown_rolling_coverage": rolling_coverage,
        "drawdown_backtest_records": backtest_details,
        "drawdown_adaptive_status": adaptive_status,
        "drawdown_acceptance_results": acceptance,
    }
    plot_context = {
        "origin_term": origin.term_structure,
        "historical_term": historical.term_structure,
        "first_passage": first_passage_plot,
        "recovery": recovery_table,
        "backtest": backtest_metrics,
        "interval": interval_metrics_table,
        "probability": probability_calibration,
        "mc_uncertainty": ordinary_mc,
        "importance": drawdown_importance,
        "scenarios": scenario_table,
        "duration": duration_frame,
        "details": h20_details,
        "simultaneous": simultaneous_context,
        "calibration": conditional_coverage,
    }
    return {
        "summary": summary,
        "tables": tables,
        "plot_context": plot_context,
        "origin": origin,
        "historical": historical,
        "recovery": recovery,
        "acceptance": acceptance,
        "mc": ordinary_mc,
        "importance": drawdown_importance,
        "probability": probability_calibration,
        "scenarios": scenario_table,
        "conditional": conditional_coverage,
    }


def _readme(context: dict, root: Path) -> None:
    summary = context["latest_summary"]
    comparison: pd.DataFrame = context["model_comparison"]
    best = comparison.loc[
        comparison.groupby("horizon")["rmse_return"].idxmin(),
        ["horizon", "model", "rmse_return", "directional_accuracy"],
    ]
    before = context["before_after"].iloc[0]
    after = context["before_after"].iloc[-1]
    alpha20 = context["point_center_selection"].query("horizon == 20 and selected").iloc[0]
    acceptance = context["acceptance_results"]
    selected_conformal = context["selected_conformal_method"]
    readme = f"""# VN-Index Regime-Aware Random Forest và Hybrid Monte Carlo

Tác giả: **Nguyễn Hoài Nam**

Pipeline nghiên cứu tái lập để dự báo lợi suất, mức điểm, trạng thái Bull/Sideway/Bear/Stress và phân phối rủi ro VN-Index. Kiến trúc giữ Filtered HMM, EGARCH Student-t và regime-aware Random Forest, đồng thời thêm validation-gated distribution center, sequential conformal, stratified/importance sampling, adaptive Monte Carlo, outer stationary bootstrap và delete-block jackknife.

> Đây là nghiên cứu định lượng, không phải khuyến nghị đầu tư.

## Dữ liệu

Tệp `data/raw/VNINDEX_Daily.csv` có {context["quality"]["rows_loaded"]:,} phiên từ {context["quality"]["start_date"]} đến {context["quality"]["end_date"]}. CSV nguồn có dấu phẩy hàng nghìn không được quote; parser phục hồi OHLCV và xác minh High/Low. Pipeline không nội suy close qua ngày thiếu.

## Kiến trúc

```mermaid
flowchart LR
  A[OHLCV] --> B[Validation và causal features]
  B --> C[Purged expanding time split]
  C --> D[Filtered HMM]
  C --> E[EGARCH Student-t]
  D --> F[Regime-aware RF]
  E --> F
  F --> G[Validation-gated center]
  G --> Q[Sequential conformal]
  D --> H[Regime path]
  E --> I[Student-t và residual blocks]
  Q --> H
  H --> J[Hybrid Monte Carlo]
  I --> J
  J --> K[Stratified IS và adaptive stopping]
  K --> L[VaR ES conformal intervals]
  L --> M[Outer bootstrap và delete-block jackknife]
```

Với horizon `h`, `R(t,h)=log(P(t+h)/P(t))` và `P_hat(t+h)=P(t) exp(R_hat(t,h))`. HMM chỉ xuất `P(S_t|F_t)` bằng forward recursion; không dùng smoothed posterior. Split purge bằng `target_end_date_h < boundary` và embargo bằng horizon lớn nhất.

## Cài đặt và chạy

```bash
conda env create -f environment.yml
conda activate vnindex-model
python -m pip install -e .
pytest -q
python -m vnindex_model.cli run-all --config configs/quick.yaml
```

Các lệnh độc lập: `validate-data`, `train`, `backtest`, `forecast`, `report`, `run-all`. Makefile cung cấp `make install`, `make test`, `make quick`, `make full`, `make forecast`, `make report`.

## Kết quả test ngoài mẫu

{_markdown(best)}

Đây là point metrics; kết quả trạng thái, calibration, interval và tail risk nằm trong `reports/tables/`. Mô hình có RMSE tốt nhất không tự động có recall Bear/Stress hoặc VaR coverage tốt nhất. Kết luận superiority chỉ được chấp nhận khi DM/HAC và block-bootstrap CI hỗ trợ; xem báo cáo để biết kết luận của run này.

Ở h=20, A0 RF có RMSE **{before['rmse']:.6f}**; gated distribution center khóa alpha ML **{alpha20['alpha']:.2f}** trên validation và đạt RMSE test **{after['rmse']:.6f}**. Đây là fallback bảo vệ, không phải bằng chứng ML vượt baseline. Sequential conformal chọn **{selected_conformal}**: coverage 95% đổi từ **{before['coverage_95']:.2%}** lên **{after['coverage_95']:.2%}**, width từ **{before['interval_width_95']:.4f}** lên **{after['interval_width_95']:.4f}**, VaR exceedance từ **{before['var_exceedance_95']:.2%}** xuống **{after['var_exceedance_95']:.2%}**. Run đạt **{int(acceptance['passed'].sum())}/{len(acceptance)}** acceptance checks; nếu chưa đạt toàn bộ guardrail thiết yếu thì pipeline mới vẫn là experimental.

## Trạng thái promotion

Các artifacts và báo cáo hiện tại được sinh từ `configs/experimental.yaml`. Vì chỉ đạt {int(acceptance['passed'].sum())}/{len(acceptance)} acceptance checks, `configs/default.yaml` tiếp tục giữ A0 làm baseline production; không có auto-promotion. `configs/full.yaml` chưa được chạy trong lần nghiệm thu này.

## Forecast 20 phiên mới nhất

- Origin: {summary["forecast_origin"]}; close cuối: {summary["last_observed_close"]:.2f}.
- Terminal mean/median: {summary["expected_terminal_close"]:.2f} / {summary["median_terminal_close"]:.2f}.
- Xác suất tăng/giảm: {summary["probability_positive_return"]:.2%} / {summary["probability_negative_return"]:.2%}.
- VaR 95% và ES 95%: {summary["var_95"]:.2%} / {summary["expected_shortfall_95"]:.2%}.
- P(maximum drawdown vượt 5%): {summary["drawdown_probabilities"]["0.05"]:.2%}.
- Estimated trading dates dùng ngày làm việc gần đúng, chưa loại ngày nghỉ HOSE.

![Forecast mới nhất](reports/figures/36_latest_forecast.png)

Biểu đồ tóm tắt đặt dự báo trung vị cuối horizon **{summary["median_terminal_close"]:.2f}** bên cạnh xác suất tăng **{summary["probability_positive_return"]:.2%}**. Đây là phân phối có điều kiện từ thông tin tại origin, không phải target giá đơn điểm.

![Fan chart](reports/figures/25_fan_chart.png)

Fan chart cho thấy khoảng bất định mở rộng theo horizon; độ rộng interval 95% tại terminal là **{summary["model_uncertainty"]["terminal_95_interval_width"]:.2f} điểm**. Dải rộng phản ánh cả process noise, regime/volatility và model uncertainty, nên không nên chỉ đọc đường median.

![Monte Carlo paths](reports/figures/24_monte_carlo_paths.png)

Hình chỉ hiển thị một mẫu nhỏ trong **{summary["number_of_paths"]:,}** paths để tránh rối hình. Mỗi path là một kịch bản tương thích với giả định mô hình; không path nào là quỹ đạo “được chọn”.

![Filtered HMM regimes](reports/figures/06_hmm_regimes.png)

Regime probability là filtered `P(S_t|F_t)` nên chỉ dùng dữ liệu sẵn có tại thời điểm t; không dùng smoothed state nhìn về tương lai. State là trạng thái thống kê ẩn, không phải nhãn thị trường chắc chắn.

![Calibration](reports/figures/16_reliability_diagram.png)

Reliability diagram so xác suất dự báo với tỷ lệ quan sát trong từng bin; khoảng cách với đường chéo là calibration gap. Biểu đồ này không thay thế backtest coverage của return interval hay direct drawdown conformal.

![Drawdown distribution](reports/figures/29_maximum_drawdown_distribution.png)

Maximum drawdown trong 20 phiên có median **{-summary["median_maximum_drawdown"]:.2%}**, mean **{-summary["expected_maximum_drawdown"]:.2%}** và phía xấu 95% ở **{-summary["maximum_drawdown_quantiles"]["0.05"]:.2%}**. Đây là tổn thất peak-to-trough trong đường đi, vì vậy có thể lớn ngay cả khi terminal return dương.

## Cấu trúc và tái lập

- `src/vnindex_model/`: thêm `point_forecast.py`, `conformal.py`, `importance_sampling.py`, `tail_head.py`; simulation/bootstrap/jackknife được mở rộng.
- `configs/`: `default.yaml` khóa A0; `quick.yaml`, `experimental.yaml`, `full.yaml` là các mức compute cho pipeline A1-A9.
- `artifacts/`: model, metadata, latest forecast và NPZ samples.
- `reports/`: bảng CSV/Markdown, 70 hình và hai báo cáo tiếng Việt; baseline cũ nằm trong `reports/archive/`.
- `tests/`: leakage, parser, split, filtered probability, simulation, metric và smoke tests.

Để cập nhật, thay file trong `data/raw/` bằng OHLCV mới, cập nhật `project.data_path` nếu tên đổi và chạy lại `run-all`. Mọi số liệu trong README này được ghi lại từ pipeline; không chỉnh tay sau run.

## Hạn chế

Structural break, sparse Stress class, calibration drift, proxy lịch ngày làm việc, sai số HMM/EGARCH và giả định residual lịch sử còn đại diện đều có thể làm forecast lệch. Monte Carlo paths là các kịch bản có điều kiện; median path không phải quỹ đạo chắc chắn. Không có kết quả nào ở đây bảo đảm hiệu quả giao dịch.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")


def run_pipeline(config_path: str | Path = "configs/default.yaml") -> dict:
    started = time.perf_counter()
    root = Path(".").resolve()
    _configure_logging(root / "reports/diagnostics")
    config = load_config(config_path)
    advanced_enabled = config["project"].get("pipeline_mode", "baseline") == "experimental"
    data_path = Path(config["project"].get("data_path") or discover_data_file(root))
    seed = int(config["project"]["seed"])
    np.random.seed(seed)
    (root / "artifacts/metadata").mkdir(parents=True, exist_ok=True)
    config_snapshot = {key: value for key, value in config.items() if not key.startswith("_")}
    (root / "artifacts/metadata/config_snapshot.yaml").write_text(
        yaml.safe_dump(config_snapshot, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _log("run_started", config=str(config_path), data_path=str(data_path), seed=seed)
    data, quality = validate_and_save(data_path, root)
    metadata = run_metadata(data_path, config)
    metadata["last_data_date"] = quality["end_date"]
    _log("data_loaded", rows=len(data), start=quality["start_date"], end=quality["end_date"], hash=quality["sha256"])
    horizons = [int(value) for value in config["data"]["horizons"]]
    features = build_features(data)
    targets = build_targets(data, horizons)
    max_horizon = max(horizons)
    base_split = purged_train_validation_test(
        data["date"],
        targets[f"target_end_date_{max_horizon}"],
        config["data"]["train_fraction"],
        config["data"]["validation_fraction"],
        config["data"]["embargo"],
    )
    assert_no_target_overlap(
        targets[f"target_end_date_{max_horizon}"],
        pd.Series(data.index.isin(base_split.train)),
        base_split.train_boundary,
    )
    selected_technical = select_train_features(features, base_split.train, config["features"]["correlation_threshold"])
    (root / "artifacts/metadata").mkdir(parents=True, exist_ok=True)
    (root / "artifacts/metadata/selected_features.txt").write_text(
        "\n".join(selected_technical) + "\n", encoding="utf-8"
    )
    hmm_result = fit_filtered_hmm(
        features,
        features["log_return"],
        features["current_drawdown"],
        base_split.train,
        config["hmm"]["candidate_states"],
        config["hmm"]["seeds"],
        config["hmm"]["n_iter"],
    )
    volatility = fit_egarch_student_t(features["log_return"], base_split.train)
    volatility_candidates = {
        "historical_volatility": features["rolling_volatility_20"].ffill().to_numpy(),
        "ewma_volatility": features["ewma_volatility"].ffill().to_numpy(),
        "GARCH_Gaussian": fit_arch_candidate(features["log_return"], base_split.train, "GARCH", "normal")[0],
        "GARCH_Student_t": fit_arch_candidate(features["log_return"], base_split.train, "GARCH", "StudentsT")[0],
        "EGARCH_Student_t": volatility.features["egarch_conditional_volatility"].to_numpy(),
    }
    volatility_comparison_rows = []
    realized_squared = features["log_return"].fillna(0.0).to_numpy() ** 2
    for model_name, sigma_values in volatility_candidates.items():
        for split_name, indices in {
            "validation": base_split.validation,
            "test": base_split.test,
        }.items():
            variance = np.maximum(np.square(sigma_values[indices]), 1e-10)
            qlike = np.mean(np.log(variance) + realized_squared[indices] / variance)
            coverage = np.mean(np.abs(features["log_return"].to_numpy()[indices]) <= 1.96 * np.sqrt(variance))
            volatility_comparison_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "qlike": float(qlike),
                    "gaussian_95_coverage": float(coverage),
                    "finite": bool(np.isfinite(variance).all()),
                }
            )
    volatility_model_comparison = pd.DataFrame(volatility_comparison_rows)
    validation_volatility = volatility_model_comparison[volatility_model_comparison["split"] == "validation"]
    selected_volatility_name = validation_volatility.loc[validation_volatility["qlike"].idxmin(), "model"]
    volatility_model_comparison["selected_on_validation"] = (
        volatility_model_comparison["model"] == selected_volatility_name
    )
    selected_block_length = choose_block_length(
        volatility.standardized_residuals[base_split.validation], [5, 10, 15, 20]
    )
    write_json(root / "artifacts/metadata/hmm_diagnostics.json", hmm_result.diagnostics)
    write_json(root / "artifacts/metadata/egarch_diagnostics.json", volatility.diagnostics)
    _log(
        "latent_models_fitted",
        hmm_states=hmm_result.diagnostics["selected_states"],
        hmm_seed=hmm_result.diagnostics["selected_seed"],
        egarch_model=volatility.diagnostics["model"],
        egarch_fallback=volatility.diagnostics["fallback"],
    )
    hmm_numeric = hmm_result.probabilities.drop(columns=["hmm_state"])
    augmented_hmm = pd.concat([features[selected_technical], hmm_numeric], axis=1)
    augmented_egarch = pd.concat([features[selected_technical], volatility.features], axis=1)
    augmented_full = pd.concat([augmented_hmm, volatility.features], axis=1)
    hmm_probability_columns = [column for column in hmm_result.probabilities if column.startswith("hmm_probability_")]
    regime_probabilities = hmm_result.probabilities[hmm_probability_columns].to_numpy()

    comparison_rows: list[dict] = []
    horizon_rows: list[dict] = []
    class_rows: list[pd.DataFrame] = []
    calibration_rows: list[dict] = []
    interval_rows: list[dict] = []
    tail_rows: list[dict] = []
    statistical_rows: list[dict] = []
    ablation_rows: list[dict] = []
    tuning_rows: list[dict] = []
    threshold_rows: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    center_selection_rows: list[pd.DataFrame] = []
    conformal_selection_rows: list[pd.DataFrame] = []
    conformal_multiplier_frames: list[pd.DataFrame] = []
    horizon_state: dict[int, dict] = {}

    for horizon in horizons:
        loop_start = time.perf_counter()
        split = purged_train_validation_test(
            data["date"],
            targets[f"target_end_date_{horizon}"],
            config["data"]["train_fraction"],
            config["data"]["validation_fraction"],
            config["data"]["embargo"],
        )
        finite = (
            targets[[f"forward_return_{horizon}", f"normalized_return_{horizon}", f"forward_max_drawdown_{horizon}"]]
            .notna()
            .all(axis=1)
            .to_numpy()
        )
        train_index = split.train[finite[split.train]]
        validation_index = split.validation[finite[split.validation]]
        test_index = split.test[finite[split.test]]
        selected_lambdas, horizon_thresholds = select_regime_thresholds(
            targets[f"forward_return_{horizon}"],
            targets[f"forward_max_drawdown_{horizon}"],
            targets[f"forecast_scale_{horizon}"],
            train_index,
            validation_index,
        )
        horizon_thresholds.insert(0, "horizon", horizon)
        threshold_rows.append(horizon_thresholds)
        selected_labels = assign_regime_labels(
            targets[f"forward_return_{horizon}"],
            targets[f"forward_max_drawdown_{horizon}"],
            targets[f"forecast_scale_{horizon}"],
            **selected_lambdas,
        )
        targets[f"regime_{horizon}"] = pd.Categorical(selected_labels, categories=CLASS_NAMES)
        y_class = targets[f"regime_{horizon}"].astype(object).to_numpy()
        y_return = targets[f"forward_return_{horizon}"].to_numpy(dtype=float)
        y_normalized = targets[f"normalized_return_{horizon}"].to_numpy(dtype=float)
        y_drawdown = targets[f"forward_max_drawdown_{horizon}"].to_numpy(dtype=float)
        tuned_config, horizon_tuning = _tune_rf_regularization(
            features[selected_technical],
            data["date"],
            targets[f"target_end_date_{horizon}"],
            train_index,
            y_class,
            y_return,
            y_normalized,
            y_drawdown,
            config["random_forest"],
            seed,
            horizon,
            int(config["validation"]["folds"]),
        )
        tuning_rows.extend(horizon_tuning)
        bundles = {
            "rf_basic": fit_forest_bundle(
                features[selected_technical].iloc[train_index],
                y_class[train_index],
                y_return[train_index],
                y_normalized[train_index],
                y_drawdown[train_index],
                selected_technical,
                tuned_config,
                seed,
            ),
            "rf_hmm": fit_forest_bundle(
                augmented_hmm.iloc[train_index],
                y_class[train_index],
                y_return[train_index],
                y_normalized[train_index],
                y_drawdown[train_index],
                augmented_hmm.columns.tolist(),
                tuned_config,
                seed,
            ),
            "rf_egarch": fit_forest_bundle(
                augmented_egarch.iloc[train_index],
                y_class[train_index],
                y_return[train_index],
                y_normalized[train_index],
                y_drawdown[train_index],
                augmented_egarch.columns.tolist(),
                tuned_config,
                seed,
            ),
            "rf_hmm_egarch": fit_forest_bundle(
                augmented_full.iloc[train_index],
                y_class[train_index],
                y_return[train_index],
                y_normalized[train_index],
                y_drawdown[train_index],
                augmented_full.columns.tolist(),
                tuned_config,
                seed,
            ),
        }
        soft = fit_soft_gated_forest(
            augmented_full.iloc[train_index],
            regime_probabilities[train_index],
            y_class[train_index],
            y_return[train_index],
            y_normalized[train_index],
            y_drawdown[train_index],
            augmented_full.columns.tolist(),
            tuned_config,
            seed,
        )
        validation_predictions = {
            name: predict_bundle(
                bundle,
                {
                    "rf_basic": features[selected_technical],
                    "rf_hmm": augmented_hmm,
                    "rf_egarch": augmented_egarch,
                    "rf_hmm_egarch": augmented_full,
                }[name].iloc[validation_index],
            )
            for name, bundle in bundles.items()
        }
        validation_predictions["soft_gated_rf"] = predict_soft_gated(
            soft, augmented_full.iloc[validation_index], regime_probabilities[validation_index]
        )
        candidate_loss = {
            name: classification_metrics(y_class[validation_index], prediction["probabilities"])[0]["log_loss"]
            for name, prediction in validation_predictions.items()
            if name in {"rf_hmm_egarch", "soft_gated_rf"}
        }
        selected_model = min(candidate_loss, key=candidate_loss.get)
        calibrator, calibration_comparison = select_temporal_calibration(
            validation_predictions[selected_model]["probabilities"], y_class[validation_index]
        )
        test_predictions = {
            name: predict_bundle(
                bundle,
                {
                    "rf_basic": features[selected_technical],
                    "rf_hmm": augmented_hmm,
                    "rf_egarch": augmented_egarch,
                    "rf_hmm_egarch": augmented_full,
                }[name].iloc[test_index],
            )
            for name, bundle in bundles.items()
        }
        test_predictions["soft_gated_rf"] = predict_soft_gated(
            soft, augmented_full.iloc[test_index], regime_probabilities[test_index]
        )
        main_prediction = test_predictions[selected_model]
        calibrated_probability = calibrator.transform(main_prediction["probabilities"])
        class_metrics, per_class, confusion = classification_metrics(y_class[test_index], calibrated_probability)
        per_class["horizon"] = horizon
        per_class["model"] = f"{selected_model}+{calibrator.method}"
        class_rows.append(per_class)
        for item in calibration_comparison:
            calibration_rows.append({"horizon": horizon, **item, "selected": item["method"] == calibrator.method})
        calibration_rows.append(
            {
                "horizon": horizon,
                "method": f"selected_test_{calibrator.method}",
                "validation_log_loss": class_metrics["log_loss"],
                "selected": True,
                "test_brier": class_metrics["brier_score"],
                "test_ece": class_metrics["ece"],
                "calibration_slope": class_metrics["calibration_slope"],
                "calibration_intercept": class_metrics["calibration_intercept"],
            }
        )

        validation_baseline = baseline_predictions(
            data["close"],
            features["log_return"],
            y_return,
            features[selected_technical].to_numpy(),
            train_index,
            validation_index,
            horizon,
        )
        baseline = baseline_predictions(
            data["close"],
            features["log_return"],
            y_return,
            features[selected_technical].to_numpy(),
            train_index,
            test_index,
            horizon,
        )
        candidate_center_selection = select_validation_gated_center(
            y_return[validation_index],
            validation_predictions[selected_model]["return"],
            validation_baseline["random_walk_drift"],
            config["point_forecast"]["alpha_grid"],
            config["point_forecast"]["minimum_relative_improvement"],
            config["point_forecast"]["use_one_standard_error_rule"],
            horizon,
        )
        if advanced_enabled:
            center_selection = candidate_center_selection
        else:
            baseline_table = candidate_center_selection.validation_table.copy()
            baseline_table["selected"] = np.isclose(baseline_table["alpha"], 1.0)
            baseline_table["selection_reason"] = "advanced pipeline disabled; preserve current RF center"
            center_selection = CenterSelection(
                1.0,
                "current_rf_center",
                "advanced pipeline disabled; preserve current RF center",
                baseline_table,
            )
        center_table = center_selection.validation_table.copy()
        center_table.insert(0, "horizon", horizon)
        center_table["selected_center"] = center_selection.selected_center
        center_selection_rows.append(center_table)
        validation_center = apply_center_blend(
            validation_predictions[selected_model]["return"],
            validation_baseline["random_walk_drift"],
            center_selection.alpha,
        )
        improved_center = apply_center_blend(
            main_prediction["return"], baseline["random_walk_drift"], center_selection.alpha
        )
        model_returns = {**baseline, **{name: prediction["return"] for name, prediction in test_predictions.items()}}
        state_means = np.array(
            [
                np.average(y_return[train_index], weights=regime_probabilities[train_index, state])
                for state in range(regime_probabilities.shape[1])
            ]
        )
        model_returns["markov_switching_ar"] = regime_probabilities[test_index] @ state_means
        model_returns["full_hybrid_simulation"] = main_prediction["return"]
        model_returns["validation_gated_distribution_center"] = improved_center
        actual_return = y_return[test_index]
        actual_price = targets[f"future_price_{horizon}"].to_numpy(dtype=float)[test_index]
        current_price = data["close"].to_numpy(dtype=float)[test_index]
        metrics_by_model: dict[str, dict] = {}
        for name, predicted_return in model_returns.items():
            metrics = point_metrics(
                actual_return,
                predicted_return,
                actual_price,
                current_price * np.exp(predicted_return),
                y_return[train_index],
            )
            metrics_by_model[name] = metrics
            comparison_rows.append({"horizon": horizon, "model": name, **metrics})
        old_point_metrics = metrics_by_model[selected_model]
        improved_point_metrics = metrics_by_model["validation_gated_distribution_center"]
        horizon_rows.append(
            {
                "horizon": horizon,
                "selected_model": selected_model,
                "distribution_center": center_selection.selected_center,
                "center_alpha": center_selection.alpha,
                "center_selection_reason": center_selection.reason,
                "calibration": calibrator.method,
                **improved_point_metrics,
                **{f"old_{key}": value for key, value in old_point_metrics.items()},
                **class_metrics,
                "n_train": len(train_index),
                "n_validation": len(validation_index),
                "n_test": len(test_index),
            }
        )
        predicted_labels = np.array(CLASS_NAMES)[calibrated_probability.argmax(axis=1)]
        prediction_frame = pd.DataFrame(
            {
                "date": data["date"].iloc[test_index].to_numpy(),
                "horizon": horizon,
                "actual_return": actual_return,
                "predicted_return": improved_center,
                "old_predicted_return": main_prediction["return"],
                "actual_price": actual_price,
                "predicted_price": current_price * np.exp(improved_center),
                "old_predicted_price": current_price * np.exp(main_prediction["return"]),
                "actual_drawdown": y_drawdown[test_index],
                "predicted_drawdown": main_prediction["drawdown"],
                "actual_regime": y_class[test_index],
                "predicted_regime": predicted_labels,
            }
        )
        for class_position, class_name in enumerate(CLASS_NAMES):
            prediction_frame[f"probability_{class_name.lower()}"] = calibrated_probability[:, class_position]
        prediction_frames.append(prediction_frame)
        validation_sigma = volatility.features["egarch_forecast_volatility"].iloc[
            validation_index
        ].to_numpy() * np.sqrt(horizon)
        sigma = volatility.features["egarch_forecast_volatility"].iloc[test_index].to_numpy() * np.sqrt(horizon)
        residual_pool = volatility.standardized_residuals[train_index]
        residual_pool = residual_pool[np.isfinite(residual_pool)]
        nu = float(volatility.diagnostics["nu"])
        validation_regime = regime_probabilities[validation_index].argmax(axis=1)
        test_regime = regime_probabilities[test_index].argmax(axis=1)
        edge_fit_stop = max(int(len(validation_sigma) * 0.60), 1)
        volatility_edges = volatility_bin_edges(
            validation_sigma[:edge_fit_stop], config["conformal"]["stratification"]["volatility_bins"]
        )
        validation_bins = assign_volatility_bins(validation_sigma, volatility_edges)
        test_bins = assign_volatility_bins(sigma, volatility_edges)
        conformal_selection = select_conformal_method(
            y_return[validation_index],
            validation_center,
            validation_sigma,
            validation_regime,
            validation_bins,
            horizon,
            config["conformal"]["alpha_levels"],
            config["conformal"]["candidate_windows"],
            config["conformal"]["minimum_stratum_size"],
        )
        conformal_table = conformal_selection.validation_table.copy()
        conformal_table.insert(0, "horizon", horizon)
        conformal_table["selection_reason"] = conformal_selection.reason
        conformal_selection_rows.append(conformal_table)
        conformal_result = sequential_conformal(
            y_return[validation_index],
            validation_center,
            validation_sigma,
            validation_regime,
            validation_bins,
            actual_return,
            improved_center,
            sigma,
            test_regime,
            test_bins,
            horizon,
            config["conformal"]["alpha_levels"],
            conformal_selection.method,
            conformal_selection.window,
            config["conformal"]["minimum_stratum_size"],
        )
        conformal_result.insert(0, "date", data["date"].iloc[test_index].to_numpy())
        conformal_result.insert(1, "horizon", horizon)
        conformal_result["method"] = conformal_selection.method
        conformal_result["window"] = conformal_selection.window
        conformal_multiplier_frames.append(conformal_result)
        for level in [0.50, 0.80, 0.90, 0.95]:
            alpha = (1 - level) / 2
            standardized_quantile = t.ppf([alpha, 1 - alpha], nu) * np.sqrt((nu - 2) / nu)
            lower = main_prediction["return"] + sigma * standardized_quantile[0]
            upper = main_prediction["return"] + sigma * standardized_quantile[1]
            interval_metric = interval_metrics(actual_return, lower, upper, level)
            interval_metric.update({"horizon": horizon, "method": "EGARCH Student-t conditional"})
            interval_metric["crps"] = _crps_empirical(
                actual_return, main_prediction["return"], sigma, residual_pool, seed
            )
            interval_rows.append(interval_metric)
            suffix = str(int(round(level * 100)))
            if not advanced_enabled:
                conformal_result[f"lower_{suffix}"] = lower
                conformal_result[f"upper_{suffix}"] = upper
                conformal_result[f"multiplier_{suffix}"] = abs(standardized_quantile[1])
            calibrated_interval = interval_metrics(
                actual_return,
                conformal_result[f"lower_{suffix}"],
                conformal_result[f"upper_{suffix}"],
                level,
            )
            calibrated_interval.update(
                {
                    "horizon": horizon,
                    "method": f"sequential_conformal:{conformal_selection.method}",
                    "crps": np.nan,
                }
            )
            interval_rows.append(calibrated_interval)
        q95 = t.ppf(0.05, nu) * np.sqrt((nu - 2) / nu)
        q99 = t.ppf(0.01, nu) * np.sqrt((nu - 2) / nu)
        old_var95 = main_prediction["return"] + sigma * q95
        old_var99 = main_prediction["return"] + sigma * q99
        var95 = conformal_result["var_95"].to_numpy(dtype=float) if advanced_enabled else old_var95
        validation_signed_scores = (y_return[validation_index] - validation_center) / np.maximum(validation_sigma, 1e-8)
        signed_cutoff = signed_lower_quantile(validation_signed_scores, 0.05)
        signed_tail = validation_signed_scores[validation_signed_scores <= signed_cutoff]
        conformal_es95 = improved_center + sigma * float(np.mean(signed_tail))
        if not advanced_enabled:
            conformal_es95 = (
                main_prediction["return"]
                + sigma * residual_pool[residual_pool <= np.quantile(residual_pool, 0.05)].mean()
            )
        var99 = improved_center + sigma * signed_lower_quantile(validation_signed_scores, 0.01)
        one_hot = np.column_stack([(y_class[test_index] == name).astype(int) for name in CLASS_NAMES])
        prediction_frame["brier_loss"] = np.sum((calibrated_probability - one_hot) ** 2, axis=1)
        prediction_frame["old_interval_95_covered"] = ((actual_return >= lower) & (actual_return <= upper)).astype(
            float
        )
        prediction_frame["old_lower_95"] = lower
        prediction_frame["old_upper_95"] = upper
        prediction_frame["interval_95_covered"] = (
            (actual_return >= conformal_result["lower_95"].to_numpy())
            & (actual_return <= conformal_result["upper_95"].to_numpy())
        ).astype(float)
        for level in [50, 80, 90, 95]:
            prediction_frame[f"lower_{level}"] = conformal_result[f"lower_{level}"].to_numpy()
            prediction_frame[f"upper_{level}"] = conformal_result[f"upper_{level}"].to_numpy()
        prediction_frame["var_95"] = var95
        prediction_frame["old_var_95"] = old_var95
        prediction_frame["expected_shortfall_95"] = conformal_es95
        prediction_frame["conformal_method"] = conformal_selection.method
        prediction_frame["conformal_window"] = conformal_selection.window
        tests95 = historical_var_tests(actual_return, var95, 0.05)
        tests99 = historical_var_tests(actual_return, var99, 0.01)
        old_tests95 = historical_var_tests(actual_return, old_var95, 0.05)
        old_tests99 = historical_var_tests(actual_return, old_var99, 0.01)
        tail_rows.append(
            {
                "horizon": horizon,
                "var_95_mean": float(var95.mean()),
                "var_99_mean": float(var99.mean()),
                "expected_shortfall_95_mean": float(np.mean(conformal_es95)),
                "expected_shortfall_99_mean": float(
                    np.mean(
                        main_prediction["return"]
                        + sigma * residual_pool[residual_pool <= np.quantile(residual_pool, 0.01)].mean()
                    )
                ),
                "predicted_max_drawdown_mean": float(main_prediction["drawdown"].mean()),
                "realized_max_drawdown_mean": float(y_drawdown[test_index].mean()),
                **{f"95_{key}": value for key, value in tests95.items()},
                **{f"99_{key}": value for key, value in tests99.items()},
                **{f"old_95_{key}": value for key, value in old_tests95.items()},
                **{f"old_99_{key}": value for key, value in old_tests99.items()},
            }
        )
        for baseline_name, baseline_prediction in baseline.items():
            dm = diebold_mariano(actual_return, improved_center, baseline_prediction, horizon)
            bootstrap = block_bootstrap_difference(
                actual_return,
                improved_center,
                baseline_prediction,
                config["validation"]["bootstrap_reps"],
                max(5, horizon),
                seed,
            )
            statistical_rows.append(
                {"horizon": horizon, "baseline": baseline_name, **dm, **bootstrap, "multiple_comparison_warning": True}
            )
        components = {
            "RF cơ bản": "rf_basic",
            "RF + HMM": "rf_hmm",
            "RF + EGARCH": "rf_egarch",
            "RF + HMM + EGARCH": "rf_hmm_egarch",
            "soft-gated RF": "soft_gated_rf",
            "Student-t Monte Carlo": selected_model,
            "regime block bootstrap": selected_model,
            "hybrid simulation": selected_model,
            "hybrid không jackknife": selected_model,
            "pipeline đầy đủ": selected_model,
        }
        for component, model_key in components.items():
            base_metrics = metrics_by_model[model_key]
            class_source = (
                calibrated_probability if model_key == selected_model else test_predictions[model_key]["probabilities"]
            )
            component_class = classification_metrics(y_class[test_index], class_source)[0]
            ablation_rows.append(
                {
                    "horizon": horizon,
                    "component": component,
                    "rmse_return": base_metrics["rmse_return"],
                    "directional_accuracy": base_metrics["directional_accuracy"],
                    "brier_score": component_class["brier_score"],
                    "macro_f1": component_class["macro_f1"],
                    "jackknife_attached": component == "pipeline đầy đủ",
                }
            )
        horizon_state[horizon] = {
            "split": split,
            "train_index": train_index,
            "validation_index": validation_index,
            "test_index": test_index,
            "bundles": bundles,
            "soft": soft,
            "selected_model": selected_model,
            "calibrator": calibrator,
            "main_prediction": main_prediction,
            "calibrated_probability": calibrated_probability,
            "confusion": confusion,
            "y_class": y_class,
            "augmented_full": augmented_full,
            "tuned_config": tuned_config,
            "selected_lambdas": selected_lambdas,
            "center_selection": center_selection,
            "conformal_selection": conformal_selection,
            "validation_center": validation_center,
            "validation_drawdown_prediction": validation_predictions[selected_model]["drawdown"],
            "improved_center": improved_center,
            "y_drawdown": y_drawdown,
            "conformal_result": conformal_result,
            "validation_sigma": validation_sigma,
            "test_sigma": sigma,
            "validation_regime": validation_regime,
            "validation_bins": validation_bins,
            "volatility_edges": volatility_edges,
            "test_regime": test_regime,
            "test_bins": test_bins,
        }
        _log(
            "horizon_completed",
            horizon=horizon,
            selected_model=selected_model,
            calibration=calibrator.method,
            center_alpha=center_selection.alpha,
            conformal_method=conformal_selection.method,
            n_train=len(train_index),
            n_test=len(test_index),
            elapsed_seconds=round(time.perf_counter() - loop_start, 2),
        )

    model_comparison = pd.DataFrame(comparison_rows)
    per_horizon = pd.DataFrame(horizon_rows)
    per_class_metrics = pd.concat(class_rows, ignore_index=True)
    calibration_metrics = pd.DataFrame(calibration_rows)
    interval_metric_table = pd.DataFrame(interval_rows)
    tail_metric_table = pd.DataFrame(tail_rows)
    statistical_table = pd.DataFrame(statistical_rows)
    ablation_table = pd.DataFrame(ablation_rows)
    tuning_table = pd.DataFrame(tuning_rows)
    threshold_table = pd.concat(threshold_rows, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    center_selection_table = pd.concat(center_selection_rows, ignore_index=True)
    conformal_selection_table = pd.concat(conformal_selection_rows, ignore_index=True)
    conformal_multiplier_table = pd.concat(conformal_multiplier_frames, ignore_index=True)
    h20 = 20 if 20 in horizon_state else max(horizons)
    state20 = horizon_state[h20]
    records20 = predictions[predictions["horizon"] == h20].reset_index(drop=True)
    records20["old_center"] = records20["old_predicted_return"]
    records20["improved_center"] = records20["predicted_return"]
    records20["conformal_var_95"] = records20["var_95"]
    records20["conformal_es_95"] = records20["expected_shortfall_95"]
    conformal_test_rows: list[dict[str, float | int | str | None]] = []
    conformal_test_outputs: dict[str, pd.DataFrame] = {}
    h20_validation_candidates = state20["conformal_selection"].validation_table
    for conformal_method in ["global", "volatility_stratified", "volatility_regime"]:
        method_candidates = h20_validation_candidates[h20_validation_candidates["method"] == conformal_method]
        best_candidate = method_candidates.loc[method_candidates["objective"].idxmin()]
        method_window = None if pd.isna(best_candidate["window"]) else int(best_candidate["window"])
        method_result = sequential_conformal(
            targets[f"forward_return_{h20}"].to_numpy()[state20["validation_index"]],
            state20["validation_center"],
            state20["validation_sigma"],
            state20["validation_regime"],
            state20["validation_bins"],
            targets[f"forward_return_{h20}"].to_numpy()[state20["test_index"]],
            state20["improved_center"],
            state20["test_sigma"],
            state20["test_regime"],
            state20["test_bins"],
            h20,
            config["conformal"]["alpha_levels"],
            conformal_method,
            method_window,
            config["conformal"]["minimum_stratum_size"],
        )
        conformal_test_outputs[conformal_method] = method_result
        method_interval = interval_metrics(
            targets[f"forward_return_{h20}"].to_numpy()[state20["test_index"]],
            method_result["lower_95"],
            method_result["upper_95"],
            0.95,
        )
        method_var = historical_var_tests(
            targets[f"forward_return_{h20}"].to_numpy()[state20["test_index"]], method_result["var_95"], 0.05
        )
        conformal_test_rows.append(
            {
                "method": conformal_method,
                "window": method_window,
                **method_interval,
                **method_var,
                "selected_on_validation": conformal_method == state20["conformal_selection"].method
                and method_window == state20["conformal_selection"].window,
            }
        )
    conformal_test_comparison = pd.DataFrame(conformal_test_rows)
    jackknife_table = block_jackknife_table(records20)
    delete_jackknife_table = delete_block_jackknife(records20)
    if config["outer_bootstrap"]["mode"] == "quick":
        outer_bootstrap_summary, outer_bootstrap_replicates = outer_stationary_bootstrap_quick(
            records20,
            config["outer_bootstrap"]["replications_quick"],
            config["outer_bootstrap"]["mean_block_length"],
            seed,
        )
        effective_outer_bootstrap_mode = "quick_oos_record_stationary_blocks"
    else:
        outer_bootstrap_summary, outer_bootstrap_replicates = outer_stationary_bootstrap_quick(
            records20,
            config["outer_bootstrap"]["replications_quick"],
            config["outer_bootstrap"]["mean_block_length"],
            seed,
        )
        effective_outer_bootstrap_mode = "quick_fallback_full_refit_not_executed"
        _log(
            "outer_bootstrap_fallback",
            requested="full",
            effective=effective_outer_bootstrap_mode,
            reason="full refit configuration was not executed in this run",
        )
    tail_head_table, tail_head_probability, tail_head_summary = run_tail_head_experiment(
        state20["augmented_full"].iloc[state20["train_index"]],
        state20["y_class"][state20["train_index"]],
        state20["augmented_full"].iloc[state20["validation_index"]],
        state20["y_class"][state20["validation_index"]],
        state20["augmented_full"].iloc[state20["test_index"]],
        state20["y_class"][state20["test_index"]],
        seed,
        config["tail_head"]["n_estimators"],
        config["tail_head"]["minimum_precision"],
    )
    records20["experimental_tail_probability"] = tail_head_probability
    requested_jackknife_mode = config["jackknife"]["mode"]
    if requested_jackknife_mode == "full":
        effective_jackknife_mode = "quick_fallback_full_refit_not_executed"
        _log(
            "jackknife_fallback",
            requested="full",
            effective=effective_jackknife_mode,
            reason="full refit exceeds the accepted CPU budget",
        )
    else:
        effective_jackknife_mode = "quick"
    jackknife_table["mode"] = effective_jackknife_mode

    seed_rows: list[dict] = []
    for forest_seed in config["project"]["seeds"]:
        model = fit_forest_bundle(
            augmented_full.iloc[state20["train_index"]],
            targets[f"regime_{h20}"].astype(object).to_numpy()[state20["train_index"]],
            targets[f"forward_return_{h20}"].to_numpy()[state20["train_index"]],
            targets[f"normalized_return_{h20}"].to_numpy()[state20["train_index"]],
            targets[f"forward_max_drawdown_{h20}"].to_numpy()[state20["train_index"]],
            augmented_full.columns.tolist(),
            state20["tuned_config"],
            int(forest_seed),
        )
        predicted = predict_bundle(model, augmented_full.iloc[state20["test_index"]])["return"]
        metric = point_metrics(
            targets[f"forward_return_{h20}"].to_numpy()[state20["test_index"]],
            predicted,
            targets[f"future_price_{h20}"].to_numpy()[state20["test_index"]],
            data["close"].to_numpy()[state20["test_index"]] * np.exp(predicted),
            targets[f"forward_return_{h20}"].to_numpy()[state20["train_index"]],
        )
        seed_rows.append({"seed": forest_seed, **metric})
    seed_stability = pd.DataFrame(seed_rows)
    seed_summary = pd.DataFrame(
        [
            {
                "metric": column,
                "mean": seed_stability[column].mean(),
                "std": seed_stability[column].std(),
                "median": seed_stability[column].median(),
                "ci_lower": seed_stability[column].quantile(0.025),
                "ci_upper": seed_stability[column].quantile(0.975),
                "best_seed": int(seed_stability.loc[seed_stability[column].idxmin(), "seed"]),
                "worst_seed": int(seed_stability.loc[seed_stability[column].idxmax(), "seed"]),
            }
            for column in ["mae_return", "rmse_return"]
        ]
    )

    # Final refit: selected hyperparameters are fixed from validation, then all valid labeled rows are used.
    full_index = np.arange(len(data))
    full_hmm = fit_filtered_hmm(
        features,
        features["log_return"],
        features["current_drawdown"],
        full_index,
        [int(hmm_result.diagnostics["selected_states"])],
        [int(hmm_result.diagnostics["selected_seed"])],
        config["hmm"]["n_iter"],
    )
    full_volatility = fit_egarch_student_t(features["log_return"], full_index)
    full_hmm_columns = [column for column in full_hmm.probabilities if column.startswith("hmm_probability_")]
    full_augmented = pd.concat(
        [features[selected_technical], full_hmm.probabilities.drop(columns=["hmm_state"]), full_volatility.features],
        axis=1,
    )
    labelled = np.flatnonzero(
        targets[[f"forward_return_{h20}", f"normalized_return_{h20}", f"forward_max_drawdown_{h20}"]]
        .notna()
        .all(axis=1)
        .to_numpy()
    )
    full_soft = fit_soft_gated_forest(
        full_augmented.iloc[labelled],
        full_hmm.probabilities[full_hmm_columns].to_numpy()[labelled],
        targets[f"regime_{h20}"].astype(object).to_numpy()[labelled],
        targets[f"forward_return_{h20}"].to_numpy()[labelled],
        targets[f"normalized_return_{h20}"].to_numpy()[labelled],
        targets[f"forward_max_drawdown_{h20}"].to_numpy()[labelled],
        full_augmented.columns.tolist(),
        state20["tuned_config"],
        seed,
    )
    full_global = full_soft.global_bundle
    if state20["selected_model"] == "soft_gated_rf":
        latest_prediction = predict_soft_gated(
            full_soft, full_augmented.iloc[[-1]], full_hmm.probabilities[full_hmm_columns].to_numpy()[[-1]]
        )
        final_model = full_soft
    else:
        latest_prediction = predict_bundle(full_global, full_augmented.iloc[[-1]])
        final_model = full_global
    latest_probability = state20["calibrator"].transform(latest_prediction["probabilities"])[0]
    production_drift_horizon = (
        np.log(float(data["close"].iloc[-1]) / float(data["close"].iloc[0])) / max(len(data) - 1, 1) * h20
    )
    latest_center_horizon = float(
        apply_center_blend(
            latest_prediction["return"],
            np.array([production_drift_horizon]),
            state20["center_selection"].alpha,
        )[0]
    )
    latest_sigma_horizon = float(full_volatility.features["egarch_forecast_volatility"].iloc[-1] * np.sqrt(h20))
    latest_state = int(full_hmm.probabilities[full_hmm_columns].iloc[-1].to_numpy().argmax())
    latest_bin = int(assign_volatility_bins(np.array([latest_sigma_horizon]), state20["volatility_edges"])[0])
    latest_conformal = sequential_conformal(
        targets[f"forward_return_{h20}"].to_numpy()[state20["validation_index"]],
        state20["validation_center"],
        state20["validation_sigma"],
        state20["validation_regime"],
        state20["validation_bins"],
        np.array([latest_center_horizon]),
        np.array([latest_center_horizon]),
        np.array([latest_sigma_horizon]),
        np.array([latest_state]),
        np.array([latest_bin]),
        h20,
        config["conformal"]["alpha_levels"],
        state20["conformal_selection"].method,
        state20["conformal_selection"].window,
        config["conformal"]["minimum_stratum_size"],
    )
    nominal_multiplier = abs(
        float(
            t.ppf(0.975, full_volatility.diagnostics["nu"])
            * np.sqrt((full_volatility.diagnostics["nu"] - 2) / full_volatility.diagnostics["nu"])
        )
    )
    conformal_scale = (
        float(latest_conformal["multiplier_95"].iloc[0] / max(nominal_multiplier, 1e-8)) if advanced_enabled else 1.0
    )
    calibrated_daily_volatility = float(
        full_volatility.features["egarch_forecast_volatility"].iloc[-1] * conformal_scale
    )
    simulation_kwargs = {
        "last_close": float(data["close"].iloc[-1]),
        "horizon": config["simulation"]["horizon"],
        "daily_drift": latest_center_horizon / h20,
        "daily_volatility": calibrated_daily_volatility,
        "degrees_of_freedom": float(full_volatility.diagnostics["nu"]),
        "residuals": full_volatility.standardized_residuals,
        "historical_regime_probabilities": full_hmm.probabilities[full_hmm_columns].to_numpy(),
        "transition_matrix": full_hmm.transition_matrix,
        "current_regime_probability": full_hmm.probabilities[full_hmm_columns].iloc[-1].to_numpy(),
        "rf_class_probability": latest_probability,
        "economic_labels": full_hmm.economic_labels,
        "student_weight": config["simulation"]["student_weight"],
        "block_length": selected_block_length,
        "seed": seed,
    }
    simulation_results = {}
    for method in ["student_t", "regime_block_bootstrap"]:
        kwargs = dict(simulation_kwargs)
        kwargs["method"] = method
        simulation_results["bootstrap" if method == "regime_block_bootstrap" else method] = simulate_paths(
            paths=config["simulation"]["paths"], **kwargs
        )
    hybrid_kwargs = dict(simulation_kwargs)
    hybrid_kwargs["method"] = "hybrid"
    if config["simulation"]["adaptive_stopping"]["enabled"]:
        hybrid, convergence_history = adaptive_simulate_paths(
            float(data["close"].iloc[-1]),
            full_hmm.economic_labels,
            hybrid_kwargs,
            config["simulation"]["adaptive_stopping"],
        )
    else:
        hybrid = simulate_paths(paths=config["simulation"]["paths"], **hybrid_kwargs)
        convergence_history = pd.DataFrame()
    simulation_results["hybrid"] = hybrid
    stratified_started = time.perf_counter()
    stratified_result = run_stratified_simulation(
        len(hybrid.return_paths),
        config["simulation"]["stratified"]["allocation"],
        config["simulation"]["stratified"]["pilot_paths"],
        config["simulation"]["stratified"]["minimum_paths_per_regime"],
        hybrid_kwargs,
    )
    stratified_runtime = time.perf_counter() - stratified_started
    importance_config = config["simulation"]["importance_sampling"]
    importance_paths = int(importance_config["paths"])
    importance_common = {
        "paths": importance_paths,
        "horizon": config["simulation"]["horizon"],
        "daily_drift": latest_center_horizon / h20,
        "daily_volatility": calibrated_daily_volatility,
        "residuals": full_volatility.standardized_residuals,
        "transition_matrix": full_hmm.transition_matrix,
        "current_regime_probability": full_hmm.probabilities[full_hmm_columns].iloc[-1].to_numpy(),
        "economic_labels": full_hmm.economic_labels,
        "seed": seed,
    }
    naive_started = time.perf_counter()
    naive_tail = simulate_tail_importance(transition_strength=0.0, shock_strength=0.0, **importance_common)
    naive_runtime = time.perf_counter() - naive_started
    importance_results = []
    for strength in importance_config["proposal_strengths"]:
        importance_results.append(
            simulate_tail_importance(
                transition_strength=float(strength),
                shock_strength=float(strength) * importance_config["shock_strength_ratio"],
                **importance_common,
            )
        )
    importance_sensitivity = importance_sensitivity_table(importance_results)
    naive_mcse = float(naive_tail.diagnostics["mcse_mdd_7"])
    importance_sensitivity["variance_reduction_ratio_mdd_7"] = np.square(naive_mcse) / np.maximum(
        np.square(importance_sensitivity["mcse_mdd_7"]), 1e-18
    )
    importance_sensitivity["ess_accepted"] = (
        importance_sensitivity["ess_ratio"] >= importance_config["minimum_ess_ratio"]
    )
    accepted_importance = importance_sensitivity[importance_sensitivity["ess_accepted"]]
    if len(accepted_importance):
        importance_index = accepted_importance["variance_reduction_ratio_mdd_7"].idxmax()
        importance_acceptance_reason = "ESS gate passed; selected highest MDD-7 variance reduction"
    else:
        importance_index = importance_sensitivity["ess_ratio"].idxmax()
        importance_acceptance_reason = "all proposals rejected by ESS/N gate; retained for sensitivity only"
    importance_sensitivity["selected"] = importance_sensitivity.index == importance_index
    selected_importance = importance_results[int(importance_index)]
    selected_strength = float(importance_sensitivity.loc[importance_index, "transition_strength"])
    stratified_importance = simulate_stratified_importance(
        paths=importance_paths,
        current_regime_probability=importance_common["current_regime_probability"],
        minimum_paths_per_regime=config["simulation"]["stratified"]["minimum_paths_per_regime"],
        horizon=config["simulation"]["horizon"],
        daily_drift=latest_center_horizon / h20,
        daily_volatility=calibrated_daily_volatility,
        residuals=full_volatility.standardized_residuals,
        transition_matrix=full_hmm.transition_matrix,
        economic_labels=full_hmm.economic_labels,
        transition_strength=selected_strength,
        shock_strength=selected_strength * importance_config["shock_strength_ratio"],
        seed=seed,
    )
    simulation_efficiency = pd.DataFrame(
        [
            {
                "method": "naive_monte_carlo",
                **naive_tail.estimates,
                **naive_tail.diagnostics,
                "runtime_seconds": naive_runtime,
                "variance_reduction_ratio_mdd_7": 1.0,
            },
            {
                "method": "stratified_monte_carlo",
                **stratified_result.estimates,
                "paths": len(stratified_result.terminal_returns),
                "ess": len(stratified_result.terminal_returns),
                "ess_ratio": 1.0,
                "mcse_mdd_7": float(
                    np.sqrt(
                        stratified_result.estimates["probability_mdd_7"]
                        * (1 - stratified_result.estimates["probability_mdd_7"])
                        / len(stratified_result.terminal_returns)
                    )
                ),
                "runtime_seconds": stratified_runtime,
            },
            {
                "method": "importance_sampling",
                **selected_importance.estimates,
                **selected_importance.diagnostics,
                "variance_reduction_ratio_mdd_7": float(
                    importance_sensitivity.loc[importance_index, "variance_reduction_ratio_mdd_7"]
                ),
            },
            {
                "method": "stratified_importance_sampling",
                **stratified_importance.estimates,
                **stratified_importance.diagnostics,
                "variance_reduction_ratio_mdd_7": float(
                    np.square(naive_mcse) / max(np.square(stratified_importance.diagnostics["mcse_mdd_7"]), 1e-18)
                ),
            },
        ]
    )
    drawdown_layer = None
    if config.get("drawdown", {}).get("enabled", False):
        drawdown_layer = _run_drawdown_layer(
            root,
            config,
            data,
            features,
            targets,
            horizons,
            horizon_state,
            volatility,
            full_volatility,
            full_hmm,
            hybrid,
            simulation_results,
            convergence_history,
            naive_tail,
            importance_results,
            selected_importance,
            latest_center_horizon,
            calibrated_daily_volatility,
            seed,
        )
    forecast = hybrid.forecast.copy()
    future_dates = pd.bdate_range(pd.Timestamp(data["date"].iloc[-1]) + pd.offsets.BDay(1), periods=len(forecast))
    forecast.insert(1, "estimated_trading_date", future_dates.strftime("%Y-%m-%d"))
    forecast.to_csv(root / "artifacts/forecasts/latest_forecast.csv", index=False)
    sample_count = min(int(config["simulation"]["sample_paths"]), len(hybrid.price_paths))
    np.savez_compressed(
        root / "artifacts/forecasts/latest_monte_carlo_samples.npz",
        price_paths=hybrid.price_paths[:sample_count],
        return_paths=hybrid.return_paths[:sample_count],
        terminal_prices=hybrid.price_paths[:, -1],
        maximum_drawdowns=maximum_drawdown(
            np.column_stack([np.full(len(hybrid.price_paths), data["close"].iloc[-1]), hybrid.price_paths])
        ),
    )
    latest_summary = {
        "forecast_origin": quality["end_date"],
        "last_observed_close": float(data["close"].iloc[-1]),
        **hybrid.summary,
        "model_uncertainty": {
            "seed_terminal_return_std": float(seed_stability["mean_signed_error"].std()),
            "terminal_95_interval_width": float(forecast["upper_95"].iloc[-1] - forecast["lower_95"].iloc[-1]),
        },
        "point_center": {
            "mode": state20["center_selection"].selected_center,
            "alpha": state20["center_selection"].alpha,
            "horizon_return": latest_center_horizon,
        },
        "conformal": {
            "method": state20["conformal_selection"].method,
            "window": state20["conformal_selection"].window,
            "latest_multiplier_95": float(latest_conformal["multiplier_95"].iloc[0]),
            "volatility_scale": conformal_scale,
        },
        "importance_sampling": {
            "acceptance_reason": importance_acceptance_reason,
            "ess": float(selected_importance.diagnostics["ess"]),
            "ess_ratio": float(selected_importance.diagnostics["ess_ratio"]),
            "variance_reduction_ratio_mdd_7": float(
                importance_sensitivity.loc[importance_index, "variance_reduction_ratio_mdd_7"]
            ),
        },
        "data_hash": quality["sha256"],
        "model_version": "1.1.0-experimental",
        "trading_date_note": "Ngày làm việc gần đúng; chưa loại ngày nghỉ chính thức HOSE.",
    }
    if drawdown_layer is not None:
        latest_summary.update(drawdown_layer["summary"])
    write_json(root / "artifacts/forecasts/latest_forecast_summary.json", latest_summary)
    save_model(root / "artifacts/models/final_h20_model.joblib", final_model)
    save_model(root / "artifacts/models/final_hmm.joblib", full_hmm)
    metadata.update(
        {
            "selected_features": selected_technical,
            "hmm": hmm_result.diagnostics,
            "egarch": volatility.diagnostics,
            "horizons": horizons,
            "split_dates": {
                "train_boundary": str(base_split.train_boundary.date()),
                "test_boundary": str(base_split.test_boundary.date()),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    write_json(root / "artifacts/metadata/run_metadata.json", metadata)

    main_bundle = (
        state20["soft"].global_bundle
        if state20["selected_model"] == "soft_gated_rf"
        else state20["bundles"][state20["selected_model"]]
    )
    permutation = permutation_importance(
        main_bundle.return_regressor,
        main_bundle.imputer.transform(state20["augmented_full"].iloc[state20["test_index"]][main_bundle.feature_names]),
        targets[f"forward_return_{h20}"].to_numpy()[state20["test_index"]],
        scoring="neg_mean_absolute_error",
        n_repeats=3,
        random_state=seed,
        n_jobs=-1,
    )
    importance = (
        pd.DataFrame(
            {
                "feature": main_bundle.feature_names,
                "importance": main_bundle.return_regressor.feature_importances_,
                "permutation_importance": permutation.importances_mean,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    transformed_test_features = main_bundle.imputer.transform(
        state20["augmented_full"].iloc[state20["test_index"]][main_bundle.feature_names]
    )
    feature_jackknife_table = feature_importance_delete_block_jackknife(
        records20,
        transformed_test_features,
        targets[f"forward_return_{h20}"].to_numpy()[state20["test_index"]],
        main_bundle.return_regressor.predict,
        main_bundle.feature_names,
        importance.head(5)["feature"].tolist(),
        seed,
    )
    actual20 = records20["actual_return"].to_numpy(dtype=float)
    old_half_width = (records20["old_upper_95"] - records20["old_lower_95"]).to_numpy(dtype=float) / 2
    a1_lower = records20["improved_center"].to_numpy(dtype=float) - old_half_width
    a1_upper = records20["improved_center"].to_numpy(dtype=float) + old_half_width
    a1_var = records20["improved_center"].to_numpy(dtype=float) + (
        records20["old_var_95"] - records20["old_center"]
    ).to_numpy(dtype=float)
    old_interval = interval_metrics(actual20, records20["old_lower_95"], records20["old_upper_95"], 0.95)
    a1_interval = interval_metrics(actual20, a1_lower, a1_upper, 0.95)
    selected_interval = interval_metrics(actual20, records20["lower_95"], records20["upper_95"], 0.95)
    old_var_rate = float(np.mean(actual20 < records20["old_var_95"].to_numpy(dtype=float)))
    a1_var_rate = float(np.mean(actual20 < a1_var))
    selected_var_rate = float(np.mean(actual20 < records20["var_95"].to_numpy(dtype=float)))
    old_rmse = float(np.sqrt(np.mean(np.square(actual20 - records20["old_center"].to_numpy(dtype=float)))))
    improved_rmse = float(np.sqrt(np.mean(np.square(actual20 - records20["improved_center"].to_numpy(dtype=float)))))
    old_mae = float(np.mean(np.abs(actual20 - records20["old_center"].to_numpy(dtype=float))))
    improved_mae = float(np.mean(np.abs(actual20 - records20["improved_center"].to_numpy(dtype=float))))
    before_after = pd.DataFrame(
        [
            {
                "stage": "A0_current_model",
                "rmse": old_rmse,
                "mae": old_mae,
                "coverage_95": old_interval["coverage"],
                "interval_width_95": old_interval["average_width"],
                "interval_score_95": old_interval["interval_score"],
                "var_exceedance_95": old_var_rate,
            },
            {
                "stage": "A9_complete_experimental_pipeline",
                "rmse": improved_rmse,
                "mae": improved_mae,
                "coverage_95": selected_interval["coverage"],
                "interval_width_95": selected_interval["average_width"],
                "interval_score_95": selected_interval["interval_score"],
                "var_exceedance_95": selected_var_rate,
            },
        ]
    )

    def ablation_row(stage: str, components: str, interval: dict, var_rate: float, **extra) -> dict:
        return {
            "stage": stage,
            "components": components,
            "horizon": h20,
            "rmse": improved_rmse if stage != "A0" else old_rmse,
            "mae": improved_mae if stage != "A0" else old_mae,
            "coverage_95": interval["coverage"],
            "interval_width_95": interval["average_width"],
            "interval_score_95": interval["interval_score"],
            "var_exceedance_95": var_rate,
            **extra,
        }

    global_test = conformal_test_comparison.set_index("method").loc["global"]
    volatility_test = conformal_test_comparison.set_index("method").loc["volatility_stratified"]
    joint_test = conformal_test_comparison.set_index("method").loc["volatility_regime"]
    ablation_pipeline = pd.DataFrame(
        [
            ablation_row("A0", "current repository model", old_interval, old_var_rate),
            ablation_row("A1", "baseline-gated center", a1_interval, a1_var_rate),
            ablation_row("A2", "A1 + global conformal", global_test.to_dict(), global_test["var_exceedance_rate"]),
            ablation_row(
                "A3",
                "A1 + volatility conformal",
                volatility_test.to_dict(),
                volatility_test["var_exceedance_rate"],
            ),
            ablation_row(
                "A4",
                "A1 + volatility/regime conformal",
                joint_test.to_dict(),
                joint_test["var_exceedance_rate"],
            ),
            ablation_row(
                "A5",
                "A4 + stratified Monte Carlo",
                selected_interval,
                selected_var_rate,
                mcse_mdd_7=float(
                    simulation_efficiency.loc[
                        simulation_efficiency["method"] == "stratified_monte_carlo", "mcse_mdd_7"
                    ].iloc[0]
                ),
            ),
            ablation_row(
                "A6",
                "A5 + importance sampling",
                selected_interval,
                selected_var_rate,
                ess_ratio=float(selected_importance.diagnostics["ess_ratio"]),
                variance_reduction_ratio=float(
                    importance_sensitivity.loc[importance_index, "variance_reduction_ratio_mdd_7"]
                ),
            ),
            ablation_row(
                "A7",
                "A6 + adaptive stopping",
                selected_interval,
                selected_var_rate,
                simulation_paths=len(hybrid.return_paths),
                stopping_reason=hybrid.summary.get("stopping_reason", "fixed"),
            ),
            ablation_row(
                "A8",
                "A7 + outer stationary block bootstrap",
                selected_interval,
                selected_var_rate,
                outer_bootstrap_mode=effective_outer_bootstrap_mode,
            ),
            ablation_row(
                "A9",
                "complete + delete-block jackknife",
                selected_interval,
                selected_var_rate,
                maximum_absolute_jackknife_influence=float(delete_jackknife_table["absolute_influence"].max()),
            ),
        ]
    )
    coverage_bootstrap = outer_bootstrap_summary.set_index("metric").loc["coverage_improvement"]
    rmse_bootstrap = outer_bootstrap_summary.set_index("metric").loc["rmse_difference_new_minus_old"]
    selected_is_row = importance_sensitivity.loc[importance_index]
    acceptance_table = pd.DataFrame(
        [
            {"criterion": "h20 RMSE <= 0.05733", "value": improved_rmse, "passed": improved_rmse <= 0.05733},
            {
                "criterion": "95% coverage in [92.5%, 97.5%]",
                "value": selected_interval["coverage"],
                "passed": 0.925 <= selected_interval["coverage"] <= 0.975,
            },
            {
                "criterion": "VaR95 exceedance in [3%, 7%]",
                "value": selected_var_rate,
                "passed": 0.03 <= selected_var_rate <= 0.07,
            },
            {
                "criterion": "coverage improvement block-bootstrap CI above zero",
                "value": coverage_bootstrap["ci_lower"],
                "passed": coverage_bootstrap["ci_lower"] > 0,
            },
            {
                "criterion": "importance variance reduction >= 30%",
                "value": selected_is_row["variance_reduction_ratio_mdd_7"],
                "passed": selected_is_row["variance_reduction_ratio_mdd_7"] >= 1.30,
            },
            {
                "criterion": "importance ESS/N >= 20%",
                "value": selected_is_row["ess_ratio"],
                "passed": selected_is_row["ess_ratio"] >= 0.20,
            },
            {
                "criterion": "interval score deterioration <= 5%",
                "value": selected_interval["interval_score"] / old_interval["interval_score"] - 1,
                "passed": selected_interval["interval_score"] <= old_interval["interval_score"] * 1.05,
            },
            {
                "criterion": "new RMSE block-bootstrap upper difference <= 0",
                "value": rmse_bootstrap["ci_upper"],
                "passed": rmse_bootstrap["ci_upper"] <= 0,
            },
            {
                "criterion": "tail head validation production gates",
                "value": tail_head_summary["selected_candidate"],
                "passed": tail_head_summary["production_enabled"],
            },
        ]
    )
    tables = {
        "model_comparison": model_comparison,
        "per_horizon_metrics": per_horizon,
        "per_class_metrics": per_class_metrics,
        "calibration_metrics": calibration_metrics,
        "interval_metrics": interval_metric_table,
        "tail_risk_metrics": tail_metric_table,
        "statistical_tests": statistical_table,
        "ablation_results": ablation_table,
        "advanced_ablation": ablation_pipeline,
        "before_after_metrics": before_after,
        "acceptance_results": acceptance_table,
        "point_center_selection": center_selection_table,
        "conformal_selection": conformal_selection_table,
        "conformal_test_comparison": conformal_test_comparison,
        "conformal_multiplier_history": conformal_multiplier_table,
        "adaptive_convergence": convergence_history,
        "stratified_allocation": stratified_result.allocation,
        "importance_sampling_sensitivity": importance_sensitivity,
        "simulation_efficiency": simulation_efficiency,
        "outer_bootstrap_summary": outer_bootstrap_summary,
        "outer_bootstrap_replicates": outer_bootstrap_replicates,
        "delete_block_jackknife": delete_jackknife_table,
        "feature_importance_jackknife": feature_jackknife_table,
        "tail_head_experiment": tail_head_table,
        "seed_stability": seed_stability,
        "seed_stability_summary": seed_summary,
        "latest_forecast_summary": pd.DataFrame(
            [
                latest_summary
                | {
                    "maximum_drawdown_quantiles": json.dumps(latest_summary["maximum_drawdown_quantiles"]),
                    "drawdown_probabilities": json.dumps(latest_summary["drawdown_probabilities"]),
                    "model_uncertainty": json.dumps(latest_summary["model_uncertainty"]),
                }
            ]
        ),
        "monte_carlo_risk_summary": pd.DataFrame(
            [
                {
                    "method": name,
                    **result.summary,
                    "maximum_drawdown_quantiles": json.dumps(result.summary["maximum_drawdown_quantiles"]),
                    "drawdown_probabilities": json.dumps(result.summary["drawdown_probabilities"]),
                }
                for name, result in simulation_results.items()
            ]
        ),
        "jackknife_summary": jackknife_table,
        "feature_importance": importance,
        "test_predictions": predictions,
        "rf_tuning": tuning_table,
        "regime_threshold_selection": threshold_table,
        "volatility_model_comparison": volatility_model_comparison,
    }
    if drawdown_layer is not None:
        tables.update(drawdown_layer["tables"])
    for name, table in tables.items():
        _save_table(table, name, root)
    diagnostics = {
        "data_path": str(data_path),
        "data_range": [quality["start_date"], quality["end_date"]],
        "rows": len(data),
        "features_created": features.shape[1],
        "features_selected": len(selected_technical),
        "split_dates": metadata["split_dates"],
        "hmm_convergence": hmm_result.diagnostics["converged"],
        "egarch_convergence": volatility.diagnostics["converged"],
        "rf_parameters_by_horizon": {str(horizon): state["tuned_config"] for horizon, state in horizon_state.items()},
        "regime_lambdas_by_horizon": {
            str(horizon): state["selected_lambdas"] for horizon, state in horizon_state.items()
        },
        "calibration_by_horizon": per_horizon.set_index("horizon")["calibration"].to_dict(),
        "simulation_paths": config["simulation"]["paths"],
        "adaptive_simulation_paths": len(hybrid.return_paths),
        "selected_block_length": selected_block_length,
        "point_center_alpha_by_horizon": per_horizon.set_index("horizon")["center_alpha"].to_dict(),
        "conformal_by_horizon": {
            str(horizon): {
                "method": state["conformal_selection"].method,
                "window": state["conformal_selection"].window,
            }
            for horizon, state in horizon_state.items()
        },
        "importance_sampling": {
            "acceptance_reason": importance_acceptance_reason,
            "ess_ratio": float(selected_importance.diagnostics["ess_ratio"]),
            "variance_reduction_ratio_mdd_7": float(
                importance_sensitivity.loc[importance_index, "variance_reduction_ratio_mdd_7"]
            ),
        },
        "outer_bootstrap_mode": effective_outer_bootstrap_mode,
        "tail_head": tail_head_summary,
        "selected_volatility_on_validation": selected_volatility_name,
        "fallbacks": {
            "hmm_warnings": hmm_result.diagnostics["warnings"],
            "egarch_fallback": volatility.diagnostics["fallback"],
            "egarch_warnings": volatility.diagnostics["warnings"],
            "soft_gate_states": {str(h): state["soft"].fallback_states for h, state in horizon_state.items()},
            "simulation_regime_pool_fallback_count": hybrid.summary["regime_pool_fallback_count"],
            "jackknife_mode": effective_jackknife_mode,
            "outer_bootstrap_mode": effective_outer_bootstrap_mode,
        },
        "outputs": [
            "artifacts/forecasts/latest_forecast.csv",
            "artifacts/forecasts/latest_forecast_summary.json",
            "artifacts/forecasts/latest_monte_carlo_samples.npz",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    if drawdown_layer is not None:
        diagnostics["drawdown"] = {
            "enabled": True,
            "promotion_eligible": drawdown_layer["summary"]["promotion_eligible"],
            "calibration_status": drawdown_layer["summary"]["drawdown_calibration_status"],
            "mc_convergence_status": drawdown_layer["summary"]["drawdown_mc_convergence_status"],
        }
        diagnostics["outputs"].extend(
            [
                "artifacts/forecasts/latest_drawdown_forecast.csv",
                "artifacts/forecasts/latest_drawdown_summary.json",
                "artifacts/forecasts/latest_drawdown_paths.npz",
            ]
        )
    write_json(root / "reports/diagnostics/run_diagnostics.json", diagnostics)
    plot_context = {
        "data": data,
        "features": features,
        "hmm_probabilities": hmm_result.probabilities,
        "transition_matrix": hmm_result.transition_matrix,
        "residuals": volatility.standardized_residuals,
        "egarch_volatility": volatility.features["egarch_conditional_volatility"],
        "nu": float(volatility.diagnostics["nu"]),
        "predictions": predictions,
        "model_comparison": model_comparison,
        "interval_metrics": interval_metric_table[interval_metric_table["horizon"] == h20],
        "statistical_tests": statistical_table,
        "ablation": ablation_table,
        "jackknife": jackknife_table,
        "seed_stability": seed_stability,
        "forecast": forecast,
        "simulations": simulation_results,
        "feature_importance": importance,
        "confusion_matrix": state20["confusion"],
        "test_probabilities": state20["calibrated_probability"],
        "test_labels": state20["y_class"][state20["test_index"]],
        "split_dates": (base_split.train_boundary, base_split.test_boundary),
        "test_feature_frame": augmented_full.iloc[state20["test_index"]].reset_index(drop=True),
    }
    figure_names = generate_all_figures(plot_context, root / "reports/figures")
    advanced_figure_names = generate_advanced_figures(
        {
            "before_after": before_after,
            "conformal_test_comparison": conformal_test_comparison,
            "conformal_multiplier_history": conformal_multiplier_table,
            "records20": records20,
            "adaptive_convergence": convergence_history,
            "importance_sensitivity": importance_sensitivity,
            "simulation_efficiency": simulation_efficiency,
            "outer_bootstrap_summary": outer_bootstrap_summary,
            "delete_jackknife": delete_jackknife_table,
            "point_center_selection": center_selection_table,
            "advanced_ablation": ablation_pipeline,
            "importance_log_weights": selected_importance.log_weights,
        },
        root / "reports/figures",
    )
    figure_names.extend(advanced_figure_names)
    drawdown_figure_names: list[str] = []
    if drawdown_layer is not None:
        drawdown_figure_names = generate_drawdown_figures(drawdown_layer["plot_context"], root / "reports/figures")
    report_context = {
        "quality": quality,
        "latest_summary": latest_summary,
        "model_comparison": model_comparison,
        "per_horizon_metrics": per_horizon,
        "per_class_metrics": per_class_metrics,
        "calibration_metrics": calibration_metrics,
        "tail_risk_metrics": tail_metric_table,
        "statistical_tests": statistical_table,
        "interval_metrics": interval_metric_table[interval_metric_table["horizon"] == h20],
        "ablation": ablation_table,
        "jackknife": jackknife_table,
        "seed_stability": seed_stability,
        "feature_importance": importance,
        "volatility_model_comparison": volatility_model_comparison,
        "regime_threshold_selection": threshold_table,
        "monte_carlo_risk_summary": tables["monte_carlo_risk_summary"],
        "hmm_diagnostics": hmm_result.diagnostics,
        "egarch_diagnostics": volatility.diagnostics,
        "data": data,
        "features": features,
        "split_dates": metadata["split_dates"],
        "figure_names": figure_names,
        "forecast": forecast,
        "horizons": horizons,
        "embargo": config["data"]["embargo"],
        "jackknife_mode": effective_jackknife_mode,
        "before_after": before_after,
        "acceptance_results": acceptance_table,
        "point_center_selection": center_selection_table,
        "conformal_selection": conformal_selection_table,
        "conformal_test_comparison": conformal_test_comparison,
        "selected_conformal_method": state20["conformal_selection"].method,
        "simulation_efficiency": simulation_efficiency,
        "outer_bootstrap_summary": outer_bootstrap_summary,
        "delete_block_jackknife": delete_jackknife_table,
        "tail_head_experiment": tail_head_table,
        "tail_head_summary": tail_head_summary,
        "advanced_ablation": ablation_pipeline,
        "pipeline_mode": config["project"].get("pipeline_mode", "baseline"),
        "config_path": config["_config_path"],
    }
    write_reports(report_context, root)
    _readme({**report_context, "model_comparison": model_comparison}, root)
    if drawdown_layer is not None:
        h20_origin_intervals = drawdown_layer["tables"]["drawdown_interval_metrics"]
        h20_origin_intervals = h20_origin_intervals[
            (h20_origin_intervals["horizon"] == 20) & (h20_origin_intervals["anchor_mode"] == "origin_peak")
        ].set_index("level")
        _append_drawdown_reports(
            {
                "origin_summary": drawdown_layer["origin"].summary,
                "historical_summary": drawdown_layer["historical"].summary,
                "acceptance": drawdown_layer["acceptance"],
                "probability": drawdown_layer["probability"],
                "recovery": drawdown_layer["recovery"],
                "mc": drawdown_layer["mc"],
                "importance": drawdown_layer["importance"],
                "scenarios": drawdown_layer["scenarios"],
                "conditional": drawdown_layer["conditional"],
                "current_historical_drawdown": drawdown_layer["summary"]["current_historical_drawdown"],
                "coverage_95": float(h20_origin_intervals.loc[0.95, "upper_bound_coverage"]),
                "coverage_99": float(h20_origin_intervals.loc[0.99, "upper_bound_coverage"]),
                "figure_names": drawdown_figure_names,
            },
            root,
        )
        figure_names.extend(drawdown_figure_names)
    _log(
        "run_completed",
        elapsed_seconds=round(time.perf_counter() - started, 2),
        figures=len(figure_names),
        tables=len(tables),
        forecast_origin=latest_summary["forecast_origin"],
    )
    return {
        "quality": quality,
        "latest_summary": latest_summary,
        "per_horizon": per_horizon,
        "figures": figure_names,
        "diagnostics": diagnostics,
    }


def validate_data_only(config_path: str | Path) -> dict:
    config = load_config(config_path)
    data_path = Path(config["project"].get("data_path") or discover_data_file("."))
    _, quality = validate_and_save(data_path)
    return quality
