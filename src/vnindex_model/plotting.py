"""High-resolution figures generated only from observed experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t
from statsmodels.graphics.tsaplots import plot_acf

from .simulation import maximum_drawdown
from .targets import CLASS_NAMES

plt.style.use("seaborn-v0_8-whitegrid")


def _save(fig, root: Path, name: str, important: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(root / f"{name}.png", dpi=180, bbox_inches="tight")
    if important:
        fig.savefig(root / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def generate_all_figures(context: dict, output_dir: str | Path = "reports/figures") -> list[str]:
    root = Path(output_dir)
    data: pd.DataFrame = context["data"]
    features: pd.DataFrame = context["features"]
    hmm_prob: pd.DataFrame = context["hmm_probabilities"]
    transition = np.asarray(context["transition_matrix"])
    residuals = np.asarray(context["residuals"])
    predictions: pd.DataFrame = context["predictions"]
    comparison: pd.DataFrame = context["model_comparison"]
    intervals: pd.DataFrame = context["interval_metrics"]
    tests: pd.DataFrame = context["statistical_tests"]
    ablation: pd.DataFrame = context["ablation"]
    jackknife: pd.DataFrame = context["jackknife"]
    seeds: pd.DataFrame = context["seed_stability"]
    forecast: pd.DataFrame = context["forecast"]
    simulations = context["simulations"]
    importance: pd.DataFrame = context["feature_importance"]
    confusion = np.asarray(context["confusion_matrix"])
    class_prob = np.asarray(context["test_probabilities"])
    class_labels = np.asarray(context["test_labels"])
    train_end, valid_end = context["split_dates"]
    dates = pd.to_datetime(data["date"])
    close = data["close"]
    returns = np.log(close).diff()
    names: list[str] = []

    def save(fig, name, important=False):
        _save(fig, root, name, important)
        names.append(name)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, close, lw=1)
    ax.axvspan(dates.iloc[0], train_end, alpha=0.12, color="green", label="Train")
    ax.axvspan(train_end, valid_end, alpha=0.12, color="orange", label="Validation")
    ax.axvspan(valid_end, dates.iloc[-1], alpha=0.12, color="red", label="Test")
    ax.set(title="Lịch sử VN-Index và phân đoạn thời gian", ylabel="Điểm")
    ax.legend()
    save(fig, "01_data_splits")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, returns, lw=0.6)
    ax.axhline(0, color="black", lw=0.7)
    ax.set(title="Log-return theo thời gian", ylabel="Log-return")
    save(fig, "02_log_returns")
    values = returns.dropna().to_numpy()
    grid = np.linspace(np.quantile(values, 0.005), np.quantile(values, 0.995), 300)
    scale = values.std()
    nu = context["nu"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=80, density=True, alpha=0.45, label="Thực nghiệm")
    ax.plot(grid, norm.pdf(grid, values.mean(), scale), label="Gaussian")
    ax.plot(grid, t.pdf((grid - values.mean()) / scale, nu) / scale, label=f"Student-t ν={nu:.1f}")
    ax.legend()
    ax.set(title="Phân phối return và phân phối tham chiếu")
    save(fig, "03_return_distribution")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, features["rolling_volatility_20"], label="20 phiên")
    ax.plot(dates, features["rolling_volatility_60"], label="60 phiên")
    ax.legend()
    ax.set(title="Biến động lăn", ylabel="Độ lệch chuẩn")
    save(fig, "04_rolling_volatility")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dates, features["current_drawdown"], 0, color="firebrick", alpha=0.6)
    ax.set(title="Drawdown lịch sử", ylabel="Drawdown")
    save(fig, "05_historical_drawdown")
    states = hmm_prob.filter(like="hmm_probability_").to_numpy().argmax(axis=1)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, close, color="black", lw=0.8)
    scatter = ax.scatter(dates, close, c=states, s=3, cmap="viridis")
    fig.colorbar(scatter, ax=ax, label="Filtered state")
    ax.set(title="Filtered HMM phủ trên VN-Index")
    save(fig, "06_hmm_regimes", True)
    fig, ax = plt.subplots(figsize=(12, 5))
    [ax.plot(dates, hmm_prob[f"hmm_probability_{i}"], label=f"State {i}", lw=0.8) for i in range(transition.shape[0])]
    ax.legend(ncol=transition.shape[0])
    ax.set(title="Filtered regime probabilities", ylim=(0, 1))
    save(fig, "07_filtered_probabilities")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(transition, cmap="Blues", vmin=0, vmax=1)
    [
        ax.text(j, i, f"{transition[i, j]:.2f}", ha="center", va="center")
        for i in range(len(transition))
        for j in range(len(transition))
    ]
    fig.colorbar(im, ax=ax)
    ax.set(title="Ma trận chuyển trạng thái", xlabel="Đến state", ylabel="Từ state")
    save(fig, "08_transition_matrix")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, context["egarch_volatility"], lw=0.8)
    ax.set(title="EGARCH conditional volatility", ylabel="Sigma")
    save(fig, "09_egarch_volatility")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(residuals[np.isfinite(residuals)], bins=70)
    axes[0].set_title("Standardized residuals")
    sorted_r = np.sort(residuals[np.isfinite(residuals)])
    theoretical = norm.ppf((np.arange(len(sorted_r)) + 0.5) / len(sorted_r))
    axes[1].scatter(theoretical, sorted_r, s=4)
    axes[1].set_title("QQ so với Gaussian")
    save(fig, "10_residual_diagnostics")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_acf(residuals[np.isfinite(residuals)], lags=30, ax=axes[0], zero=False)
    plot_acf(np.square(residuals[np.isfinite(residuals)]), lags=30, ax=axes[1], zero=False)
    axes[0].set_title("ACF residual")
    axes[1].set_title("ACF residual bình phương")
    save(fig, "11_residual_acf")
    top = importance.head(20).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set(title="Random Forest feature importance")
    save(fig, "12_feature_importance")
    permutation_column = "permutation_importance" if "permutation_importance" in importance else "importance"
    grouped = (
        importance.assign(group=importance["feature"].str.split("_").str[0])
        .groupby("group", as_index=False)[permutation_column]
        .sum()
        .nlargest(15, permutation_column)
        .sort_values(permutation_column)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(grouped["group"], grouped[permutation_column])
    ax.set(title="Grouped feature importance")
    save(fig, "13_grouped_permutation_importance")
    key = importance.iloc[0]["feature"]
    sample = context["test_feature_frame"][[key]].copy()
    sample["prediction"] = predictions.loc[predictions["horizon"] == 20, "predicted_return"].to_numpy()[: len(sample)]
    sample["bin"] = pd.qcut(sample[key], 10, duplicates="drop")
    partial = sample.groupby("bin", observed=True).agg(x=(key, "mean"), y=("prediction", "mean"))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(partial["x"], partial["y"], marker="o")
    ax.set(title=f"Hiệu ứng tích lũy gần đúng: {key}", xlabel=key, ylabel="Return dự báo")
    save(fig, "14_partial_dependence")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap="Blues")
    [ax.text(j, i, str(confusion[i, j]), ha="center", va="center") for i in range(4) for j in range(4)]
    ax.set_xticks(range(4), CLASS_NAMES, rotation=30)
    ax.set_yticks(range(4), CLASS_NAMES)
    ax.set(title="Confusion matrix test", xlabel="Dự báo", ylabel="Thực tế")
    fig.colorbar(im, ax=ax)
    save(fig, "15_confusion_matrix")
    fig, ax = plt.subplots(figsize=(6, 5))
    bins = np.linspace(0, 1, 8)
    [
        ax.plot(
            [
                (
                    class_prob[(class_prob[:, i] >= left) & (class_prob[:, i] < right), i].mean()
                    if ((class_prob[:, i] >= left) & (class_prob[:, i] < right)).any()
                    else np.nan
                )
                for left, right in zip(bins[:-1], bins[1:], strict=True)
            ],
            [
                (
                    np.mean(class_labels[(class_prob[:, i] >= left) & (class_prob[:, i] < right)] == name)
                    if ((class_prob[:, i] >= left) & (class_prob[:, i] < right)).any()
                    else np.nan
                )
                for left, right in zip(bins[:-1], bins[1:], strict=True)
            ],
            marker="o",
            label=name,
        )
        for i, name in enumerate(CLASS_NAMES)
    ]
    ax.plot([0, 1], [0, 1], "k--")
    ax.legend()
    ax.set(title="Reliability diagram", xlabel="Xác suất dự báo", ylabel="Tần suất thực tế")
    save(fig, "16_reliability_diagram", True)
    p20 = predictions[predictions["horizon"] == 20]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pd.to_datetime(p20["date"]), p20["actual_return"], label="Thực tế", lw=0.8)
    ax.plot(pd.to_datetime(p20["date"]), p20["predicted_return"], label="Mô hình", lw=0.8)
    ax.legend()
    ax.set(title="Actual versus predicted return h=20")
    save(fig, "17_actual_predicted_return")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pd.to_datetime(p20["date"]), p20["actual_price"], label="Thực tế")
    ax.plot(pd.to_datetime(p20["date"]), p20["predicted_price"], label="Dự báo")
    ax.legend()
    ax.set(title="Actual versus predicted price h=20")
    save(fig, "18_actual_predicted_price")
    rolling = (p20["actual_return"] - p20["predicted_return"]).abs().rolling(60).mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(pd.to_datetime(p20["date"]), rolling)
    ax.set(title="Rolling MAE 60 dự báo h=20")
    save(fig, "19_rolling_error")
    best = comparison[comparison["horizon"] == 20].sort_values("rmse_return").head(12)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(best["model"], best["rmse_return"])
    ax.tick_params(axis="x", rotation=45)
    ax.set(title="So sánh RMSE return h=20")
    save(fig, "20_model_comparison")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(intervals["level"].astype(str), intervals["coverage"])
    axes[0].plot(intervals["level"].astype(str), intervals["level"], "ko--")
    axes[0].set_title("Coverage")
    axes[1].bar(intervals["level"].astype(str), intervals["average_width"])
    axes[1].set_title("Interval width")
    save(fig, "21_interval_coverage_width")
    fig, ax = plt.subplots(figsize=(10, 5))
    view = tests[tests["horizon"] == 20]
    ax.barh(view["baseline"], view["dm_statistic"])
    ax.axvline(-1.96, color="red", ls="--")
    ax.axvline(1.96, color="red", ls="--")
    ax.set(title="Diebold-Mariano h=20")
    save(fig, "22_diebold_mariano")
    fig, ax = plt.subplots(figsize=(10, 5))
    view = ablation[ablation["horizon"] == 20].sort_values("rmse_return")
    ax.bar(view["component"], view["rmse_return"])
    ax.tick_params(axis="x", rotation=50)
    ax.set(title="Ablation study h=20")
    save(fig, "23_ablation_study")
    hybrid = simulations["hybrid"]
    sample = hybrid.price_paths[: min(100, len(hybrid.price_paths))]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(1, sample.shape[1] + 1), sample.T, alpha=0.08, color="steelblue")
    ax.set(title="Các đường Monte Carlo đại diện", xlabel="Phiên", ylabel="VN-Index")
    save(fig, "24_monte_carlo_paths")
    f = forecast
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(f["step"], f["lower_95"], f["upper_95"], alpha=0.15, label="95%")
    ax.fill_between(f["step"], f["lower_90"], f["upper_90"], alpha=0.18, label="90%")
    ax.fill_between(f["step"], f["lower_80"], f["upper_80"], alpha=0.22, label="80%")
    ax.fill_between(f["step"], f["lower_50"], f["upper_50"], alpha=0.3, label="50%")
    ax.plot(f["step"], f["median"], color="black", label="Median")
    ax.legend()
    ax.set(title="Hybrid Monte Carlo fan chart", xlabel="Phiên", ylabel="VN-Index")
    save(fig, "25_fan_chart", True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(f["step"], f["mean"], label="Mean")
    ax.plot(f["step"], f["median"], label="Median")
    ax.plot(f["step"], f["lower_90"], ls="--", label="Lower 90")
    ax.plot(f["step"], f["upper_90"], ls="--", label="Upper 90")
    ax.legend()
    ax.set(title="Mean, median và quantile forecast")
    save(fig, "26_forecast_quantiles")
    terminal = hybrid.price_paths[:, -1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(terminal, bins=70)
    ax.axvline(np.median(terminal), color="black", label="Median")
    ax.legend()
    ax.set(title="Phân phối terminal price")
    save(fig, "27_terminal_price_distribution")
    terminal_r = np.exp(hybrid.return_paths.sum(axis=1)) - 1
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(terminal_r, bins=70)
    ax.axvline(0, color="black")
    ax.set(title="Phân phối terminal return")
    save(fig, "28_terminal_return_distribution")
    mdd = maximum_drawdown(
        np.column_stack([np.full(len(hybrid.price_paths), data["close"].iloc[-1]), hybrid.price_paths])
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mdd, bins=70)
    ax.set(title="Phân phối maximum drawdown")
    save(fig, "29_maximum_drawdown_distribution", True)
    summary = hybrid.summary
    fig, ax = plt.subplots(figsize=(8, 4))
    keys = ["var_95", "expected_shortfall_95", "var_99", "expected_shortfall_99"]
    ax.bar(keys, [summary[k] for k in keys])
    ax.tick_params(axis="x", rotation=25)
    ax.set(title="VaR và expected shortfall")
    save(fig, "30_var_expected_shortfall")
    probs = summary["drawdown_probabilities"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(probs), list(probs.values()))
    ax.set(title="Xác suất drawdown vượt ngưỡng", ylabel="Xác suất")
    save(fig, "31_drawdown_probabilities")
    fig, ax = plt.subplots(figsize=(10, 5))
    [ax.plot(f["step"], f[f"probability_{name.lower()}"], label=name) for name in CLASS_NAMES]
    ax.legend()
    ax.set(title="Regime probability trong forecast horizon", ylim=(0, 1))
    save(fig, "32_forecast_regime_probabilities")
    fig, ax = plt.subplots(figsize=(8, 5))
    [
        ax.hist(np.exp(result.return_paths.sum(axis=1)) - 1, bins=60, alpha=0.35, density=True, label=name)
        for name, result in simulations.items()
    ]
    ax.legend()
    ax.set(title="Student-t, bootstrap và hybrid")
    save(fig, "33_simulation_comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        jackknife["metric"],
        jackknife["full_estimate"],
        yerr=[jackknife["full_estimate"] - jackknife["ci_lower"], jackknife["ci_upper"] - jackknife["full_estimate"]],
        fmt="o",
    )
    ax.tick_params(axis="x", rotation=40)
    ax.set(title="Block jackknife sensitivity")
    save(fig, "34_jackknife_sensitivity")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(seeds["seed"].astype(str), seeds["rmse_return"], fmt="o-")
    ax.set(title="Multi-seed stability h=20", ylabel="RMSE")
    save(fig, "35_seed_stability")
    recent = data.tail(120)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(pd.to_datetime(recent["date"]), recent["close"], label="Lịch sử")
    future_dates = pd.to_datetime(f["estimated_trading_date"])
    ax.fill_between(future_dates, f["lower_95"], f["upper_95"], alpha=0.18)
    ax.plot(future_dates, f["median"], label="Forecast median")
    ax.legend()
    ax.set(title="Forecast mới nhất 20 phiên")
    save(fig, "36_latest_forecast", True)
    return names


def generate_advanced_figures(context: dict, output_dir: str | Path = "reports/figures") -> list[str]:
    """Create calibration and numerical-efficiency figures from experiment tables."""
    root = Path(output_dir)
    before_after: pd.DataFrame = context["before_after"]
    conformal: pd.DataFrame = context["conformal_test_comparison"]
    multiplier: pd.DataFrame = context["conformal_multiplier_history"]
    records: pd.DataFrame = context["records20"]
    convergence: pd.DataFrame = context["adaptive_convergence"]
    sensitivity: pd.DataFrame = context["importance_sensitivity"]
    efficiency: pd.DataFrame = context["simulation_efficiency"]
    outer: pd.DataFrame = context["outer_bootstrap_summary"]
    jackknife: pd.DataFrame = context["delete_jackknife"]
    alpha: pd.DataFrame = context["point_center_selection"]
    ablation: pd.DataFrame = context["advanced_ablation"]
    log_weights = np.asarray(context["importance_log_weights"], dtype=float)
    names: list[str] = []

    def save(fig, name):
        _save(fig, root, name)
        names.append(name)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(before_after["stage"], before_after["coverage_95"])
    ax.axhspan(0.925, 0.975, alpha=0.15, color="green", label="Acceptance")
    ax.axhline(0.95, color="black", ls="--")
    ax.tick_params(axis="x", rotation=20)
    ax.set(title="Before/after 95% interval coverage", ylabel="Coverage")
    ax.legend()
    save(fig, "37_before_after_coverage")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(conformal["average_width"], conformal["coverage"], s=70)
    for row in conformal.itertuples():
        ax.annotate(row.method, (row.average_width, row.coverage), fontsize=8)
    ax.axhline(0.95, color="black", ls="--")
    ax.set(title="Coverage-width frontier", xlabel="Average width", ylabel="Coverage")
    save(fig, "38_coverage_width_frontier")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(conformal["method"], conformal["interval_score"])
    ax.tick_params(axis="x", rotation=25)
    ax.set(title="Interval score theo conformal method", ylabel="Interval score")
    save(fig, "39_conformal_interval_score")

    h20_multiplier = multiplier[multiplier["horizon"] == 20]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(pd.to_datetime(h20_multiplier["date"]), h20_multiplier["multiplier_95"], lw=0.8)
    ax.set(title="Sequential conformal multiplier 95%", ylabel="Multiplier")
    save(fig, "40_sequential_conformal_multiplier")

    fig, ax = plt.subplots(figsize=(7, 4))
    h20_multiplier.boxplot(column="multiplier_95", by="volatility_bin", ax=ax)
    fig.suptitle("")
    ax.set(title="Conformal multiplier theo volatility stratum", xlabel="Volatility bin", ylabel="Multiplier")
    save(fig, "41_multiplier_by_volatility_stratum")

    rolling_var = (records["actual_return"] < records["conformal_var_95"]).rolling(120).mean()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(pd.to_datetime(records["date"]), rolling_var)
    ax.axhline(0.05, color="black", ls="--")
    ax.set(title="Rolling VaR 95% exceedance (120 records)", ylabel="Exceedance")
    save(fig, "42_var_exceedance_rolling")

    fig, ax = plt.subplots(figsize=(8, 4))
    if len(convergence):
        for column in ["probability_negative", "probability_mdd_3", "probability_mdd_5", "probability_mdd_7"]:
            ax.plot(convergence["paths"], convergence[column], marker="o", label=column)
    ax.legend(fontsize=7)
    ax.set(title="Monte Carlo probability convergence", xlabel="Paths", ylabel="Probability")
    save(fig, "43_mc_probability_convergence")

    fig, ax = plt.subplots(figsize=(8, 4))
    if len(convergence):
        ax.plot(convergence["paths"], convergence["max_probability_mcse"], marker="o")
    ax.set(title="MCSE theo số paths", xlabel="Paths", ylabel="Maximum probability MCSE")
    save(fig, "44_mcse_by_paths")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sensitivity["transition_strength"], sensitivity["ess_ratio"], marker="o")
    ax.axhline(0.20, color="red", ls="--")
    ax.set(title="ESS theo proposal strength", xlabel="Transition tilt", ylabel="ESS/N")
    save(fig, "45_ess_by_proposal_strength")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(log_weights, bins=70)
    ax.set(title="Importance log-weight distribution", xlabel="log weight")
    save(fig, "46_importance_weight_distribution")

    fig, ax = plt.subplots(figsize=(8, 4))
    view = efficiency[efficiency["method"].isin(["naive_monte_carlo", "importance_sampling"])]
    ax.bar(view["method"], view["mcse_mdd_7"])
    ax.set(title="Naive Monte Carlo versus importance sampling", ylabel="MCSE P(MDD<-7%)")
    save(fig, "47_naive_vs_importance")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(efficiency["method"], efficiency["tail_events_generated_mdd_7"].fillna(0))
    ax.tick_params(axis="x", rotation=25)
    ax.set(title="Tail-event count theo phương pháp", ylabel="Generated MDD<-7% events")
    save(fig, "48_tail_event_count")

    fig, ax = plt.subplots(figsize=(9, 5))
    outer_view = outer.nlargest(10, "standard_error").sort_values("standard_error")
    ax.barh(outer_view["metric"], outer_view["standard_error"])
    ax.set(title="Outer-bootstrap uncertainty decomposition", xlabel="Bootstrap standard error")
    save(fig, "49_outer_bootstrap_uncertainty")

    fig, ax = plt.subplots(figsize=(9, 5))
    influence = jackknife.nlargest(15, "absolute_influence").sort_values("absolute_influence")
    labels = influence["block_type"] + ":" + influence["block_label"] + ":" + influence["metric"]
    ax.barh(labels, influence["absolute_influence"])
    ax.set(title="Delete-block jackknife influence ranking")
    save(fig, "50_delete_block_jackknife_influence")

    fig, ax = plt.subplots(figsize=(8, 4))
    h20_alpha = alpha[alpha["horizon"] == 20]
    ax.errorbar(h20_alpha["alpha"], h20_alpha["rmse"], yerr=h20_alpha["rmse_standard_error"], marker="o")
    selected = h20_alpha[h20_alpha["selected"]]
    ax.scatter(selected["alpha"], selected["rmse"], color="red", zorder=3, label="Selected")
    ax.legend()
    ax.set(title="Validation alpha của baseline-gated center", xlabel="Alpha ML", ylabel="Validation RMSE")
    save(fig, "51_validation_alpha_blend")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(ablation["stage"], ablation["coverage_95"])
    ax.axhline(0.95, color="black", ls="--")
    ax.set(title="Ablation A0-A9: 95% coverage", ylabel="Coverage")
    save(fig, "52_advanced_ablation")

    fig, ax = plt.subplots(figsize=(11, 4))
    dates = pd.to_datetime(records["date"])
    ax.plot(dates, records["actual_return"], label="Actual", lw=0.7)
    ax.plot(dates, records["old_center"], label="Current model", lw=0.8)
    ax.plot(dates, records["improved_center"], label="Improved center", lw=0.8)
    ax.legend()
    ax.set(title="Current and improved model on identical h=20 test periods", ylabel="Return")
    save(fig, "53_current_vs_improved_same_test")
    return names


def generate_drawdown_figures(context: dict, output_dir: str | Path = "reports/figures") -> list[str]:
    """Generate figures 54-70 from drawdown path and backtest artifacts."""
    root = Path(output_dir)
    origin = context["origin_term"]
    historical = context["historical_term"]
    first_passage = context["first_passage"]
    recovery = context["recovery"]
    interval = context["interval"]
    probability = context["probability"]
    mc = context["mc_uncertainty"]
    importance = context["importance"]
    scenarios = context["scenarios"]
    duration = context["duration"]
    details = context["details"]
    simultaneous = context["simultaneous"]
    calibration = context["calibration"]
    names: list[str] = []

    def save(fig, name):
        _save(fig, root, name)
        names.append(name)

    def fan(table, title, name, running=False):
        prefix = "running_mdd" if running else "drawdown"
        median = f"{prefix}_median"
        lower = "drawdown_q025" if not running else "running_mdd_median"
        upper = "drawdown_q975" if not running else "running_mdd_q95"
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(table["step"], table[lower], table[upper], alpha=0.2, label="Band")
        ax.plot(table["step"], table[median], lw=1.8, label="Median")
        ax.set(title=title, xlabel="Phiên", ylabel="Drawdown severity")
        ax.legend()
        save(fig, name)

    fan(origin, "Drawdown fan chart — origin peak", "54_drawdown_fan_chart_origin_peak")
    fan(historical, "Drawdown fan chart — historical peak", "55_drawdown_fan_chart_historical_peak")
    fan(origin, "Running maximum drawdown severity", "56_running_maximum_drawdown_fan_chart", running=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for column in [value for value in origin if value.startswith("probability_breach_")]:
        ax.plot(origin["step"], origin[column], label=column.replace("probability_breach_", ""))
    ax.set(title="Cumulative drawdown breach probabilities", xlabel="Phiên", ylabel="Xác suất", ylim=(0, 1))
    ax.legend(title="Ngưỡng %")
    save(fig, "57_drawdown_breach_probability_term_structure")

    fig, ax = plt.subplots(figsize=(9, 5))
    passage_values = first_passage.dropna(subset=["first_passage_time"])
    for threshold, group in passage_values.groupby("threshold"):
        ax.hist(
            group["first_passage_time"],
            bins=np.arange(0.5, origin["step"].max() + 1.5),
            alpha=0.35,
            label=f"{threshold:.0%}",
        )
    ax.set(title="Conditional first-passage time distribution", xlabel="Phiên breach", ylabel="Số paths")
    ax.legend()
    save(fig, "58_first_passage_time_distribution")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(origin["step"], origin["running_mdd_q90"], label="MDaR 90")
    ax.plot(origin["step"], origin["running_mdd_q95"], label="MDaR 95")
    ax.plot(origin["step"], origin["running_mdd_q99"], label="MDaR 99")
    ax.set(title="Maximum Drawdown at Risk term structure", xlabel="Phiên", ylabel="Severity")
    ax.legend()
    save(fig, "59_mdar_ced_term_structure")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(recovery["step"], recovery["probability_recovery_by_step"], where="post", label="Recovery CDF")
    ax.step(recovery["step"], recovery["unrecovered_survival"], where="post", label="Unrecovered survival")
    ax.set(title="Historical-peak recovery probability", xlabel="Phiên", ylabel="Xác suất", ylim=(0, 1))
    ax.legend()
    save(fig, "60_recovery_probability_curve")

    reliability = probability.groupby(["method", "threshold"], as_index=False).agg(
        mean_probability=("mean_probability", "mean"), event_rate=("event_rate", "mean")
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    for method, group in reliability.groupby("method"):
        ax.plot(group["mean_probability"], group["event_rate"], marker="o", label=method)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(title="Drawdown breach reliability", xlabel="Xác suất dự báo", ylabel="Tần suất thực tế")
    ax.legend(fontsize=7)
    save(fig, "61_drawdown_probability_reliability")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(details["median_hybrid_monte_carlo"], details["actual_severity"], s=8, alpha=0.5)
    limit = float(max(details["median_hybrid_monte_carlo"].max(), details["actual_severity"].max()))
    ax.plot([0, limit], [0, limit], "k--")
    ax.set(title="Realized vs predicted median drawdown", xlabel="Predicted severity", ylabel="Realized severity")
    save(fig, "62_realized_vs_predicted_drawdown")

    selected_interval = interval[interval["method"] == "hybrid_direct_drawdown_conformal"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for horizon, group in selected_interval.groupby("horizon"):
        ax.plot(group["level"], group["upper_bound_coverage"], marker="o", label=f"h={horizon}")
    ax.plot([0.8, 1], [0.8, 1], "k--")
    ax.set(title="Direct drawdown upper-bound coverage", xlabel="Nominal", ylabel="OOS coverage")
    ax.legend(ncol=2)
    save(fig, "63_drawdown_upper_bound_coverage")

    fig, ax = plt.subplots(figsize=(10, 5))
    step = np.arange(1, len(simultaneous["center"]) + 1)
    ax.fill_between(
        step,
        simultaneous["simultaneous_lower"],
        simultaneous["simultaneous_upper"],
        alpha=0.2,
        label="Simultaneous 95%",
    )
    ax.fill_between(
        step, simultaneous["pointwise_lower"], simultaneous["pointwise_upper"], alpha=0.35, label="Pointwise 95%"
    )
    ax.plot(step, simultaneous["center"], color="black", label="Median")
    ax.set(title="Pointwise vs simultaneous drawdown band", xlabel="Phiên", ylabel="Severity")
    ax.legend()
    save(fig, "64_pointwise_vs_simultaneous_drawdown_band")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(origin["step"], origin["drawdown_median"], label="Origin peak")
    ax.plot(historical["step"], historical["drawdown_median"], label="Historical peak")
    ax.set(title="Origin-peak vs historical-peak drawdown", xlabel="Phiên", ylabel="Median severity")
    ax.legend()
    save(fig, "65_origin_vs_historical_peak_drawdown")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(mc["statistic"], mc["mcse"])
    ax.tick_params(axis="x", rotation=35)
    ax.set(title="Drawdown probability Monte Carlo error", ylabel="MCSE")
    save(fig, "66_drawdown_mcse_by_paths")

    selected_importance = importance[importance["selected"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(selected_importance["threshold"].astype(str), selected_importance["variance_reduction_ratio"])
    ax.axhline(1.3, color="red", linestyle="--", label="Acceptance 1.30x")
    ax.set(
        title="Threshold-specific importance-sampling efficiency", xlabel="MDD threshold", ylabel="Variance reduction"
    )
    ax.legend()
    save(fig, "67_drawdown_importance_sampling_efficiency")

    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = scenarios.sort_values("mdar_95")
    ax.barh(ordered["scenario"], ordered["mdar_95"], alpha=0.75)
    ax.scatter(ordered["ced_95"], ordered["scenario"], color="firebrick", label="CED 95")
    ax.set(title="Drawdown scenario risk cone", xlabel="Severity")
    ax.legend()
    save(fig, "68_drawdown_scenario_risk_cone")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        duration["maximum_underwater_duration"],
        bins=np.arange(0.5, origin["step"].max() + 1.5),
        alpha=0.55,
        label="Maximum",
    )
    ax.hist(
        duration["ending_underwater_duration"],
        bins=np.arange(0.5, origin["step"].max() + 1.5),
        alpha=0.55,
        label="Ending",
    )
    ax.set(title="Drawdown duration distribution", xlabel="Số phiên", ylabel="Paths")
    ax.legend()
    save(fig, "69_drawdown_duration_distribution")

    regime_calibration = calibration[
        (calibration["horizon"] == 20)
        & (calibration["anchor_mode"] == "origin_peak")
        & (calibration["level"] == 0.95)
        & (calibration["stratum_type"] == "regime")
    ].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(regime_calibration["stratum"].astype(str), regime_calibration["conditional_coverage"])
    ax.axhline(0.95, color="black", linestyle="--", label="Nominal 95%")
    ax.tick_params(axis="x", rotation=30)
    ax.set(title="OOS drawdown calibration by filtered regime", ylabel="Conditional coverage", ylim=(0, 1))
    ax.legend()
    save(fig, "70_drawdown_calibration_by_regime")
    return names
