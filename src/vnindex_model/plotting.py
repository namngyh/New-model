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
                class_prob[(class_prob[:, i] >= left) & (class_prob[:, i] < right), i].mean()
                if ((class_prob[:, i] >= left) & (class_prob[:, i] < right)).any()
                else np.nan
                for left, right in zip(bins[:-1], bins[1:], strict=True)
            ],
            [
                np.mean(class_labels[(class_prob[:, i] >= left) & (class_prob[:, i] < right)] == name)
                if ((class_prob[:, i] >= left) & (class_prob[:, i] < right)).any()
                else np.nan
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
