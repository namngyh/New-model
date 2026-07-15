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
from .bootstrap import choose_block_length
from .calibration import select_temporal_calibration
from .config import load_config
from .data import discover_data_file, validate_and_save
from .evaluation import classification_metrics, historical_var_tests, interval_metrics, point_metrics
from .features import build_features, select_train_features
from .hmm import fit_filtered_hmm
from .jackknife import block_jackknife_table
from .persistence import run_metadata, save_model, write_json
from .plotting import generate_all_figures
from .random_forest import (
    fit_forest_bundle,
    fit_soft_gated_forest,
    predict_bundle,
    predict_soft_gated,
)
from .reporting import write_reports
from .simulation import maximum_drawdown, simulate_paths
from .splits import purged_train_validation_test
from .statistical_tests import block_bootstrap_difference, diebold_mariano
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


def _readme(context: dict, root: Path) -> None:
    summary = context["latest_summary"]
    comparison: pd.DataFrame = context["model_comparison"]
    best = comparison.loc[
        comparison.groupby("horizon")["rmse_return"].idxmin(),
        ["horizon", "model", "rmse_return", "directional_accuracy"],
    ]
    main20 = context["per_horizon_metrics"].loc[context["per_horizon_metrics"]["horizon"] == 20].iloc[0]
    interval95 = (
        context["interval_metrics"]
        .loc[(context["interval_metrics"]["horizon"] == 20) & (context["interval_metrics"]["level"] == 0.95)]
        .iloc[0]
    )
    best20 = best[best["horizon"] == 20].iloc[0]
    readme = f"""# VN-Index Regime-Aware Random Forest và Hybrid Monte Carlo

Tác giả: **Nguyễn Hoài Nam**

Pipeline nghiên cứu tái lập để dự báo riêng lợi suất, mức điểm, trạng thái Bull/Sideway/Bear/Stress và phân phối rủi ro VN-Index. Kiến trúc chính kết hợp Filtered HMM, EGARCH Student-t, soft-gated Random Forest, regime-conditioned block bootstrap, hybrid Monte Carlo và block jackknife.

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
  F --> G[Temporal calibration]
  D --> H[Regime path]
  E --> I[Student-t và residual blocks]
  G --> H
  H --> J[Hybrid Monte Carlo]
  I --> J
  J --> K[VaR ES drawdown intervals]
  K --> L[Block jackknife và reports]
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

Ở h=20, mô hình chính có RMSE **{main20["rmse_return"]:.6f}**, Bear/Stress recall **{main20["recall_bear"]:.2%}/{main20["recall_stress"]:.2%}**, trong khi **{best20["model"]}** có RMSE **{best20["rmse_return"]:.6f}**. Interval 95% chỉ cover **{interval95["coverage"]:.2%}**. Do đó run này **không đủ bằng chứng để kết luận mô hình chính tốt hơn baseline** và tail calibration chưa đạt nominal.

## Forecast 20 phiên mới nhất

- Origin: {summary["forecast_origin"]}; close cuối: {summary["last_observed_close"]:.2f}.
- Terminal mean/median: {summary["expected_terminal_close"]:.2f} / {summary["median_terminal_close"]:.2f}.
- Xác suất tăng/giảm: {summary["probability_positive_return"]:.2%} / {summary["probability_negative_return"]:.2%}.
- VaR 95% và ES 95%: {summary["var_95"]:.2%} / {summary["expected_shortfall_95"]:.2%}.
- P(maximum drawdown vượt 5%): {summary["drawdown_probabilities"]["0.05"]:.2%}.
- Estimated trading dates dùng ngày làm việc gần đúng, chưa loại ngày nghỉ HOSE.

![Forecast mới nhất](reports/figures/36_latest_forecast.png)

![Fan chart](reports/figures/25_fan_chart.png)

![Monte Carlo paths](reports/figures/24_monte_carlo_paths.png)

![Filtered HMM regimes](reports/figures/06_hmm_regimes.png)

![Calibration](reports/figures/16_reliability_diagram.png)

![Drawdown distribution](reports/figures/29_maximum_drawdown_distribution.png)

## Cấu trúc và tái lập

- `src/vnindex_model/`: module dữ liệu, target, split, HMM, EGARCH, RF, calibration, simulation, bootstrap, jackknife, baseline, evaluation và reporting.
- `configs/`: quick/default/full với seed, paths và compute budget rõ ràng.
- `artifacts/`: model, metadata, latest forecast và NPZ samples.
- `reports/`: CSV/Markdown, 36 hình và hai báo cáo tiếng Việt.
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

        baseline = baseline_predictions(
            data["close"],
            features["log_return"],
            y_return,
            features[selected_technical].to_numpy(),
            train_index,
            test_index,
            horizon,
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
        horizon_rows.append(
            {
                "horizon": horizon,
                "selected_model": selected_model,
                "calibration": calibrator.method,
                **metrics_by_model[selected_model],
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
                "predicted_return": main_prediction["return"],
                "actual_price": actual_price,
                "predicted_price": current_price * np.exp(main_prediction["return"]),
                "actual_drawdown": y_drawdown[test_index],
                "predicted_drawdown": main_prediction["drawdown"],
                "actual_regime": y_class[test_index],
                "predicted_regime": predicted_labels,
            }
        )
        for class_position, class_name in enumerate(CLASS_NAMES):
            prediction_frame[f"probability_{class_name.lower()}"] = calibrated_probability[:, class_position]
        prediction_frames.append(prediction_frame)
        sigma = volatility.features["egarch_forecast_volatility"].iloc[test_index].to_numpy() * np.sqrt(horizon)
        residual_pool = volatility.standardized_residuals[train_index]
        residual_pool = residual_pool[np.isfinite(residual_pool)]
        nu = float(volatility.diagnostics["nu"])
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
        q95 = t.ppf(0.05, nu) * np.sqrt((nu - 2) / nu)
        q99 = t.ppf(0.01, nu) * np.sqrt((nu - 2) / nu)
        var95 = main_prediction["return"] + sigma * q95
        var99 = main_prediction["return"] + sigma * q99
        one_hot = np.column_stack([(y_class[test_index] == name).astype(int) for name in CLASS_NAMES])
        prediction_frame["brier_loss"] = np.sum((calibrated_probability - one_hot) ** 2, axis=1)
        prediction_frame["interval_95_covered"] = ((actual_return >= lower) & (actual_return <= upper)).astype(float)
        prediction_frame["var_95"] = var95
        prediction_frame["expected_shortfall_95"] = (
            main_prediction["return"] + sigma * residual_pool[residual_pool <= np.quantile(residual_pool, 0.05)].mean()
        )
        tests95 = historical_var_tests(actual_return, var95, 0.05)
        tests99 = historical_var_tests(actual_return, var99, 0.01)
        tail_rows.append(
            {
                "horizon": horizon,
                "var_95_mean": float(var95.mean()),
                "var_99_mean": float(var99.mean()),
                "expected_shortfall_95_mean": float(
                    np.mean(
                        main_prediction["return"]
                        + sigma * residual_pool[residual_pool <= np.quantile(residual_pool, 0.05)].mean()
                    )
                ),
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
            }
        )
        for baseline_name, baseline_prediction in baseline.items():
            dm = diebold_mariano(actual_return, main_prediction["return"], baseline_prediction, horizon)
            bootstrap = block_bootstrap_difference(
                actual_return,
                main_prediction["return"],
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
        }
        _log(
            "horizon_completed",
            horizon=horizon,
            selected_model=selected_model,
            calibration=calibrator.method,
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
    h20 = 20 if 20 in horizon_state else max(horizons)
    state20 = horizon_state[h20]
    records20 = predictions[predictions["horizon"] == h20].reset_index(drop=True)
    jackknife_table = block_jackknife_table(records20)
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
    simulation_results = {}
    for method in ["student_t", "regime_block_bootstrap", "hybrid"]:
        simulation_results["bootstrap" if method == "regime_block_bootstrap" else method] = simulate_paths(
            float(data["close"].iloc[-1]),
            config["simulation"]["horizon"],
            config["simulation"]["paths"],
            float(latest_prediction["return"][0] / h20),
            float(full_volatility.features["egarch_forecast_volatility"].iloc[-1]),
            float(full_volatility.diagnostics["nu"]),
            full_volatility.standardized_residuals,
            full_hmm.probabilities[full_hmm_columns].to_numpy(),
            full_hmm.transition_matrix,
            full_hmm.probabilities[full_hmm_columns].iloc[-1].to_numpy(),
            latest_probability,
            full_hmm.economic_labels,
            method=method,
            student_weight=config["simulation"]["student_weight"],
            block_length=selected_block_length,
            seed=seed,
        )
    hybrid = simulation_results["hybrid"]
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
        "data_hash": quality["sha256"],
        "model_version": "1.0.0",
        "trading_date_note": "Ngày làm việc gần đúng; chưa loại ngày nghỉ chính thức HOSE.",
    }
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
    tables = {
        "model_comparison": model_comparison,
        "per_horizon_metrics": per_horizon,
        "per_class_metrics": per_class_metrics,
        "calibration_metrics": calibration_metrics,
        "interval_metrics": interval_metric_table,
        "tail_risk_metrics": tail_metric_table,
        "statistical_tests": statistical_table,
        "ablation_results": ablation_table,
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
        "selected_block_length": selected_block_length,
        "selected_volatility_on_validation": selected_volatility_name,
        "fallbacks": {
            "hmm_warnings": hmm_result.diagnostics["warnings"],
            "egarch_fallback": volatility.diagnostics["fallback"],
            "egarch_warnings": volatility.diagnostics["warnings"],
            "soft_gate_states": {str(h): state["soft"].fallback_states for h, state in horizon_state.items()},
            "simulation_regime_pool_fallback_count": hybrid.summary["regime_pool_fallback_count"],
            "jackknife_mode": effective_jackknife_mode,
        },
        "outputs": [
            "artifacts/forecasts/latest_forecast.csv",
            "artifacts/forecasts/latest_forecast_summary.json",
            "artifacts/forecasts/latest_monte_carlo_samples.npz",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
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
    }
    write_reports(report_context, root)
    _readme({**report_context, "model_comparison": model_comparison}, root)
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
