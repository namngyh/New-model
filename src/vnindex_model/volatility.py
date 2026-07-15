"""EGARCH Student-t estimation, causal filtering, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


@dataclass
class VolatilityResult:
    features: pd.DataFrame
    standardized_residuals: np.ndarray
    diagnostics: dict[str, object]


def fit_arch_candidate(
    returns: pd.Series,
    train_index: np.ndarray,
    volatility: str,
    distribution: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit a fixed-parameter GARCH/EGARCH candidate and filter all rows causally."""
    values = returns.fillna(0.0).astype(float)
    try:
        model = arch_model(
            values.iloc[train_index] * 100,
            mean="Constant",
            vol=volatility,
            p=1,
            o=1 if volatility == "EGARCH" else 0,
            q=1,
            dist=distribution,
            rescale=False,
        ).fit(disp="off", show_warning=False)
        parameters = {str(key): float(value) for key, value in model.params.items()}
        if model.convergence_flag != 0:
            raise RuntimeError(f"convergence_flag={model.convergence_flag}")
        omega = parameters.get("omega", 0.01)
        alpha = parameters.get("alpha[1]", 0.1)
        gamma = parameters.get("gamma[1]", 0.0)
        beta = float(np.clip(parameters.get("beta[1]", 0.85), 0, 0.998))
        mean = parameters.get("mu", 0.0)
        log_variance = np.full(len(values), np.log(max((values.iloc[train_index] * 100).var(), 1e-4)))
        for row in range(1, len(values)):
            previous_return = values.iloc[row - 1] * 100 - mean
            if volatility == "EGARCH":
                standardized = previous_return / np.sqrt(np.exp(log_variance[row - 1]))
                log_variance[row] = (
                    omega + beta * log_variance[row - 1] + alpha * (abs(standardized) - 0.8) + gamma * standardized
                )
            else:
                variance = omega + alpha * previous_return**2 + beta * np.exp(log_variance[row - 1])
                log_variance[row] = np.log(max(variance, 1e-8))
            log_variance[row] = np.clip(log_variance[row], -20, 20)
        sigma = np.sqrt(np.exp(log_variance)) / 100
        plausible = np.isfinite(sigma).all() and np.median(sigma) < 0.2
        return sigma, {
            "model": f"{volatility} {distribution}",
            "converged": True,
            "plausible": plausible,
            "parameters": parameters,
            "warning": None if plausible else "volatility outside plausibility guardrail",
        }
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        return np.full(len(values), np.nan), {
            "model": f"{volatility} {distribution}",
            "converged": False,
            "plausible": False,
            "parameters": {},
            "warning": str(error),
        }


def standardized_student_t(
    rng: np.random.Generator, degrees_of_freedom: float, size: int | tuple[int, ...]
) -> np.ndarray:
    nu = max(float(degrees_of_freedom), 2.01)
    return rng.standard_t(nu, size=size) * np.sqrt((nu - 2) / nu)


def _fit_model(train_returns_percent: pd.Series, volatility: str):
    return arch_model(
        train_returns_percent,
        mean="Constant",
        vol=volatility,
        p=1,
        o=1 if volatility == "EGARCH" else 0,
        q=1,
        dist="StudentsT",
        rescale=False,
    ).fit(disp="off", show_warning=False)


def fit_egarch_student_t(returns: pd.Series, train_index: np.ndarray) -> VolatilityResult:
    values = returns.fillna(0.0).astype(float)
    train_percent = values.iloc[train_index] * 100
    fallback: str | None = None
    warnings: list[str] = []
    result = None
    try:
        result = _fit_model(train_percent, "EGARCH")
        if result.convergence_flag != 0:
            raise RuntimeError(f"convergence_flag={result.convergence_flag}")
        egarch_parameters = result.params
        if (
            float(egarch_parameters.get("nu", 8.0)) <= 2.1
            or abs(float(egarch_parameters.get("alpha[1]", 0.0))) > 5
            or abs(float(egarch_parameters.get("gamma[1]", 0.0))) > 5
            or abs(float(egarch_parameters.get("mu", 0.0))) > 5
        ):
            raise RuntimeError("EGARCH parameters outside plausibility guardrails")
        model_name = "EGARCH(1,1) Student-t"
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        warnings.append(f"EGARCH không hội tụ sạch: {error}")
        try:
            result = _fit_model(train_percent, "GARCH")
            fallback = "GARCH(1,1) Student-t"
            model_name = fallback
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as second_error:
            warnings.append(f"GARCH fallback thất bại: {second_error}; dùng EWMA")
            result = None
            fallback = "EWMA"
            model_name = fallback
    if result is None:
        sigma = (
            values.ewm(span=40, adjust=False, min_periods=5)
            .std()
            .fillna(values.expanding(2).std())
            .fillna(1e-4)
            .clip(lower=1e-6)
            .to_numpy()
        )
        standardized = (values / sigma).clip(-20, 20).to_numpy()
        parameters: dict[str, float] = {"nu": 8.0}
        converged = False
    else:
        parameters = {str(key): float(value) for key, value in result.params.items()}
        omega = parameters.get("omega", -0.1)
        alpha = parameters.get("alpha[1]", 0.1)
        gamma = parameters.get("gamma[1]", 0.0)
        beta = float(np.clip(parameters.get("beta[1]", 0.9), 0.0, 0.998))
        mean = parameters.get("mu", 0.0)
        log_variance = np.full(len(values), np.log(max(train_percent.var(), 1e-4)))
        standardized = np.zeros(len(values))
        expected_absolute = 0.8
        for row in range(1, len(values)):
            previous = (values.iloc[row - 1] * 100 - mean) / np.sqrt(np.exp(log_variance[row - 1]))
            standardized[row - 1] = previous
            if model_name.startswith("EGARCH"):
                log_variance[row] = (
                    omega
                    + beta * log_variance[row - 1]
                    + alpha * (abs(previous) - expected_absolute)
                    + gamma * previous
                )
            else:
                variance = (
                    omega + alpha * (values.iloc[row - 1] * 100 - mean) ** 2 + beta * np.exp(log_variance[row - 1])
                )
                log_variance[row] = np.log(max(variance, 1e-8))
            log_variance[row] = np.clip(log_variance[row], -20, 20)
        standardized[-1] = (values.iloc[-1] * 100 - mean) / np.sqrt(np.exp(log_variance[-1]))
        sigma = np.sqrt(np.exp(log_variance)) / 100
        converged = result.convergence_flag == 0
        empirical_scale = float(values.iloc[train_index].std())
        if not np.isfinite(sigma).all() or np.median(sigma) > max(0.20, 10 * empirical_scale):
            warnings.append("Conditional volatility outside plausibility guardrails; dùng EWMA")
            sigma = (
                values.ewm(span=40, adjust=False, min_periods=5)
                .std()
                .fillna(values.expanding(2).std())
                .fillna(1e-4)
                .clip(lower=1e-6)
                .to_numpy()
            )
            standardized = (values / sigma).clip(-20, 20).to_numpy()
            fallback = "EWMA after volatility guardrail"
            model_name = fallback
            converged = False
    residual_train = np.asarray(standardized)[train_index]
    residual_train = residual_train[np.isfinite(residual_train)]
    lags = min(20, max(1, len(residual_train) // 10))
    ljung = acorr_ljungbox(residual_train, lags=[lags], return_df=True)
    squared_ljung = acorr_ljungbox(residual_train**2, lags=[lags], return_df=True)
    arch_lm = het_arch(residual_train, nlags=lags)
    feature_frame = pd.DataFrame(index=returns.index)
    feature_frame["egarch_conditional_volatility"] = sigma
    feature_frame["egarch_forecast_volatility"] = pd.Series(sigma, index=returns.index).ewm(span=5, adjust=False).mean()
    feature_frame["egarch_standardized_residual"] = standardized
    feature_frame["student_t_degrees_freedom"] = parameters.get("nu", 8.0)
    diagnostics = {
        "model": model_name,
        "converged": converged,
        "fallback": fallback,
        "parameters": parameters,
        "nu": parameters.get("nu", 8.0),
        "ljung_box_pvalue": float(ljung["lb_pvalue"].iloc[0]),
        "squared_ljung_box_pvalue": float(squared_ljung["lb_pvalue"].iloc[0]),
        "arch_lm_pvalue": float(arch_lm[1]),
        "warnings": warnings,
    }
    return VolatilityResult(feature_frame, np.asarray(standardized), diagnostics)
