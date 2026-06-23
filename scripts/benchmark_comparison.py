"""Benchmark TabICL survival predictions against classical survival methods.

Usage:
    cd tabicl-main
    python scripts/benchmark_comparison.py
    python scripts/benchmark_comparison.py --n-trials 1 --n-samples 256 --skip-tabicl

The benchmark generates synthetic proportional-hazards data from one known
baseline family at a time. Each method is fitted only on the context
``(X, t_obs, delta)`` rows and evaluated on held-out query rows.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr as norm_cdf
from scipy.special import ndtri as norm_ppf


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _REPO_ROOT.parent
DEFAULT_CHECKPOINT_PATH = _PROJECT_ROOT / "checkpoints" / "step-5000.ckpt"
DEFAULT_SEED = 20260607
DEFAULT_N_SAMPLES = 768
DEFAULT_N_FEATURES = 5
DEFAULT_N_CONTEXT = 512
DEFAULT_N_TRIALS = 5
DEFAULT_GRID_SIZE = 100
DEFAULT_BETA = np.array([0.8, -0.6, 0.45, 0.0, 0.25], dtype=np.float64)
EPS = 1e-12


def _trapezoid(y, x, *, axis: int = -1):
    rule = getattr(np, "trapezoid", None)
    if rule is None:
        rule = np.trapz
    return rule(y, x, axis=axis)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for the synthetic benchmark."""

    n_samples: int = DEFAULT_N_SAMPLES
    n_features: int = DEFAULT_N_FEATURES
    n_context: int = DEFAULT_N_CONTEXT
    n_trials: int = DEFAULT_N_TRIALS
    seed: int = DEFAULT_SEED
    censor_scale: float = 18.0
    grid_size: int = DEFAULT_GRID_SIZE
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH
    device: str = "cpu"
    query_batch_size: int = 64
    skip_tabicl: bool = False

    @property
    def beta(self) -> np.ndarray:
        return DEFAULT_BETA[: self.n_features]


@dataclass(frozen=True)
class PHFamily:
    """Baseline family for proportional-hazards generation and fitting."""

    key: str
    label: str
    param_names: tuple[str, ...]
    true_params: dict[str, float]


FAMILIES: dict[str, PHFamily] = {
    "weibull": PHFamily("weibull", "Weibull", ("log_shape", "log_scale"), {"shape": 1.6, "scale": 10.0}),
    "gompertz": PHFamily("gompertz", "Gompertz", ("log_rate", "log_gamma"), {"rate": 1.0, "gamma": 0.15}),
    "loglogistic": PHFamily(
        "loglogistic",
        "LogLogistic",
        ("log_shape", "log_scale"),
        {"shape": 1.8, "scale": 1.0},
    ),
    "lognormal": PHFamily("lognormal", "LogNormal", ("mu", "log_sigma"), {"mu": 0.0, "sigma": 0.8}),
}
FAMILY_KEYS = tuple(FAMILIES)


@dataclass
class SyntheticData:
    """One generated train/query survival task."""

    family_key: str
    X: np.ndarray
    t_obs: np.ndarray
    delta: np.ndarray
    t_event: np.ndarray
    true_survival: Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class FittedModel:
    """Minimal prediction interface shared by all benchmark models."""

    name: str
    model_type: str
    specification: str
    risk: Callable[[np.ndarray], np.ndarray]
    survival: Callable[[np.ndarray, np.ndarray], np.ndarray]


def _clip_time(t: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(t, dtype=np.float64), EPS)


def _unpack_params(family_key: str, theta: np.ndarray | None = None, params: dict[str, float] | None = None) -> dict[str, float]:
    if params is not None:
        return dict(params)
    if theta is None:
        raise ValueError("Either theta or params must be provided.")
    if family_key == "weibull":
        return {"shape": float(np.exp(theta[0])), "scale": float(np.exp(theta[1]))}
    if family_key == "gompertz":
        return {"rate": float(np.exp(theta[0])), "gamma": float(np.exp(theta[1]))}
    if family_key == "loglogistic":
        return {"shape": float(np.exp(theta[0])), "scale": float(np.exp(theta[1]))}
    if family_key == "lognormal":
        return {"mu": float(theta[0]), "sigma": float(np.exp(theta[1]))}
    raise ValueError(family_key)


def cumulative_hazard0(t: np.ndarray, family_key: str, params: dict[str, float]) -> np.ndarray:
    """Return baseline cumulative hazard ``H0(t)``."""

    t = _clip_time(t)
    if family_key == "weibull":
        return (t / params["scale"]) ** params["shape"]
    if family_key == "gompertz":
        gamma = params["gamma"]
        return params["rate"] * np.expm1(np.minimum(gamma * t, 50.0)) / gamma
    if family_key == "loglogistic":
        return np.log1p((t / params["scale"]) ** params["shape"])
    if family_key == "lognormal":
        z = (params["mu"] - np.log(t)) / params["sigma"]
        return -np.log(np.clip(norm_cdf(z), EPS, 1.0))
    raise ValueError(family_key)


def log_hazard0(t: np.ndarray, family_key: str, params: dict[str, float]) -> np.ndarray:
    """Return baseline log-hazard ``log h0(t)``."""

    t = _clip_time(t)
    if family_key == "weibull":
        shape = params["shape"]
        scale = params["scale"]
        return np.log(shape) - np.log(scale) + (shape - 1.0) * (np.log(t) - np.log(scale))
    if family_key == "gompertz":
        return np.log(params["rate"]) + np.minimum(params["gamma"] * t, 50.0)
    if family_key == "loglogistic":
        shape = params["shape"]
        scale = params["scale"]
        z = (t / scale) ** shape
        return np.log(shape) - np.log(scale) + (shape - 1.0) * (np.log(t) - np.log(scale)) - np.log1p(z)
    if family_key == "lognormal":
        mu = params["mu"]
        sigma = params["sigma"]
        z = (np.log(t) - mu) / sigma
        log_pdf = -np.log(t) - np.log(sigma) - 0.5 * np.log(2.0 * np.pi) - 0.5 * z**2
        log_surv = -cumulative_hazard0(t, family_key, params)
        return log_pdf - log_surv
    raise ValueError(family_key)


def inverse_cumulative_hazard0(h: np.ndarray, family_key: str, params: dict[str, float]) -> np.ndarray:
    """Return ``H0^{-1}(h)`` for synthetic PH sampling."""

    h = np.maximum(np.asarray(h, dtype=np.float64), EPS)
    if family_key == "weibull":
        return params["scale"] * h ** (1.0 / params["shape"])
    if family_key == "gompertz":
        inner = 1.0 + params["gamma"] * h / params["rate"]
        return np.log(np.maximum(inner, 1.0 + EPS)) / params["gamma"]
    if family_key == "loglogistic":
        return params["scale"] * np.expm1(np.minimum(h, 50.0)) ** (1.0 / params["shape"])
    if family_key == "lognormal":
        s0 = np.exp(-np.minimum(h, 50.0))
        z = norm_ppf(np.clip(s0, 1e-10, 1.0 - 1e-10))
        return np.exp(params["mu"] - params["sigma"] * z)
    raise ValueError(family_key)


def baseline_survival(t: np.ndarray, family_key: str, params: dict[str, float]) -> np.ndarray:
    return np.exp(-cumulative_hazard0(t, family_key, params))


def generate_ph_data(family_key: str, config: BenchmarkConfig, seed: int) -> SyntheticData:
    """Generate one right-censored PH synthetic task."""

    rng = np.random.default_rng(seed)
    beta = config.beta
    params = FAMILIES[family_key].true_params
    X = rng.normal(size=(config.n_samples, config.n_features)).astype(np.float64)
    eta = X @ beta
    u = np.clip(rng.uniform(size=config.n_samples), 1e-10, 1.0 - 1e-10)
    event_hazard = -np.log(u) / np.exp(eta)
    t_event = inverse_cumulative_hazard0(event_hazard, family_key, params)
    censor_time = rng.exponential(scale=config.censor_scale, size=config.n_samples)
    t_obs = np.minimum(t_event, censor_time)
    delta = (t_event <= censor_time).astype(np.float64)

    def true_surv(times: np.ndarray, X_query: np.ndarray) -> np.ndarray:
        h0 = cumulative_hazard0(times, family_key, params)
        eta_query = np.asarray(X_query, dtype=np.float64) @ beta
        return np.exp(-np.exp(eta_query)[:, None] * h0[None, :])

    return SyntheticData(
        family_key=family_key,
        X=X.astype(np.float32),
        t_obs=t_obs.astype(np.float64),
        delta=delta.astype(np.float64),
        t_event=t_event.astype(np.float64),
        true_survival=true_surv,
    )


def _negative_log_likelihood(theta: np.ndarray, family_key: str, X: np.ndarray, t: np.ndarray, delta: np.ndarray) -> float:
    beta = theta[: X.shape[1]]
    params = _unpack_params(family_key, theta[X.shape[1] :])
    eta = np.clip(X @ beta, -20.0, 20.0)
    h0 = np.clip(cumulative_hazard0(t, family_key, params), EPS, 1e100)
    log_h0 = log_hazard0(t, family_key, params)
    ll = delta * (log_h0 + eta) - h0 * np.exp(eta)
    value = -float(np.sum(ll))
    if not np.isfinite(value):
        return 1e100
    return value + 1e-4 * float(np.sum(beta**2))


def fit_parametric_ph(
    family_key: str,
    X_train: np.ndarray,
    t_train: np.ndarray,
    delta_train: np.ndarray,
    data_family_key: str,
) -> FittedModel:
    """Fit one parametric PH family by censored maximum likelihood."""

    family = FAMILIES[family_key]
    beta0 = np.zeros(X_train.shape[1], dtype=np.float64)
    if family_key == "weibull":
        family0 = np.array([np.log(1.2), np.log(np.median(t_train))])
    elif family_key == "gompertz":
        family0 = np.array([np.log(1.0), np.log(0.05)])
    elif family_key == "loglogistic":
        family0 = np.array([np.log(1.2), np.log(np.median(t_train))])
    else:
        family0 = np.array([np.log(np.median(t_train)), np.log(1.0)])
    theta0 = np.concatenate([beta0, family0])
    result = minimize(
        _negative_log_likelihood,
        theta0,
        args=(family_key, np.asarray(X_train, dtype=np.float64), t_train, delta_train),
        method="L-BFGS-B",
        options={"maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(f"{family.label} PH fit failed: {result.message}")

    beta_hat = result.x[: X_train.shape[1]]
    params_hat = _unpack_params(family_key, result.x[X_train.shape[1] :])
    specification = "correct" if family_key == data_family_key else "misspecified"

    def risk(X_query: np.ndarray) -> np.ndarray:
        return np.asarray(X_query, dtype=np.float64) @ beta_hat

    def survival(X_query: np.ndarray, times: np.ndarray) -> np.ndarray:
        h0 = cumulative_hazard0(times, family_key, params_hat)
        eta = np.clip(risk(X_query), -20.0, 20.0)
        return np.exp(-np.exp(eta)[:, None] * h0[None, :])

    return FittedModel(
        name=f"Parametric PH ({family.label})",
        model_type="parametric",
        specification=specification,
        risk=risk,
        survival=survival,
    )


def fit_oracle_ph(data_family_key: str, config: BenchmarkConfig) -> FittedModel:
    params = FAMILIES[data_family_key].true_params
    beta = config.beta
    family_label = FAMILIES[data_family_key].label

    def risk(X_query: np.ndarray) -> np.ndarray:
        return np.asarray(X_query, dtype=np.float64) @ beta

    def survival(X_query: np.ndarray, times: np.ndarray) -> np.ndarray:
        h0 = cumulative_hazard0(times, data_family_key, params)
        eta = risk(X_query)
        return np.exp(-np.exp(eta)[:, None] * h0[None, :])

    return FittedModel(
        name=f"Oracle PH ({family_label})",
        model_type="oracle",
        specification="oracle",
        risk=risk,
        survival=survival,
    )


def fit_kaplan_meier(t_train: np.ndarray, delta_train: np.ndarray) -> FittedModel:
    from lifelines import KaplanMeierFitter

    kmf = KaplanMeierFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmf.fit(t_train, event_observed=delta_train.astype(bool))

    def risk(X_query: np.ndarray) -> np.ndarray:
        return np.zeros(len(X_query), dtype=np.float64)

    def survival(X_query: np.ndarray, times: np.ndarray) -> np.ndarray:
        values = np.asarray(kmf.survival_function_at_times(times).values, dtype=np.float64)
        values = np.where(times <= kmf.survival_function_.index[-1], values, 0.0)
        return np.tile(values, (len(X_query), 1))

    return FittedModel("Kaplan-Meier", "non_parametric", "non_parametric", risk, survival)


def fit_cox_ph(X_train: np.ndarray, t_train: np.ndarray, delta_train: np.ndarray) -> FittedModel:
    from lifelines import CoxPHFitter

    columns = [f"x{i}" for i in range(X_train.shape[1])]
    df = pd.DataFrame(X_train, columns=columns)
    df["duration"] = t_train
    df["event"] = delta_train.astype(bool)
    cph = CoxPHFitter(penalizer=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(df, duration_col="duration", event_col="event", show_progress=False)

    def _df(X_query: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(X_query, columns=columns)

    def risk(X_query: np.ndarray) -> np.ndarray:
        return np.log(np.asarray(cph.predict_partial_hazard(_df(X_query)).values, dtype=np.float64).ravel())

    def survival(X_query: np.ndarray, times: np.ndarray) -> np.ndarray:
        curves = cph.predict_survival_function(_df(X_query), times=times)
        return np.asarray(curves.values, dtype=np.float64).T

    return FittedModel("Cox PH", "semi_parametric", "semi_parametric", risk, survival)


def fit_tabicl(
    X_train: np.ndarray,
    t_train: np.ndarray,
    delta_train: np.ndarray,
    config: BenchmarkConfig,
) -> FittedModel | None:
    if config.skip_tabicl:
        return None
    if not config.checkpoint_path.is_file():
        print(f"[SKIP] TabICL checkpoint not found: {config.checkpoint_path}")
        return None
    from tabicl.survival import TabICLSurvivalEstimator

    estimator = TabICLSurvivalEstimator(
        checkpoint_path=str(config.checkpoint_path),
        device=config.device,
        max_context_size=config.n_context,
        query_batch_size=config.query_batch_size,
    )
    estimator.fit(X_train.astype(np.float32), t=t_train.astype(np.float32), delta=delta_train.astype(np.float32))

    def risk(X_query: np.ndarray) -> np.ndarray:
        median = estimator.predict(X_query.astype(np.float32))
        return -np.log(np.maximum(median, EPS))

    def survival(X_query: np.ndarray, times: np.ndarray) -> np.ndarray:
        return np.asarray(
            estimator.predict_survival_function(X_query.astype(np.float32), times=times.astype(np.float32)),
            dtype=np.float64,
        )

    return FittedModel("TabICL", "icl", "icl", risk, survival)


def censoring_survival_function(t_train: np.ndarray, delta_train: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    from lifelines import KaplanMeierFitter

    kmf = KaplanMeierFitter()
    censor_event = 1.0 - delta_train
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmf.fit(t_train, event_observed=censor_event.astype(bool))

    def g(times: np.ndarray) -> np.ndarray:
        values = np.asarray(kmf.survival_function_at_times(np.asarray(times, dtype=np.float64)).values, dtype=np.float64)
        return np.clip(values, 1e-6, 1.0)

    return g


def harrell_c_index(t_obs: np.ndarray, delta: np.ndarray, risk: np.ndarray) -> float:
    from lifelines.utils import concordance_index

    return float(concordance_index(t_obs, -risk, event_observed=delta.astype(bool)))


def ipcw_brier_score(
    t_obs: np.ndarray,
    delta: np.ndarray,
    survival_at_horizon: np.ndarray,
    horizon: float,
    censor_survival: Callable[[np.ndarray], np.ndarray],
) -> float:
    t_obs = np.asarray(t_obs, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    y = (t_obs > horizon).astype(np.float64)
    weights = np.zeros_like(t_obs, dtype=np.float64)
    event_by_horizon = (t_obs <= horizon) & (delta == 1)
    still_at_risk = t_obs > horizon
    weights[event_by_horizon] = 1.0 / censor_survival(t_obs[event_by_horizon])
    weights[still_at_risk] = 1.0 / float(censor_survival(np.array([horizon]))[0])
    return float(np.mean(weights * (y - survival_at_horizon) ** 2))


def integrated_brier_score(
    t_obs: np.ndarray,
    delta: np.ndarray,
    survival_curves: np.ndarray,
    grid: np.ndarray,
    censor_survival: Callable[[np.ndarray], np.ndarray],
) -> float:
    scores = [
        ipcw_brier_score(t_obs, delta, survival_curves[:, i], float(t), censor_survival)
        for i, t in enumerate(grid)
    ]
    return float(_trapezoid(scores, grid) / (grid[-1] - grid[0]))


def calibration_at_horizon(
    t_obs: np.ndarray,
    delta: np.ndarray,
    survival_at_horizon: np.ndarray,
    horizon: float,
    censor_survival: Callable[[np.ndarray], np.ndarray],
    n_bins: int = 5,
) -> tuple[float, pd.DataFrame]:
    predicted_event = np.clip(1.0 - survival_at_horizon, 0.0, 1.0)
    event_by_horizon = ((t_obs <= horizon) & (delta == 1)).astype(np.float64)
    weights = np.zeros_like(t_obs, dtype=np.float64)
    observed_or_at_risk = ((t_obs <= horizon) & (delta == 1)) | (t_obs > horizon)
    weights[(t_obs <= horizon) & (delta == 1)] = 1.0 / censor_survival(t_obs[(t_obs <= horizon) & (delta == 1)])
    weights[t_obs > horizon] = 1.0 / float(censor_survival(np.array([horizon]))[0])
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.digitize(predicted_event, bins, right=True)
    indices = np.clip(indices, 1, n_bins)
    rows = []
    total_weight = float(np.sum(weights[observed_or_at_risk]))
    error = 0.0
    for bin_idx in range(1, n_bins + 1):
        mask = indices == bin_idx
        bin_weight = float(np.sum(weights[mask]))
        if bin_weight <= 0.0:
            continue
        pred_mean = float(np.average(predicted_event[mask], weights=weights[mask]))
        obs_mean = float(np.average(event_by_horizon[mask], weights=weights[mask]))
        frac = bin_weight / total_weight if total_weight > 0 else 0.0
        error += frac * abs(pred_mean - obs_mean)
        rows.append(
            {
                "bin": bin_idx,
                "bin_left": bins[bin_idx - 1],
                "bin_right": bins[bin_idx],
                "predicted_event": pred_mean,
                "observed_event": obs_mean,
                "weight": bin_weight,
            },
        )
    return float(error), pd.DataFrame(rows)


def oracle_curve_error(pred_survival: np.ndarray, true_survival: np.ndarray, grid: np.ndarray) -> tuple[float, float]:
    ise = _trapezoid((pred_survival - true_survival) ** 2, grid, axis=1) / (grid[-1] - grid[0])
    iae = _trapezoid(np.abs(pred_survival - true_survival), grid, axis=1) / (grid[-1] - grid[0])
    return float(np.mean(ise)), float(np.mean(iae))


def evaluation_grid(t_train: np.ndarray, delta_train: np.ndarray, grid_size: int) -> tuple[np.ndarray, dict[str, float]]:
    events = t_train[delta_train == 1]
    source = events if len(events) >= 5 else t_train
    horizons = {
        "q25": float(np.quantile(source, 0.25)),
        "q50": float(np.quantile(source, 0.50)),
        "q75": float(np.quantile(source, 0.75)),
    }
    high = max(float(np.quantile(source, 0.90)), horizons["q75"] * 1.25, horizons["q50"] + EPS)
    grid = np.linspace(max(EPS, horizons["q25"] * 0.5), high, grid_size)
    return grid, horizons


def fit_models_for_task(
    data_family_key: str,
    X_context: np.ndarray,
    t_context: np.ndarray,
    delta_context: np.ndarray,
    config: BenchmarkConfig,
) -> list[FittedModel]:
    models = [
        fit_oracle_ph(data_family_key, config),
        fit_kaplan_meier(t_context, delta_context),
        fit_cox_ph(X_context, t_context, delta_context),
    ]
    tabicl = fit_tabicl(X_context, t_context, delta_context, config)
    if tabicl is not None:
        models.append(tabicl)
    for family_key in FAMILY_KEYS:
        models.append(fit_parametric_ph(family_key, X_context, t_context, delta_context, data_family_key))
    return models


def evaluate_model(
    model: FittedModel,
    data_family_key: str,
    trial: int,
    X_query: np.ndarray,
    t_query: np.ndarray,
    delta_query: np.ndarray,
    true_survival_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    grid: np.ndarray,
    horizons: dict[str, float],
    censor_survival: Callable[[np.ndarray], np.ndarray],
    event_rate: float,
) -> tuple[list[dict], pd.DataFrame]:
    risk = model.risk(X_query)
    curves = np.clip(model.survival(X_query, grid), 0.0, 1.0)
    true_curves = true_survival_fn(grid, X_query)
    c_index = harrell_c_index(t_query, delta_query, risk)
    ibs = integrated_brier_score(t_query, delta_query, curves, grid, censor_survival)
    oracle_ise, oracle_iae = oracle_curve_error(curves, true_curves, grid)
    rows = []
    calibration_frames = []
    for horizon_label, horizon in horizons.items():
        survival_at_horizon = np.array([np.interp(horizon, grid, curve) for curve in curves])
        brier = ipcw_brier_score(t_query, delta_query, survival_at_horizon, horizon, censor_survival)
        calibration_error, calibration_df = calibration_at_horizon(
            t_query,
            delta_query,
            survival_at_horizon,
            horizon,
            censor_survival,
        )
        if not calibration_df.empty:
            calibration_df = calibration_df.assign(
                data_family=FAMILIES[data_family_key].label,
                model=model.name,
                trial=trial,
                horizon_label=horizon_label,
                horizon=horizon,
            )
            calibration_frames.append(calibration_df)
        rows.append(
            {
                "data_family": FAMILIES[data_family_key].label,
                "model": model.name,
                "model_type": model.model_type,
                "specification": model.specification,
                "trial": trial,
                "horizon": horizon_label,
                "horizon_time": horizon,
                "c_index": c_index,
                "brier": brier,
                "ibs": ibs,
                "calibration_error": calibration_error,
                "oracle_ise": oracle_ise,
                "oracle_iae": oracle_iae,
                "event_rate": event_rate,
            },
        )
    calibration = pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame()
    return rows, calibration


def run_benchmark(config: BenchmarkConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the complete benchmark and return result, calibration, and example data."""

    if config.n_context >= config.n_samples:
        raise ValueError("n_context must be smaller than n_samples.")
    rows: list[dict] = []
    calibration_rows: list[pd.DataFrame] = []
    examples: dict[str, dict] = {}

    for family_key in FAMILY_KEYS:
        family = FAMILIES[family_key]
        for trial in range(config.n_trials):
            print(f"[{family.label}] trial {trial + 1}/{config.n_trials}")
            seed = config.seed + 10_000 * FAMILY_KEYS.index(family_key) + trial
            data = generate_ph_data(family_key, config, seed)
            X_context = data.X[: config.n_context]
            X_query = data.X[config.n_context :]
            t_context = data.t_obs[: config.n_context]
            delta_context = data.delta[: config.n_context]
            t_query = data.t_obs[config.n_context :]
            delta_query = data.delta[config.n_context :]
            event_rate = float(delta_context.mean())
            grid, horizons = evaluation_grid(t_context, delta_context, config.grid_size)
            censor_survival = censoring_survival_function(t_context, delta_context)

            try:
                models = fit_models_for_task(family_key, X_context, t_context, delta_context, config)
            except Exception as exc:
                raise RuntimeError(f"Failed to fit benchmark models for {family.label} trial {trial}: {exc}") from exc

            example_curves = {}
            example_index = 0
            for model in models:
                try:
                    model_rows, calibration = evaluate_model(
                        model,
                        family_key,
                        trial,
                        X_query,
                        t_query,
                        delta_query,
                        data.true_survival,
                        grid,
                        horizons,
                        censor_survival,
                        event_rate,
                    )
                except Exception as exc:
                    raise RuntimeError(f"Failed to evaluate {model.name} on {family.label} trial {trial}: {exc}") from exc
                rows.extend(model_rows)
                if not calibration.empty:
                    calibration_rows.append(calibration)
                if trial == 0:
                    example_curves[model.name] = np.clip(model.survival(X_query[[example_index]], grid)[0], 0.0, 1.0)

            if trial == 0:
                examples[family.label] = {
                    "grid": grid,
                    "true": data.true_survival(grid, X_query[[example_index]])[0],
                    "curves": example_curves,
                }

    result = pd.DataFrame(rows)
    calibration = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()
    return result, calibration, examples


def summarize_results(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create compact summary tables for notebook and CLI reporting."""

    by_model = results.groupby(["data_family", "model", "model_type", "specification"])
    metric_summary = by_model.agg(
        c_index_mean=("c_index", "mean"),
        c_index_std=("c_index", "std"),
        ibs_mean=("ibs", "mean"),
        ibs_std=("ibs", "std"),
        calibration_error_mean=("calibration_error", "mean"),
        oracle_ise_mean=("oracle_ise", "mean"),
        oracle_iae_mean=("oracle_iae", "mean"),
    ).round(4)
    brier_summary = results.groupby(["data_family", "model", "horizon"])["brier"].agg(["mean", "std"]).round(4)
    calibration_summary = by_model.agg(
        ibs_mean=("ibs", "mean"),
        brier_mean=("brier", "mean"),
        calibration_error_mean=("calibration_error", "mean"),
        oracle_ise_mean=("oracle_ise", "mean"),
        oracle_iae_mean=("oracle_iae", "mean"),
        c_index_mean=("c_index", "mean"),
    ).round(4)
    return {"metrics": metric_summary, "brier": brier_summary, "calibration": calibration_summary}


def parametric_calibration_gaps(results: pd.DataFrame) -> pd.DataFrame:
    """Compare correct and misspecified parametric PH calibration per generated family."""

    parametric = results[results["model_type"] == "parametric"]
    summary = parametric.groupby(["data_family", "specification"]).agg(
        ibs=("ibs", "mean"),
        brier=("brier", "mean"),
        calibration_error=("calibration_error", "mean"),
        oracle_ise=("oracle_ise", "mean"),
        oracle_iae=("oracle_iae", "mean"),
    )
    wide = summary.unstack("specification")
    rows = []
    for family in wide.index:
        row = {"data_family": family}
        for metric in ["ibs", "brier", "calibration_error", "oracle_ise", "oracle_iae"]:
            correct = float(wide.loc[family, (metric, "correct")])
            misspecified = float(wide.loc[family, (metric, "misspecified")])
            row[f"{metric}_correct"] = correct
            row[f"{metric}_misspecified_mean"] = misspecified
            row[f"{metric}_gap"] = misspecified - correct
        rows.append(row)
    return pd.DataFrame(rows).set_index("data_family").round(4)


def design_matrix() -> pd.DataFrame:
    """Return correct/misspecified labels for the parametric benchmark grid."""

    labels = [FAMILIES[key].label for key in FAMILY_KEYS]
    return pd.DataFrame(
        [[("correct" if row == col else "misspecified") for col in labels] for row in labels],
        index=pd.Index(labels, name="Generated family"),
        columns=pd.Index(labels, name="Parametric PH model"),
    )


def _load_pyplot():
    if "MPLCONFIGDIR" not in os.environ:
        cache_dir = Path(tempfile.gettempdir()) / "tabicl_matplotlib_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)
    import matplotlib.pyplot as plt

    return plt


def plot_design_heatmap():
    plt = _load_pyplot()

    labels = [FAMILIES[key].label for key in FAMILY_KEYS]
    values = np.eye(len(labels))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.imshow(values, cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Fitted parametric PH family")
    ax.set_ylabel("Generated PH family")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, "correct" if i == j else "misspecified", ha="center", va="center", fontsize=9)
    ax.set_title("Correct vs misspecified parametric models")
    fig.tight_layout()
    return fig


def plot_survival_examples(examples: dict):
    plt = _load_pyplot()

    fig, axes = plt.subplots(1, len(examples), figsize=(4.6 * len(examples), 3.8), sharey=True)
    if len(examples) == 1:
        axes = [axes]
    for ax, (family, example) in zip(axes, examples.items()):
        grid = example["grid"]
        ax.plot(grid, example["true"], color="black", linewidth=2.2, label="True")
        preferred = [
            "TabICL",
            "Cox PH",
            "Kaplan-Meier",
            f"Parametric PH ({family})",
        ]
        wrong = next(
            (name for name in example["curves"] if name.startswith("Parametric PH") and f"({family})" not in name),
            None,
        )
        if wrong is not None:
            preferred.append(wrong)
        for name in preferred:
            if name in example["curves"]:
                ax.plot(grid, example["curves"][name], linewidth=1.4, label=name)
        ax.set_title(f"{family} data")
        ax.set_xlabel("Time")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Survival probability")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    return fig


def plot_metric_bars(results: pd.DataFrame, metric: str):
    plt = _load_pyplot()

    summary = results.groupby(["data_family", "model"])[metric].mean().reset_index()
    families = list(summary["data_family"].drop_duplicates())
    models = list(summary["model"].drop_duplicates())
    fig, axes = plt.subplots(1, len(families), figsize=(4.6 * len(families), 4), sharey=True)
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        sub = summary[summary["data_family"] == family].set_index("model").reindex(models)
        ax.bar(range(len(models)), sub[metric].values)
        ax.set_title(family)
        ax.set_xticks(range(len(models)), models, rotation=55, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(metric.replace("_", " ").title())
    fig.tight_layout()
    return fig


def plot_brier_horizons(results: pd.DataFrame):
    plt = _load_pyplot()

    summary = results.groupby(["horizon", "model"])["brier"].mean().reset_index()
    horizons = ["q25", "q50", "q75"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, sub in summary.groupby("model"):
        sub = sub.set_index("horizon").reindex(horizons)
        ax.plot(horizons, sub["brier"], marker="o", label=model)
    ax.set_xlabel("Evaluation horizon")
    ax.set_ylabel("IPCW Brier score")
    ax.set_title("Survival-function error by horizon")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    return fig


def plot_calibration(calibration: pd.DataFrame, horizon_label: str = "q50", models: list[str] | None = None):
    plt = _load_pyplot()

    sub = calibration[calibration["horizon_label"] == horizon_label]
    if models is None:
        models = list(sub["model"].drop_duplicates())[:6]
    else:
        models = [model for model in models if model in set(sub["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.2), sharex=True, sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        model_df = sub[sub["model"] == model]
        grouped = model_df.groupby("bin").agg(
            predicted_event=("predicted_event", "mean"),
            observed_event=("observed_event", "mean"),
        )
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        ax.scatter(grouped["predicted_event"], grouped["observed_event"], s=30)
        ax.set_title(model, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("IPCW observed event frequency")
    for ax in axes:
        ax.set_xlabel("Predicted event probability")
    fig.suptitle(f"Calibration at {horizon_label}")
    fig.tight_layout()
    return fig


def plot_calibration_by_family(calibration: pd.DataFrame, horizon_label: str = "q50"):
    plt = _load_pyplot()

    sub = calibration[calibration["horizon_label"] == horizon_label]
    families = list(sub["data_family"].drop_duplicates())
    fig, axes = plt.subplots(1, len(families), figsize=(4.2 * len(families), 3.8), sharex=True, sharey=True)
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        family_df = sub[sub["data_family"] == family]
        candidate_models = [
            f"Oracle PH ({family})",
            f"Parametric PH ({family})",
            "Cox PH",
            "TabICL",
        ]
        wrong = next(
            (
                model
                for model in family_df["model"].drop_duplicates()
                if model.startswith("Parametric PH") and f"({family})" not in model
            ),
            None,
        )
        if wrong is not None:
            candidate_models.append(wrong)
        for model in candidate_models:
            model_df = family_df[family_df["model"] == model]
            if model_df.empty:
                continue
            grouped = model_df.groupby("bin").agg(
                predicted_event=("predicted_event", "mean"),
                observed_event=("observed_event", "mean"),
            )
            ax.plot(grouped["predicted_event"], grouped["observed_event"], marker="o", linewidth=1.4, label=model)
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{family} data")
        ax.set_xlabel("Predicted event probability")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("IPCW observed event frequency")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle(f"Calibration curves at {horizon_label}")
    fig.tight_layout()
    return fig


def plot_calibration_error_bars(results: pd.DataFrame):
    return plot_metric_bars(results, "calibration_error")


def plot_predicted_observed_event_bars(calibration: pd.DataFrame, horizon_label: str = "q50"):
    plt = _load_pyplot()

    sub = calibration[calibration["horizon_label"] == horizon_label]
    summary = sub.groupby(["data_family", "model"]).agg(
        predicted_event=("predicted_event", "mean"),
        observed_event=("observed_event", "mean"),
    ).reset_index()
    families = list(summary["data_family"].drop_duplicates())
    fig, axes = plt.subplots(1, len(families), figsize=(4.8 * len(families), 4), sharey=True)
    if len(families) == 1:
        axes = [axes]
    for ax, family in zip(axes, families):
        family_df = summary[summary["data_family"] == family]
        models = list(family_df["model"])
        x = np.arange(len(models))
        width = 0.38
        ax.bar(x - width / 2, family_df["predicted_event"], width, label="Predicted")
        ax.bar(x + width / 2, family_df["observed_event"], width, label="Observed")
        ax.set_title(family)
        ax.set_xticks(x, models, rotation=55, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Event probability")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle(f"Predicted vs IPCW-observed event probability at {horizon_label}")
    fig.tight_layout()
    return fig


def save_figures(results: pd.DataFrame, calibration: pd.DataFrame, examples: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "design_heatmap.png": plot_design_heatmap(),
        "survival_examples.png": plot_survival_examples(examples),
        "c_index.png": plot_metric_bars(results, "c_index"),
        "ibs.png": plot_metric_bars(results, "ibs"),
        "calibration_error.png": plot_calibration_error_bars(results),
        "oracle_ise.png": plot_metric_bars(results, "oracle_ise"),
        "brier_horizons.png": plot_brier_horizons(results),
    }
    if not calibration.empty:
        figures["calibration_q50.png"] = plot_calibration(calibration, "q50")
        figures["calibration_by_family_q50.png"] = plot_calibration_by_family(calibration, "q50")
        figures["predicted_observed_q50.png"] = plot_predicted_observed_event_bars(calibration, "q50")
    for filename, fig in figures.items():
        fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--n-context", type=int, default=DEFAULT_N_CONTEXT)
    parser.add_argument("--n-features", type=int, default=DEFAULT_N_FEATURES)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--skip-tabicl", action="store_true")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--calibration-csv", type=Path)
    parser.add_argument("--plot-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_context=args.n_context,
        n_trials=args.n_trials,
        seed=args.seed,
        grid_size=args.grid_size,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        query_batch_size=args.query_batch_size,
        skip_tabicl=args.skip_tabicl,
    )
    results, calibration, examples = run_benchmark(config)
    summaries = summarize_results(results)
    print("\n=== Survival Benchmark: TabICL vs Classical Baselines ===")
    print(f"Trials per generated family: {config.n_trials}")
    print("\n--- Correct vs misspecified parametric PH grid ---")
    print(design_matrix().to_string())
    print("\n--- Metrics by generated family and model ---")
    print(summaries["metrics"].to_string())
    print("\n--- Calibration-focused metrics ---")
    print(summaries["calibration"].to_string())
    print("\n--- Parametric correct vs misspecified calibration gaps ---")
    print(parametric_calibration_gaps(results).to_string())
    print("\n--- IPCW Brier score by horizon ---")
    print(summaries["brier"].to_string())
    if args.output_csv is not None:
        results.to_csv(args.output_csv, index=False)
    if args.calibration_csv is not None:
        calibration.to_csv(args.calibration_csv, index=False)
    if args.plot_dir is not None:
        save_figures(results, calibration, examples, args.plot_dir)


if __name__ == "__main__":
    main()
