#!/usr/bin/env python
"""Compare imputation quality on real survival benchmarks.

Real right-censored datasets do not reveal the event times for naturally
censored rows.  This benchmark therefore creates a verifiable holdout task:
observed-event rows are randomly selected, artificially censored before their
known event time, and then imputed from the masked dataset.  Scores are
computed against the original observed event time of those masked rows.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from benchmark_comparison import (
        DEFAULT_CHECKPOINT_PATH,
        EPS,
        FAMILIES,
        _unpack_params,
        cumulative_hazard0,
        log_hazard0,
    )
except ImportError:
    from scripts.benchmark_comparison import (
        DEFAULT_CHECKPOINT_PATH,
        EPS,
        FAMILIES,
        _unpack_params,
        cumulative_hazard0,
        log_hazard0,
    )

from tabicl.survival._real_datasets import (
    DEFAULT_REAL_SURVIVAL_DATA_DIR,
    dataset_names as registered_dataset_names,
    load_real_survival_benchmark,
)


DEFAULT_OUTPUT_DIR = Path("survival_eval_results") / "real_imputation_quality_comparison"
QUALITY_METRICS = [
    "median_mae",
    "median_rmse",
    "sample_mean_mae",
    "sample_draw_mae",
    "sample_crps",
    "median_log_mae",
    "sample_mean_log_mae",
    "sample_draw_log_mae",
    "sample_crps_normalized",
    "median_relative_mae",
    "sample_mean_relative_mae",
    "median_bias",
    "early_median_fraction",
    "early_sample_fraction",
]


@dataclass(frozen=True)
class RealImputationQualityConfig:
    """Configuration for real-data masked-event imputation quality."""

    datasets: tuple[str, ...] = ()
    data_dir: Path = DEFAULT_REAL_SURVIVAL_DATA_DIR
    n_trials: int = 3
    seed: int = 20260623
    holdout_fraction: float = 0.25
    max_holdout_events: int = 64
    min_holdout_events: int = 5
    min_context_events: int = 10
    censor_fraction_low: float = 0.2
    censor_fraction_high: float = 0.8
    artificial_censoring: str = "empirical"
    grid_size: int = 256
    n_imputation_samples: int = 100
    parametric_fit_families: tuple[str, ...] = ("weibull", "gompertz", "lognormal", "loglogistic")
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH
    device: str = "cpu"
    query_batch_size: int = 64
    max_context_size: int | None = None
    skip_tabicl: bool = False
    include_unconditional: bool = True


@dataclass
class ImputationOutput:
    """Point and sample imputations for masked holdout rows."""

    method: str
    mode: str
    median: np.ndarray
    samples: np.ndarray


@dataclass
class ParametricPHEstimate:
    """Fitted parametric PH model state used for imputation."""

    beta: np.ndarray
    baseline_params: dict[str, float]


def _trapezoid(y, x, *, axis: int = -1):
    rule = getattr(np, "trapezoid", None)
    if rule is None:
        rule = np.trapz
    return rule(y, x, axis=axis)


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def encode_real_features(X) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return standardized numeric features for real benchmark methods."""
    frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    encoded = pd.get_dummies(frame, drop_first=True, dtype=np.float64)
    if encoded.shape[1] == 0:
        raise ValueError("Encoded feature matrix has no columns.")
    values = encoded.to_numpy(dtype=np.float64)
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return ((values - center) / scale).astype(np.float32), tuple(str(col) for col in encoded.columns)


def imputation_grid(t_obs: np.ndarray, *, grid_size: int) -> np.ndarray:
    """Build a raw-time grid without using held-out event times."""
    t_obs = np.asarray(t_obs, dtype=np.float64)
    upper = max(float(t_obs.max()) * 2.0, float(np.quantile(t_obs, 0.95)) * 3.0, EPS * 10)
    return np.linspace(EPS, upper, grid_size)


def km_survival_curve(t_obs: np.ndarray, delta: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Kaplan-Meier survival evaluated on ``grid``."""
    t_obs = np.asarray(t_obs, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    event_times = np.sort(np.unique(t_obs[delta == 1.0]))
    survival = 1.0
    values = np.ones_like(grid, dtype=np.float64)
    cursor = 0.0
    for event_time in event_times:
        values[(grid >= cursor) & (grid < event_time)] = survival
        at_risk = float(np.sum(t_obs >= event_time))
        events = float(np.sum((t_obs == event_time) & (delta == 1.0)))
        if at_risk > 0:
            survival *= max(0.0, 1.0 - events / at_risk)
        cursor = event_time
    values[grid >= cursor] = survival
    return np.minimum.accumulate(np.clip(values, 0.0, 1.0))


def condition_survival_curve(
    survival: np.ndarray,
    grid: np.ndarray,
    censor_time: float,
    *,
    condition_on_censoring: bool,
) -> np.ndarray:
    """Optionally convert ``S(t | X)`` to ``S(t | T > censor_time, X)``."""
    if not condition_on_censoring:
        return survival.astype(np.float64, copy=True)
    s_c = float(np.interp(censor_time, grid, survival, left=1.0, right=survival[-1]))
    conditioned = survival.astype(np.float64, copy=True) / max(s_c, 1e-8)
    conditioned[grid <= censor_time] = 1.0
    return np.minimum.accumulate(np.clip(conditioned, 0.0, 1.0))


def median_from_survival(grid: np.ndarray, survival: np.ndarray, lower_bound: float) -> float:
    """Return median event time from a survival curve on ``grid``."""
    eligible = (grid > lower_bound) & (survival <= 0.5)
    if not eligible.any():
        return float(max(lower_bound, grid[-1]))
    hit = int(np.argmax(eligible))
    prev = max(hit - 1, 0)
    s0 = float(survival[prev])
    s1 = float(survival[hit])
    t0 = float(grid[prev])
    t1 = float(grid[hit])
    if s0 == s1:
        return t1
    weight = np.clip((0.5 - s0) / (s1 - s0), 0.0, 1.0)
    return float(max(lower_bound, t0 + weight * (t1 - t0)))


def sample_from_survival(
    grid: np.ndarray,
    survival: np.ndarray,
    lower_bound: float,
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    """Sample event times from a survival curve represented on ``grid``."""
    event_cdf = 1.0 - survival
    draws = np.empty(n_samples, dtype=np.float32)
    for idx in range(n_samples):
        u = float(rng.uniform())
        eligible = (grid > lower_bound) & (event_cdf >= u)
        if not eligible.any():
            draws[idx] = np.float32(max(lower_bound, grid[-1]))
            continue
        hit = int(np.argmax(eligible))
        prev = max(hit - 1, 0)
        f0 = float(event_cdf[prev])
        f1 = float(event_cdf[hit])
        t0 = float(grid[prev])
        t1 = float(grid[hit])
        if f0 == f1:
            draws[idx] = np.float32(t1)
        else:
            weight = np.clip((u - f0) / (f1 - f0), 0.0, 1.0)
            draws[idx] = np.float32(max(lower_bound, t0 + weight * (t1 - t0)))
    return draws


def impute_from_survival_curves(
    *,
    method: str,
    mode: str,
    grid: np.ndarray,
    curves: np.ndarray,
    censor_times: np.ndarray,
    condition_on_censoring: bool,
    n_samples: int,
    rng: np.random.Generator,
) -> ImputationOutput:
    """Convert survival curves into median and sampled imputations."""
    medians = np.empty(curves.shape[0], dtype=np.float32)
    samples = np.empty((curves.shape[0], n_samples), dtype=np.float32)
    for row_idx, curve in enumerate(curves):
        conditioned = condition_survival_curve(
            curve,
            grid,
            float(censor_times[row_idx]),
            condition_on_censoring=condition_on_censoring,
        )
        lower = float(censor_times[row_idx]) if condition_on_censoring else 0.0
        medians[row_idx] = np.float32(median_from_survival(grid, conditioned, lower))
        samples[row_idx] = sample_from_survival(grid, conditioned, lower, rng, n_samples)
    return ImputationOutput(method=method, mode=mode, median=medians, samples=samples)


def km_imputation(
    t_obs: np.ndarray,
    delta: np.ndarray,
    censor_times: np.ndarray,
    grid: np.ndarray,
    *,
    condition_on_censoring: bool,
    n_samples: int,
    rng: np.random.Generator,
) -> ImputationOutput:
    """Marginal Kaplan-Meier imputation."""
    curve = km_survival_curve(t_obs, delta, grid)
    curves = np.tile(curve[None, :], (len(censor_times), 1))
    return impute_from_survival_curves(
        method="kaplan_meier",
        mode=_mode_name(condition_on_censoring),
        grid=grid,
        curves=curves,
        censor_times=censor_times,
        condition_on_censoring=condition_on_censoring,
        n_samples=n_samples,
        rng=rng,
    )


def ph_negative_log_likelihood(
    theta: np.ndarray,
    family_key: str,
    X: np.ndarray,
    t: np.ndarray,
    delta: np.ndarray,
) -> float:
    """Right-censored parametric PH negative log-likelihood."""
    beta = theta[: X.shape[1]]
    params = _unpack_params(family_key, theta[X.shape[1] :])
    eta = np.clip(np.asarray(X, dtype=np.float64) @ beta, -20.0, 20.0)
    h0 = np.clip(cumulative_hazard0(t, family_key, params), EPS, 1e100)
    log_h0 = log_hazard0(t, family_key, params)
    log_likelihood = delta * (log_h0 + eta) - h0 * np.exp(eta)
    value = -float(np.sum(log_likelihood))
    if not np.isfinite(value):
        return 1e100
    return value


def fit_parametric_ph_mle(
    family_key: str,
    X: np.ndarray,
    t: np.ndarray,
    delta: np.ndarray,
) -> ParametricPHEstimate:
    """Fit a parametric PH family by censored MLE."""
    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    beta0 = np.zeros(X.shape[1], dtype=np.float64)
    median_time = float(np.median(t))
    if family_key == "weibull":
        baseline0 = np.array([np.log(1.2), np.log(median_time)], dtype=np.float64)
    elif family_key == "gompertz":
        baseline0 = np.array([np.log(1.0), np.log(0.05)], dtype=np.float64)
    elif family_key == "loglogistic":
        baseline0 = np.array([np.log(1.2), np.log(median_time)], dtype=np.float64)
    elif family_key == "lognormal":
        baseline0 = np.array([np.log(median_time), np.log(1.0)], dtype=np.float64)
    else:
        raise ValueError(family_key)

    result = minimize(
        ph_negative_log_likelihood,
        np.concatenate([beta0, baseline0]),
        args=(family_key, X, t, delta),
        method="L-BFGS-B",
        options={"maxiter": 700},
    )
    if not result.success:
        raise RuntimeError(f"{FAMILIES[family_key].label} PH MLE failed: {result.message}")
    theta = np.asarray(result.x, dtype=np.float64)
    return ParametricPHEstimate(
        beta=theta[: X.shape[1]],
        baseline_params=_unpack_params(family_key, theta[X.shape[1] :]),
    )


def empirical_crps(samples: np.ndarray, truth: np.ndarray) -> float:
    """Empirical CRPS averaged over rows."""
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    first = np.mean(np.abs(samples - truth[:, None]), axis=1)
    sorted_samples = np.sort(samples, axis=1)
    m = samples.shape[1]
    coeff = (2 * np.arange(1, m + 1) - m - 1).astype(np.float64)
    pairwise = (2.0 / (m * m)) * np.sum(coeff[None, :] * sorted_samples, axis=1)
    return float(np.mean(first - 0.5 * pairwise))


def score_imputation(
    output: ImputationOutput,
    truth: np.ndarray,
    censor_times: np.ndarray,
    *,
    trial: int,
    event_rate: float,
    censored_count: int,
) -> dict:
    """Score one imputation output against known holdout event times."""
    median_error = output.median.astype(np.float64) - truth
    sample_mean = output.samples.astype(np.float64).mean(axis=1)
    sample_mean_error = sample_mean - truth
    draw_error = output.samples.astype(np.float64) - truth[:, None]
    return {
        "trial": trial,
        "method": output.method,
        "mode": output.mode,
        "event_rate_original": event_rate,
        "censored_count": censored_count,
        "median_mae": float(np.mean(np.abs(median_error))),
        "median_rmse": float(np.sqrt(np.mean(median_error**2))),
        "median_bias": float(np.mean(median_error)),
        "sample_mean_mae": float(np.mean(np.abs(sample_mean_error))),
        "sample_draw_mae": float(np.mean(np.abs(draw_error))),
        "sample_crps": empirical_crps(output.samples, truth),
        "early_median_fraction": float(np.mean(output.median < censor_times)),
        "early_sample_fraction": float(np.mean(output.samples < censor_times[:, None])),
    }


def choose_holdout_event_indices(
    event: np.ndarray,
    rng: np.random.Generator,
    *,
    holdout_fraction: float,
    max_holdout_events: int,
    min_holdout_events: int,
    min_context_events: int,
) -> np.ndarray:
    """Select observed-event rows to mask while leaving event context rows."""
    event_indices = np.flatnonzero(np.asarray(event, dtype=np.float64) == 1.0)
    max_allowed = event_indices.size - min_context_events
    if max_allowed < min_holdout_events:
        return np.empty(0, dtype=np.int64)
    target = int(np.ceil(event_indices.size * holdout_fraction))
    n_holdout = min(max_holdout_events, max(min_holdout_events, target), max_allowed)
    return np.sort(rng.choice(event_indices, size=n_holdout, replace=False)).astype(np.int64)


def artificial_censor_times(
    true_event_times: np.ndarray,
    natural_censor_times: np.ndarray,
    rng: np.random.Generator,
    *,
    strategy: str,
    fraction_low: float,
    fraction_high: float,
) -> np.ndarray:
    """Generate artificial censoring times strictly before known events."""
    if strategy not in {"empirical", "fraction"}:
        raise ValueError("strategy must be 'empirical' or 'fraction'.")
    if not (0.0 < fraction_low < fraction_high < 1.0):
        raise ValueError("censor fractions must satisfy 0 < low < high < 1.")

    natural = np.asarray(natural_censor_times, dtype=np.float64)
    natural = natural[np.isfinite(natural) & (natural > EPS)]
    event_times = np.asarray(true_event_times, dtype=np.float64)
    censor_times = np.empty_like(event_times, dtype=np.float64)
    for idx, event_time in enumerate(event_times):
        chosen = np.nan
        if strategy == "empirical" and natural.size:
            preferred = natural[
                (natural >= fraction_low * event_time)
                & (natural <= fraction_high * event_time)
                & (natural < event_time)
            ]
            if preferred.size:
                chosen = float(rng.choice(preferred))
            else:
                candidates = natural[(natural > EPS) & (natural < fraction_high * event_time)]
                if candidates.size:
                    chosen = float(rng.choice(candidates))
        if not np.isfinite(chosen):
            chosen = float(event_time * rng.uniform(fraction_low, fraction_high))
        censor_times[idx] = min(max(chosen, EPS), np.nextafter(event_time, 0.0))
    return censor_times.astype(np.float32)


def _mode_name(condition_on_censoring: bool) -> str:
    return "conditional" if condition_on_censoring else "unconditional"


def parametric_ph_real_imputation(
    X: np.ndarray,
    t_obs: np.ndarray,
    delta: np.ndarray,
    holdout_indices: np.ndarray,
    grid: np.ndarray,
    *,
    fit_family_key: str,
    condition_on_censoring: bool,
    n_samples: int,
    rng: np.random.Generator,
) -> ImputationOutput:
    """Parametric PH imputation for masked real-data holdout rows."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        estimate = fit_parametric_ph_mle(fit_family_key, X, t_obs, delta)
        eta = np.clip(np.asarray(X[holdout_indices], dtype=np.float64) @ estimate.beta, -20.0, 20.0)
        h0 = np.clip(cumulative_hazard0(grid, fit_family_key, estimate.baseline_params), EPS, 1e100)
        curves = np.exp(-np.exp(eta)[:, None] * h0[None, :])
    curves = np.nan_to_num(curves, nan=0.0, posinf=0.0, neginf=1.0)
    curves = np.minimum.accumulate(np.clip(curves, 0.0, 1.0), axis=1)
    return impute_from_survival_curves(
        method=f"{fit_family_key}_ph_mle",
        mode=_mode_name(condition_on_censoring),
        grid=grid,
        curves=curves,
        censor_times=t_obs[holdout_indices],
        condition_on_censoring=condition_on_censoring,
        n_samples=n_samples,
        rng=rng,
    )


def tabicl_real_imputations(
    checkpoint_path: Path,
    X: np.ndarray,
    t_obs: np.ndarray,
    delta: np.ndarray,
    holdout_indices: np.ndarray,
    grid: np.ndarray,
    *,
    condition_modes: tuple[bool, ...],
    n_samples: int,
    seed: int,
    device: str,
    query_batch_size: int,
    max_context_size: int | None,
) -> list[ImputationOutput]:
    """Pretrained TabICL imputation for masked real-data holdout rows."""
    from tabicl.survival import TabICLSurvivalEstimator

    estimator = TabICLSurvivalEstimator(
        checkpoint_path,
        device=device,
        max_context_size=max_context_size,
        query_batch_size=query_batch_size,
        standardize_features=False,
    ).fit(X, t=t_obs, delta=delta)
    eval_grid, curves = estimator.predict_survival_function(
        X[holdout_indices],
        times=grid.astype(np.float32),
        return_times=True,
    )
    outputs = []
    for mode_idx, condition_on_censoring in enumerate(condition_modes):
        rng = np.random.default_rng(seed + 9973 * mode_idx)
        outputs.append(
            impute_from_survival_curves(
                method="tabicl_pretrained",
                mode=_mode_name(condition_on_censoring),
                grid=eval_grid,
                curves=curves,
                censor_times=t_obs[holdout_indices],
                condition_on_censoring=condition_on_censoring,
                n_samples=n_samples,
                rng=rng,
            )
        )
    return outputs


def score_real_imputation(
    output: ImputationOutput,
    truth: np.ndarray,
    censor_times: np.ndarray,
    *,
    dataset: str,
    trial: int,
    event_rate_original: float,
    event_rate_masked: float,
    natural_censored_count: int,
    holdout_count: int,
    context_event_count: int,
) -> dict:
    """Score real-data masked-event imputation against known event times."""
    row = score_imputation(
        output,
        truth,
        censor_times,
        trial=trial,
        event_rate=event_rate_original,
        censored_count=natural_censored_count + holdout_count,
    )
    median = np.maximum(output.median.astype(np.float64), EPS)
    samples = np.maximum(output.samples.astype(np.float64), EPS)
    sample_mean = np.maximum(samples.mean(axis=1), EPS)
    truth_safe = np.maximum(np.asarray(truth, dtype=np.float64), EPS)
    sample_crps = float(row["sample_crps"])
    row.update(
        {
            "status": "ok",
            "failure_message": "",
            "dataset": dataset,
            "event_rate_masked": event_rate_masked,
            "natural_censored_count": natural_censored_count,
            "holdout_count": holdout_count,
            "context_event_count": context_event_count,
            "median_log_mae": float(np.mean(np.abs(np.log(median) - np.log(truth_safe)))),
            "sample_mean_log_mae": float(np.mean(np.abs(np.log(sample_mean) - np.log(truth_safe)))),
            "sample_draw_log_mae": float(np.mean(np.abs(np.log(samples) - np.log(truth_safe[:, None])))),
            "sample_crps_normalized": sample_crps / max(float(np.median(truth_safe)), EPS),
            "median_relative_mae": float(np.mean(np.abs(median - truth_safe) / truth_safe)),
            "sample_mean_relative_mae": float(np.mean(np.abs(sample_mean - truth_safe) / truth_safe)),
        }
    )
    return row


def failure_row(
    *,
    dataset: str,
    trial: int,
    method: str,
    mode: str,
    message: str,
    event_rate_original: float,
    event_rate_masked: float,
    natural_censored_count: int,
    holdout_count: int,
    context_event_count: int,
) -> dict:
    """Return a failed method row with metric columns present."""
    row = {
        "status": "failed",
        "failure_message": message,
        "dataset": dataset,
        "trial": trial,
        "method": method,
        "mode": mode,
        "event_rate_original": event_rate_original,
        "event_rate_masked": event_rate_masked,
        "censored_count": natural_censored_count + holdout_count,
        "natural_censored_count": natural_censored_count,
        "holdout_count": holdout_count,
        "context_event_count": context_event_count,
    }
    for metric in QUALITY_METRICS:
        row[metric] = np.nan
    return row


def run_real_imputation_quality_comparison(config: RealImputationQualityConfig) -> pd.DataFrame:
    """Run masked-event imputation quality comparison on real benchmarks."""
    available = set(registered_dataset_names())
    datasets = config.datasets or registered_dataset_names()
    unknown = sorted(set(datasets) - available)
    if unknown:
        raise ValueError(f"Unknown real survival datasets: {unknown}. Available: {sorted(available)}")
    if not (0.0 < config.holdout_fraction < 1.0):
        raise ValueError("holdout_fraction must be in (0, 1).")
    if config.max_holdout_events < 1 or config.min_holdout_events < 1 or config.min_context_events < 1:
        raise ValueError("holdout and context event counts must be positive.")
    if config.grid_size < 4:
        raise ValueError("grid_size must be at least 4.")
    if config.n_imputation_samples < 1:
        raise ValueError("n_imputation_samples must be positive.")
    for fit_family in config.parametric_fit_families:
        if fit_family not in FAMILIES:
            raise ValueError(f"Unsupported parametric fit family {fit_family!r}.")
    if not config.skip_tabicl and not Path(config.checkpoint_path).is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {config.checkpoint_path}. Pass --skip-tabicl "
            "to compare conventional baselines only."
        )

    rows: list[dict] = []
    condition_modes = (True, False) if config.include_unconditional else (True,)
    for dataset_idx, dataset in enumerate(datasets):
        data = load_real_survival_benchmark(dataset, data_dir=config.data_dir)
        X, feature_names = encode_real_features(data.X)
        time = np.asarray(data.time, dtype=np.float64)
        event = np.asarray(data.event, dtype=np.float64)
        natural_censored_count = int(np.sum(event == 0.0))
        event_rate_original = float(event.mean())
        dataset_row_start = len(rows)

        for trial in range(config.n_trials):
            rng = np.random.default_rng(config.seed + 100_003 * dataset_idx + trial)
            holdout_indices = choose_holdout_event_indices(
                event,
                rng,
                holdout_fraction=config.holdout_fraction,
                max_holdout_events=config.max_holdout_events,
                min_holdout_events=config.min_holdout_events,
                min_context_events=config.min_context_events,
            )
            if holdout_indices.size == 0:
                raise RuntimeError(
                    f"{dataset} has too few observed events for the requested holdout settings."
                )
            truth = time[holdout_indices]
            artificial_censor = artificial_censor_times(
                truth,
                time[event == 0.0],
                rng,
                strategy=config.artificial_censoring,
                fraction_low=config.censor_fraction_low,
                fraction_high=config.censor_fraction_high,
            )
            t_masked = time.copy()
            delta_masked = event.copy()
            t_masked[holdout_indices] = artificial_censor
            delta_masked[holdout_indices] = 0.0
            grid = imputation_grid(t_masked, grid_size=config.grid_size)
            event_rate_masked = float(delta_masked.mean())
            context_event_count = int(delta_masked.sum())

            for condition_on_censoring in condition_modes:
                mode = _mode_name(condition_on_censoring)
                km_output = km_imputation(
                    t_masked,
                    delta_masked,
                    t_masked[holdout_indices],
                    grid,
                    condition_on_censoring=condition_on_censoring,
                    n_samples=config.n_imputation_samples,
                    rng=rng,
                )
                rows.append(
                    score_real_imputation(
                        km_output,
                        truth,
                        t_masked[holdout_indices],
                        dataset=dataset,
                        trial=trial,
                        event_rate_original=event_rate_original,
                        event_rate_masked=event_rate_masked,
                        natural_censored_count=natural_censored_count,
                        holdout_count=int(holdout_indices.size),
                        context_event_count=context_event_count,
                    )
                )

                for fit_family in config.parametric_fit_families:
                    try:
                        output = parametric_ph_real_imputation(
                            X,
                            t_masked,
                            delta_masked,
                            holdout_indices,
                            grid,
                            fit_family_key=fit_family,
                            condition_on_censoring=condition_on_censoring,
                            n_samples=config.n_imputation_samples,
                            rng=rng,
                        )
                    except Exception as exc:
                        rows.append(
                            failure_row(
                                dataset=dataset,
                                trial=trial,
                                method=f"{fit_family}_ph_mle",
                                mode=mode,
                                message=str(exc),
                                event_rate_original=event_rate_original,
                                event_rate_masked=event_rate_masked,
                                natural_censored_count=natural_censored_count,
                                holdout_count=int(holdout_indices.size),
                                context_event_count=context_event_count,
                            )
                        )
                    else:
                        rows.append(
                            score_real_imputation(
                                output,
                                truth,
                                t_masked[holdout_indices],
                                dataset=dataset,
                                trial=trial,
                                event_rate_original=event_rate_original,
                                event_rate_masked=event_rate_masked,
                                natural_censored_count=natural_censored_count,
                                holdout_count=int(holdout_indices.size),
                                context_event_count=context_event_count,
                            )
                        )

            if not config.skip_tabicl:
                try:
                    tabicl_outputs = tabicl_real_imputations(
                        config.checkpoint_path,
                        X,
                        t_masked,
                        delta_masked,
                        holdout_indices,
                        grid,
                        condition_modes=condition_modes,
                        n_samples=config.n_imputation_samples,
                        seed=config.seed + 1_000_003 * dataset_idx + 10_000 * trial,
                        device=config.device,
                        query_batch_size=config.query_batch_size,
                        max_context_size=config.max_context_size,
                    )
                except Exception as exc:
                    for condition_on_censoring in condition_modes:
                        rows.append(
                            failure_row(
                                dataset=dataset,
                                trial=trial,
                                method="tabicl_pretrained",
                                mode=_mode_name(condition_on_censoring),
                                message=str(exc),
                                event_rate_original=event_rate_original,
                                event_rate_masked=event_rate_masked,
                                natural_censored_count=natural_censored_count,
                                holdout_count=int(holdout_indices.size),
                                context_event_count=context_event_count,
                            )
                        )
                else:
                    for output in tabicl_outputs:
                        rows.append(
                            score_real_imputation(
                                output,
                                truth,
                                t_masked[holdout_indices],
                                dataset=dataset,
                                trial=trial,
                                event_rate_original=event_rate_original,
                                event_rate_masked=event_rate_masked,
                                natural_censored_count=natural_censored_count,
                                holdout_count=int(holdout_indices.size),
                                context_event_count=context_event_count,
                            )
                        )

        for row in rows[dataset_row_start:]:
            row["feature_count_encoded"] = X.shape[1]
            row["feature_names_encoded"] = ",".join(feature_names)

    return pd.DataFrame(rows)


def _successful(results: pd.DataFrame) -> pd.DataFrame:
    if "status" not in results:
        return results
    return results[results["status"] == "ok"].copy()


def summarize_real_quality(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize real-data imputation metrics across datasets and trials."""
    ok = _successful(results)
    return ok.groupby(["method", "mode"])[QUALITY_METRICS].agg(["mean", "std"]).round(4)


def summarize_real_quality_by_dataset(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize real-data imputation metrics per dataset."""
    ok = _successful(results)
    metrics = [
        "median_log_mae",
        "sample_mean_log_mae",
        "sample_crps_normalized",
        "median_relative_mae",
        "early_sample_fraction",
    ]
    return ok.groupby(["dataset", "method", "mode"])[metrics].agg(["mean", "std"]).round(4)


def summarize_real_ranks(
    results: pd.DataFrame,
    *,
    metric: str = "sample_crps_normalized",
) -> pd.DataFrame:
    """Summarize within-dataset trial ranks for one scale-stable metric."""
    ok = _successful(results)
    ranked = ok.copy()
    ranked["rank"] = ranked.groupby(["dataset", "trial", "mode"])[metric].rank(method="average")
    return ranked.groupby(["method", "mode"])["rank"].agg(["mean", "std", "count"]).round(4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated real benchmark names. Empty means all registered datasets.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_REAL_SURVIVAL_DATA_DIR)
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--max-holdout-events", type=int, default=64)
    parser.add_argument("--min-holdout-events", type=int, default=5)
    parser.add_argument("--min-context-events", type=int, default=10)
    parser.add_argument("--censor-fraction-low", type=float, default=0.2)
    parser.add_argument("--censor-fraction-high", type=float, default=0.8)
    parser.add_argument("--artificial-censoring", choices=("empirical", "fraction"), default="empirical")
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--n-imputation-samples", type=int, default=100)
    parser.add_argument(
        "--parametric-fit-families",
        default="weibull,gompertz,lognormal,loglogistic",
        help="Comma-separated PH baseline families fitted by conventional MLE.",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--max-context-size", type=int, default=None)
    parser.add_argument("--skip-tabicl", action="store_true")
    parser.add_argument("--conditional-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    config = RealImputationQualityConfig(
        datasets=_csv_tuple(args.datasets),
        data_dir=args.data_dir,
        n_trials=args.n_trials,
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
        max_holdout_events=args.max_holdout_events,
        min_holdout_events=args.min_holdout_events,
        min_context_events=args.min_context_events,
        censor_fraction_low=args.censor_fraction_low,
        censor_fraction_high=args.censor_fraction_high,
        artificial_censoring=args.artificial_censoring,
        grid_size=args.grid_size,
        n_imputation_samples=args.n_imputation_samples,
        parametric_fit_families=_csv_tuple(args.parametric_fit_families),
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        query_batch_size=args.query_batch_size,
        max_context_size=args.max_context_size,
        skip_tabicl=args.skip_tabicl,
        include_unconditional=not args.conditional_only,
    )
    results = run_real_imputation_quality_comparison(config)
    summary = summarize_real_quality(results)
    by_dataset = summarize_real_quality_by_dataset(results)
    ranks = summarize_real_ranks(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "per_holdout_trial.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv")
    by_dataset.to_csv(args.output_dir / "per_dataset_summary.csv")
    ranks.to_csv(args.output_dir / "rank_summary.csv")
    config_json = asdict(config)
    config_json["data_dir"] = str(config.data_dir)
    config_json["checkpoint_path"] = str(config.checkpoint_path)
    (args.output_dir / "config.json").write_text(
        json.dumps(config_json, indent=2, sort_keys=True),
    )
    print(summary)
    print("\nMean within-dataset ranks by sample_crps_normalized:")
    print(ranks)


if __name__ == "__main__":
    main()
