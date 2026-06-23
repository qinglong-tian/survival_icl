"""Shared survival-curve utilities."""

from __future__ import annotations

import warnings

import numpy as np


def trapezoid(y, x, *, axis: int = -1):
    """Integrate with NumPy's trapezoid rule across supported NumPy versions."""
    rule = getattr(np, "trapezoid", None)
    if rule is None:
        rule = np.trapz
    return rule(y, x, axis=axis)


def kaplan_meier_survival_curve(t_obs: np.ndarray, delta: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Evaluate the Kaplan-Meier survival estimate on ``grid``."""
    from lifelines import KaplanMeierFitter

    kmf = KaplanMeierFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmf.fit(
            np.asarray(t_obs, dtype=np.float64),
            event_observed=np.asarray(delta, dtype=bool),
        )
    values = np.asarray(kmf.survival_function_at_times(grid).values, dtype=np.float64)
    return np.minimum.accumulate(np.clip(values, 0.0, 1.0))


def condition_survival_curves(
    curves: np.ndarray,
    grid: np.ndarray,
    censor_times: np.ndarray,
    *,
    condition_on_censoring: bool,
) -> np.ndarray:
    """Optionally convert ``S(t | X)`` curves to ``S(t | T > censor_time, X)``."""
    curves = np.asarray(curves, dtype=np.float64)
    if not condition_on_censoring:
        return curves.copy()

    grid = np.asarray(grid, dtype=np.float64)
    censor_times = np.asarray(censor_times, dtype=np.float64)
    conditioned = curves.copy()
    for row_idx, censor_time in enumerate(censor_times):
        s_c = float(np.interp(censor_time, grid, curves[row_idx], left=1.0, right=curves[row_idx, -1]))
        after = grid > censor_time
        conditioned[row_idx, ~after] = 1.0
        conditioned[row_idx, after] = np.clip(conditioned[row_idx, after] / max(s_c, 1e-8), 0.0, 1.0)
        conditioned[row_idx] = np.minimum.accumulate(conditioned[row_idx])
    return conditioned


def survival_median(
    grid: np.ndarray,
    curves: np.ndarray,
    lower_bounds: np.ndarray,
) -> np.ndarray:
    """Return median event times from survival curves represented on ``grid``."""
    grid = np.asarray(grid, dtype=np.float64)
    curves = np.asarray(curves, dtype=np.float64)
    lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
    medians = np.empty(curves.shape[0], dtype=np.float32)
    for row_idx, survival in enumerate(curves):
        lower = float(lower_bounds[row_idx])
        eligible = (grid > lower) & (survival <= 0.5)
        if not eligible.any():
            medians[row_idx] = np.float32(max(lower, grid[-1]))
            continue
        hit = int(np.argmax(eligible))
        prev = max(hit - 1, 0)
        s0 = float(survival[prev])
        s1 = float(survival[hit])
        t0 = float(grid[prev])
        t1 = float(grid[hit])
        if s0 == s1:
            medians[row_idx] = np.float32(t1)
        else:
            weight = np.clip((0.5 - s0) / (s1 - s0), 0.0, 1.0)
            medians[row_idx] = np.float32(t0 + weight * (t1 - t0))
    return np.maximum(medians, lower_bounds).astype(np.float32)


def survival_restricted_mean(
    grid: np.ndarray,
    curves: np.ndarray,
    lower_bounds: np.ndarray,
) -> np.ndarray:
    """Return restricted mean event times from survival curves on ``grid``."""
    grid = np.asarray(grid, dtype=np.float64)
    curves = np.asarray(curves, dtype=np.float64)
    lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
    means = np.empty(curves.shape[0], dtype=np.float32)
    for row_idx, survival in enumerate(curves):
        lower = float(lower_bounds[row_idx])
        post_grid = grid[grid > lower]
        if post_grid.size == 0:
            means[row_idx] = np.float32(lower)
            continue
        s_lower = float(np.interp(lower, grid, survival, left=1.0, right=survival[-1]))
        eval_grid = np.concatenate([[lower], post_grid])
        eval_survival = np.concatenate([[s_lower], survival[grid > lower]])
        restricted_residual = trapezoid(eval_survival, eval_grid)
        means[row_idx] = np.float32(lower + restricted_residual)
    return np.maximum(means, lower_bounds).astype(np.float32)


def sample_survival_times(
    grid: np.ndarray,
    curves: np.ndarray,
    lower_bounds: np.ndarray,
    rng: np.random.Generator,
    *,
    n_samples: int = 1,
) -> np.ndarray:
    """Sample event times from survival curves represented on ``grid``."""
    grid = np.asarray(grid, dtype=np.float64)
    curves = np.asarray(curves, dtype=np.float64)
    lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
    samples = np.empty((curves.shape[0], n_samples), dtype=np.float32)
    for sample_idx in range(n_samples):
        for row_idx, survival in enumerate(curves):
            lower = float(lower_bounds[row_idx])
            u = float(rng.uniform())
            event_cdf = 1.0 - survival
            eligible = (grid > lower) & (event_cdf >= u)
            if not eligible.any():
                samples[row_idx, sample_idx] = np.float32(max(lower, grid[-1]))
                continue
            hit = int(np.argmax(eligible))
            prev = max(hit - 1, 0)
            f0 = float(event_cdf[prev])
            f1 = float(event_cdf[hit])
            t0 = float(grid[prev])
            t1 = float(grid[hit])
            if f0 == f1:
                samples[row_idx, sample_idx] = np.float32(t1)
            else:
                weight = np.clip((u - f0) / (f1 - f0), 0.0, 1.0)
                samples[row_idx, sample_idx] = np.float32(t0 + weight * (t1 - t0))
    return np.maximum(samples, lower_bounds[:, None]).astype(np.float32)
