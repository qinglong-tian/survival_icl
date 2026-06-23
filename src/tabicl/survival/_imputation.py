"""Imputation utilities for censored survival observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from tabicl.survival._sklearn import TabICLSurvivalEstimator, _as_1d_float


def _trapezoid(y, x, *, axis: int = -1):
    rule = getattr(np, "trapezoid", None)
    if rule is None:
        rule = np.trapz
    return rule(y, x, axis=axis)


@dataclass
class CensoredImputationResult:
    """Imputed event times for censored rows in one survival dataset."""

    censored_indices: np.ndarray
    hard_times: np.ndarray
    soft_times: np.ndarray
    completed_hard_times: np.ndarray
    completed_soft_times: np.ndarray
    time_grid: np.ndarray
    conditional_survival: np.ndarray
    hard_method: str
    condition_on_censoring: bool

    @property
    def survival_curves(self) -> np.ndarray:
        """Survival curves used for imputation."""
        return self.conditional_survival


def _take_rows(X, indices: np.ndarray):
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    return np.asarray(X)[indices]


def _survival_median(
    grid: np.ndarray,
    curves: np.ndarray,
    lower_bounds: np.ndarray,
) -> np.ndarray:
    medians = np.empty(curves.shape[0], dtype=np.float32)
    for row_idx, survival in enumerate(curves):
        lower = lower_bounds[row_idx]
        after = grid > lower
        eligible = after & (survival <= 0.5)
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
            weight = (0.5 - s0) / (s1 - s0)
            medians[row_idx] = np.float32(t0 + np.clip(weight, 0.0, 1.0) * (t1 - t0))
    return np.maximum(medians, lower_bounds).astype(np.float32)


def _survival_restricted_mean(
    grid: np.ndarray,
    curves: np.ndarray,
    lower_bounds: np.ndarray,
) -> np.ndarray:
    means = np.empty(curves.shape[0], dtype=np.float32)
    for row_idx, survival in enumerate(curves):
        lower = float(lower_bounds[row_idx])
        post_grid = grid[grid > lower]
        if post_grid.size == 0:
            means[row_idx] = np.float32(lower)
            continue
        s_lower = float(np.interp(lower, grid, survival, left=1.0, right=survival[-1]))
        eval_grid = np.concatenate([[lower], post_grid])
        eval_survival = np.concatenate([[s_lower], survival[grid > lower].astype(np.float64)])
        restricted_residual = _trapezoid(eval_survival, eval_grid)
        means[row_idx] = np.float32(lower + restricted_residual)
    return np.maximum(means, lower_bounds).astype(np.float32)


def _sample_survival_times(
    grid: np.ndarray,
    curves: np.ndarray,
    lower_bounds: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    samples = np.empty(curves.shape[0], dtype=np.float32)
    for row_idx, survival in enumerate(curves):
        lower = float(lower_bounds[row_idx])
        u = float(rng.uniform())
        event_cdf = 1.0 - survival
        after = grid > lower
        eligible = after & (event_cdf >= u)
        if not eligible.any():
            samples[row_idx] = np.float32(max(lower, grid[-1]))
            continue
        hit = int(np.argmax(eligible))
        prev = max(hit - 1, 0)
        f0 = float(event_cdf[prev])
        f1 = float(event_cdf[hit])
        t0 = float(grid[prev])
        t1 = float(grid[hit])
        if f0 == f1:
            samples[row_idx] = np.float32(t1)
        else:
            weight = (u - f0) / (f1 - f0)
            samples[row_idx] = np.float32(t0 + np.clip(weight, 0.0, 1.0) * (t1 - t0))
    return np.maximum(samples, lower_bounds).astype(np.float32)


def impute_censored_survival_times(
    checkpoint_path: str | Path,
    X,
    t,
    delta,
    *,
    hard_method: Literal["median", "mean"] = "median",
    n_soft_samples: int = 1,
    random_state: int | np.random.Generator | None = None,
    device: str = "cpu",
    max_context_size: int | None = None,
    query_batch_size: int = 512,
    standardize_features: bool = True,
    times: Sequence[float] | None = None,
    condition_on_censoring: bool = True,
) -> CensoredImputationResult:
    """Impute event times for censored units with a pretrained survival model.

    The full survival dataset is used as the in-context support set, including
    the censored rows being imputed.  This is a transductive imputation: each
    censored unit's own ``(t, delta=0)`` observation is present in the prompt
    while its event time is queried.

    When ``condition_on_censoring=True`` (default), hard and soft imputations
    use ``S(t | T > censor_time, X)`` and are constrained to be at least the
    observed censoring time.  When ``False``, they use the unconditional
    ``S(t | X)`` predicted by the model; imputed event times may then be earlier
    than the observed censoring time.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to a modern TabICL survival checkpoint.
    X : array-like or DataFrame, shape (n_samples, n_features)
        Features for the whole survival dataset.
    t : array-like, shape (n_samples,)
        Observed times, ``min(event_time, censoring_time)``.
    delta : array-like, shape (n_samples,)
        Event indicators, 1=event observed and 0=right-censored.
    hard_method : {"median", "mean"}, default="median"
        Hard imputation rule.
    n_soft_samples : int, default=1
        Number of stochastic imputations per censored row.
    random_state : int, Generator, or None, default=None
        Random state for soft imputation.
    device, max_context_size, query_batch_size, standardize_features :
        Passed to :class:`TabICLSurvivalEstimator`.
    times : sequence of float, optional
        Raw time grid for conditional survival evaluation.  If omitted, the
        checkpoint's context-specific raw grid is used.
    condition_on_censoring : bool, default=True
        Whether to condition query curves on each censored row's observed
        censoring time.

    Returns
    -------
    CensoredImputationResult
        Hard/soft imputed event times for censored rows plus completed full
        time vectors where observed event rows keep their original times.
    """
    t_arr = _as_1d_float("t", t)
    delta_arr = _as_1d_float("delta", delta, len(t_arr))
    if not np.isin(delta_arr, [0.0, 1.0]).all():
        raise ValueError("delta must contain only 0 or 1.")
    if not (t_arr > 0).all():
        raise ValueError("Observed survival times must be strictly positive.")
    if hard_method not in {"median", "mean"}:
        raise ValueError("hard_method must be either 'median' or 'mean'.")
    if n_soft_samples < 1:
        raise ValueError("n_soft_samples must be positive.")

    censored_indices = np.flatnonzero(delta_arr == 0.0)
    if censored_indices.size == 0:
        empty_soft = np.empty((0, n_soft_samples), dtype=np.float32)
        return CensoredImputationResult(
            censored_indices=censored_indices,
            hard_times=np.empty(0, dtype=np.float32),
            soft_times=empty_soft,
            completed_hard_times=t_arr.copy(),
            completed_soft_times=np.tile(t_arr[:, None], (1, n_soft_samples)),
            time_grid=np.empty(0, dtype=np.float64),
            conditional_survival=np.empty((0, 0), dtype=np.float32),
            hard_method=hard_method,
            condition_on_censoring=condition_on_censoring,
        )

    estimator = TabICLSurvivalEstimator(
        checkpoint_path,
        device=device,
        max_context_size=max_context_size,
        query_batch_size=query_batch_size,
        standardize_features=standardize_features,
    ).fit(X, t=t_arr, delta=delta_arr)

    X_censored = _take_rows(X, censored_indices)
    censor_times = t_arr[censored_indices]
    grid, curves = estimator.predict_survival_function(
        X_censored,
        times=times,
        conditional_time=censor_times if condition_on_censoring else None,
        return_times=True,
    )
    if condition_on_censoring:
        lower_bounds = censor_times
    else:
        lower_bounds = np.zeros_like(censor_times, dtype=np.float32)
    if hard_method == "median":
        hard_times = _survival_median(grid, curves, lower_bounds)
    else:
        hard_times = _survival_restricted_mean(grid, curves, lower_bounds)

    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    soft_times = np.column_stack([
        _sample_survival_times(grid, curves, lower_bounds, rng)
        for _ in range(n_soft_samples)
    ]).astype(np.float32)

    completed_hard = t_arr.copy()
    completed_hard[censored_indices] = hard_times
    completed_soft = np.tile(t_arr[:, None], (1, n_soft_samples)).astype(np.float32)
    completed_soft[censored_indices] = soft_times

    return CensoredImputationResult(
        censored_indices=censored_indices,
        hard_times=hard_times,
        soft_times=soft_times,
        completed_hard_times=completed_hard,
        completed_soft_times=completed_soft,
        time_grid=grid,
        conditional_survival=curves,
        hard_method=hard_method,
        condition_on_censoring=condition_on_censoring,
    )
