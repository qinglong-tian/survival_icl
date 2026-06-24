"""Unit tests for shared survival-curve utilities."""

from __future__ import annotations

import numpy as np
import pytest

from tabicl.survival._curves import (
    condition_survival_curves,
    sample_survival_times,
    survival_median,
    survival_restricted_mean,
    trapezoid,
)


def test_trapezoid_backward_compat():
    x = np.linspace(0.0, 1.0, 5)
    y = x**2
    area = trapezoid(y, x)
    assert np.isfinite(area)
    assert 0.2 < area < 0.4


def test_survival_median_fallback_when_no_crossing():
    grid = np.linspace(0.0, 10.0, 50)
    curves = np.ones((2, 50), dtype=np.float64)
    lower_bounds = np.array([1.0, 2.0], dtype=np.float64)

    medians = survival_median(grid, curves, lower_bounds)
    assert medians.shape == (2,)
    assert np.all(medians >= lower_bounds)
    assert np.all(medians >= grid[-1])
    assert medians.dtype == np.float32


def test_survival_median_normal_crossing():
    grid = np.linspace(0.0, 20.0, 200)
    curves = np.exp(-0.1 * grid[None, :])
    lower_bounds = np.array([2.0, 4.0], dtype=np.float64)

    medians = survival_median(grid, curves, lower_bounds)
    assert medians.shape == (2,)
    assert np.all(medians > lower_bounds)
    assert np.all(medians < grid[-1])


def test_survival_median_ties_produce_hit():
    grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    curves = np.array([[1.0, 1.0, 0.3, 0.3, 0.1]], dtype=np.float64)
    lower_bounds = np.array([0.5], dtype=np.float64)
    medians = survival_median(grid, curves, lower_bounds)
    assert np.isfinite(medians[0])
    assert medians[0] >= 0.5


def test_survival_restricted_mean_basic():
    grid = np.linspace(0.0, 10.0, 100)
    curves = np.exp(-0.2 * grid[None, :])
    lower_bounds = np.array([1.0], dtype=np.float64)

    means = survival_restricted_mean(grid, curves, lower_bounds)
    assert means.shape == (1,)
    assert means[0] > lower_bounds[0]
    assert np.isfinite(means[0])
    assert means.dtype == np.float32


def test_survival_restricted_mean_empty_post_window():
    grid = np.linspace(0.0, 10.0, 100)
    curves = np.ones((2, 100), dtype=np.float64)
    lower_bounds = np.array([9.9, 10.0], dtype=np.float64)

    means = survival_restricted_mean(grid, curves, lower_bounds)
    assert means.shape == (2,)
    assert np.all(means >= lower_bounds)


def test_condition_survival_curves_passthrough_when_disabled():
    grid = np.linspace(0.0, 10.0, 20)
    curves = np.exp(-0.1 * grid[None, :])
    curves = np.tile(curves, (3, 1))
    censor_times = np.array([2.0, 3.0, 5.0])

    out = condition_survival_curves(curves, grid, censor_times, condition_on_censoring=False)
    assert out.shape == curves.shape
    assert np.allclose(out, curves)


def test_condition_survival_curves_preserves_monotonicity():
    grid = np.linspace(0.0, 20.0, 200)
    curves = np.exp(-0.05 * grid[None, :])
    curves = np.tile(curves, (2, 1))
    censor_times = np.array([4.0, 8.0])

    cond = condition_survival_curves(curves, grid, censor_times, condition_on_censoring=True)
    assert cond.shape == curves.shape
    assert np.all(cond[:, 0] == 1.0)
    assert np.all(np.diff(cond, axis=1) <= 0.0)
    for i, censor_time in enumerate(censor_times):
        assert np.allclose(cond[i, grid <= censor_time], 1.0)


def test_condition_survival_curves_tiny_s_c():
    grid = np.linspace(0.0, 10.0, 100)
    curves = np.array([[1.0] * 50 + [0.0] * 50], dtype=np.float64)
    censor_times = np.array([6.0], dtype=np.float64)

    cond = condition_survival_curves(curves, grid, censor_times, condition_on_censoring=True)
    assert cond.shape == curves.shape
    assert np.all(np.isfinite(cond))
    assert np.all(cond >= 0.0)
    assert np.all(cond <= 1.0)


def test_condition_survival_curves_zero_s_c():
    grid = np.linspace(0.0, 10.0, 100)
    curves = np.array([[1.0] * 40 + [0.0] * 60], dtype=np.float64)
    censor_times = np.array([9.0], dtype=np.float64)

    cond = condition_survival_curves(curves, grid, censor_times, condition_on_censoring=True)
    assert np.all(np.isfinite(cond))
    assert np.all(cond >= 0.0)
    assert np.all(cond <= 1.0)


def test_sample_survival_times_shape():
    rng = np.random.default_rng(42)
    grid = np.linspace(0.0, 10.0, 100)
    curves = np.exp(-0.2 * grid[None, :])
    curves = np.tile(curves, (4, 1))
    lower_bounds = np.array([0.5, 1.0, 1.5, 2.0])

    samples = sample_survival_times(grid, curves, lower_bounds, rng, n_samples=3)
    assert samples.shape == (4, 3)
    assert samples.dtype == np.float32


def test_sample_survival_times_lower_bound_respected():
    rng = np.random.default_rng(42)
    grid = np.linspace(0.0, 10.0, 100)
    curves = np.exp(-0.3 * grid[None, :])
    curves = np.tile(curves, (3, 1))
    lower_bounds = np.array([3.0, 5.0, 7.0])

    samples = sample_survival_times(grid, curves, lower_bounds, rng, n_samples=20)
    assert np.all(samples >= lower_bounds[:, None])


def test_sample_survival_times_reproducible():
    rng = np.random.default_rng(42)
    grid = np.linspace(0.0, 10.0, 100)
    curves = np.exp(-0.2 * grid[None, :])
    curves = np.tile(curves, (2, 1))
    lower_bounds = np.array([0.0, 0.0])

    s1 = sample_survival_times(grid, curves, lower_bounds, rng, n_samples=5)
    rng2 = np.random.default_rng(42)
    s2 = sample_survival_times(grid, curves, lower_bounds, rng2, n_samples=5)
    assert np.allclose(s1, s2)


def test_sample_survival_times_fallback_when_no_event():
    rng = np.random.default_rng(42)
    grid = np.linspace(0.0, 10.0, 50)
    curves = np.ones((2, 50), dtype=np.float64)
    lower_bounds = np.array([3.0, 5.0], dtype=np.float64)

    samples = sample_survival_times(grid, curves, lower_bounds, rng, n_samples=4)
    assert samples.shape == (2, 4)
    assert np.all(samples >= lower_bounds[:, None])
    assert np.all(samples >= grid[-1])


def test_kaplan_meier_survival_curve_basic():
    from tabicl.survival._curves import kaplan_meier_survival_curve

    t_obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    delta = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    grid = np.linspace(0.5, 7.0, 50)

    curve = kaplan_meier_survival_curve(t_obs, delta, grid)
    assert curve.shape == grid.shape
    assert np.all(curve >= 0.0)
    assert np.all(curve <= 1.0)
    assert np.all(np.diff(curve) <= 0.0)
    assert curve[0] == 1.0
    assert curve[-1] < 1.0
