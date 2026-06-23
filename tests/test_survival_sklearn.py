"""Tests for the sklearn-style survival estimator."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabicl._model.tabicl import TabICL
from tabicl.survival import (
    TabICLSurvivalEstimator,
    TimeBinner,
    impute_censored_survival_times,
)


def tiny_survival_checkpoint(path):
    config = {
        "max_classes": 0,
        "num_quantiles": 8,
        "embed_dim": 16,
        "col_num_blocks": 1,
        "col_nhead": 2,
        "col_num_inds": 4,
        "row_num_blocks": 1,
        "row_nhead": 2,
        "row_num_cls": 2,
        "icl_num_blocks": 1,
        "icl_nhead": 2,
        "ff_factor": 2,
        "survival": True,
    }
    model = TabICL(**config)
    binner = TimeBinner.from_standardized_range(num_bins=8)
    torch.save({
        "config": config,
        "state_dict": model.state_dict(),
        "curr_step": 0,
        "survival_metadata": {
            "task": "survival",
            "time_scale": "km_hybrid_log",
            "num_bins": 8,
            "binner_edges": binner.bin_edges,
            "binner_means": binner.bin_means,
            "time_scaler": {"eps": 1e-8, "min_scale": 0.1, "z_min": -6.0, "z_max": 6.0},
        },
    }, path)


def test_survival_estimator_predicts_raw_scale_monotone_curves(tmp_path):
    checkpoint = tmp_path / "step-0.ckpt"
    tiny_survival_checkpoint(checkpoint)
    X = np.array([
        [0.0, 10.0],
        [1.0, 12.0],
        [2.0, 14.0],
        [3.0, 16.0],
        [4.0, 18.0],
        [5.0, 20.0],
    ], dtype=np.float32)
    y = np.column_stack([
        np.array([1.0, 2.0, 3.0, 5.0, 8.0, 13.0], dtype=np.float32),
        np.array([1, 0, 1, 1, 0, 1], dtype=np.float32),
    ])
    estimator = TabICLSurvivalEstimator(
        checkpoint,
        device="cpu",
        query_batch_size=2,
    ).fit(X, y)

    grid, curves = estimator.predict_survival_function(
        X[:3], return_times=True,
    )

    assert grid.ndim == 1
    assert curves.shape == (3, grid.shape[0])
    assert np.all(np.diff(grid) > 0)
    assert np.isfinite(curves).all()
    assert np.all(np.diff(curves, axis=1) <= 1e-6)
    assert estimator.feature_scaler_.mean_.tolist() == pytest.approx([2.5, 15.0])


def test_survival_estimator_conditions_on_censored_query_times(tmp_path):
    checkpoint = tmp_path / "step-0.ckpt"
    tiny_survival_checkpoint(checkpoint)
    X = np.random.default_rng(0).normal(size=(8, 3)).astype(np.float32)
    estimator = TabICLSurvivalEstimator(checkpoint, device="cpu").fit(
        X,
        t=np.linspace(1.0, 8.0, 8, dtype=np.float32),
        delta=np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=np.float32),
    )
    times = np.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float32)
    conditional_time = np.array([4.0, 8.0], dtype=np.float32)

    unconditional = estimator.predict_survival_function(X[:2], times=times)
    conditional = estimator.predict_survival_function(
        X[:2], times=times, conditional_time=conditional_time,
    )

    assert conditional.shape == unconditional.shape
    assert np.allclose(conditional[0, times <= 4.0], 1.0)
    assert np.allclose(conditional[1, times <= 8.0], 1.0)
    assert np.all((conditional >= 0.0) & (conditional <= 1.0))
    assert np.all(np.diff(conditional, axis=1) <= 1e-6)


def test_survival_estimator_predict_returns_median(tmp_path):
    checkpoint = tmp_path / "step-0.ckpt"
    tiny_survival_checkpoint(checkpoint)
    X = np.random.default_rng(1).normal(size=(6, 2)).astype(np.float32)
    y = {"time": np.arange(1.0, 7.0), "event": np.ones(6)}
    estimator = TabICLSurvivalEstimator(checkpoint, device="cpu").fit(X, y)

    median = estimator.predict(X[:4])
    quantile = estimator.predict_quantiles(X[:4], quantile_levels=(0.5,))

    assert median.shape == (4,)
    assert np.allclose(median, quantile[:, 0])


def test_impute_censored_survival_times_returns_completed_vectors(tmp_path):
    checkpoint = tmp_path / "step-0.ckpt"
    tiny_survival_checkpoint(checkpoint)
    X = np.random.default_rng(2).normal(size=(8, 3)).astype(np.float32)
    t = np.linspace(1.0, 8.0, 8, dtype=np.float32)
    delta = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=np.float32)

    result = impute_censored_survival_times(
        checkpoint,
        X,
        t,
        delta,
        n_soft_samples=3,
        random_state=123,
        device="cpu",
        query_batch_size=2,
    )

    assert result.censored_indices.tolist() == [1, 4, 6]
    assert result.hard_times.shape == (3,)
    assert result.soft_times.shape == (3, 3)
    assert result.completed_hard_times.shape == t.shape
    assert result.completed_soft_times.shape == (8, 3)
    assert np.all(result.hard_times >= t[result.censored_indices])
    assert np.all(result.soft_times >= t[result.censored_indices, None])
    assert np.allclose(result.completed_hard_times[delta == 1], t[delta == 1])
    assert np.allclose(result.completed_soft_times[delta == 1], t[delta == 1, None])
    assert np.all(np.diff(result.conditional_survival, axis=1) <= 1e-6)


def test_impute_censored_survival_times_supports_mean_and_reproducibility(tmp_path):
    checkpoint = tmp_path / "step-0.ckpt"
    tiny_survival_checkpoint(checkpoint)
    X = np.random.default_rng(3).normal(size=(7, 2)).astype(np.float32)
    t = np.arange(1.0, 8.0, dtype=np.float32)
    delta = np.array([1, 0, 1, 0, 1, 1, 0], dtype=np.float32)

    first = impute_censored_survival_times(
        checkpoint,
        X,
        t,
        delta,
        hard_method="mean",
        n_soft_samples=2,
        random_state=12,
    )
    second = impute_censored_survival_times(
        checkpoint,
        X,
        t,
        delta,
        hard_method="mean",
        n_soft_samples=2,
        random_state=12,
    )

    assert first.hard_method == "mean"
    assert np.all(first.hard_times >= t[first.censored_indices])
    assert np.allclose(first.soft_times, second.soft_times)


def test_impute_censored_survival_times_supports_unconditional_mode(tmp_path):
    checkpoint = tmp_path / "step-0.ckpt"
    tiny_survival_checkpoint(checkpoint)
    X = np.random.default_rng(6).normal(size=(7, 2)).astype(np.float32)
    t = np.arange(1.0, 8.0, dtype=np.float32)
    delta = np.array([1, 0, 1, 0, 1, 1, 0], dtype=np.float32)

    result = impute_censored_survival_times(
        checkpoint,
        X,
        t,
        delta,
        n_soft_samples=2,
        random_state=10,
        condition_on_censoring=False,
    )

    assert result.condition_on_censoring is False
    assert result.survival_curves is result.conditional_survival
    assert result.hard_times.shape == (3,)
    assert result.soft_times.shape == (3, 2)
    assert np.isfinite(result.hard_times).all()
    assert np.isfinite(result.soft_times).all()


def test_impute_censored_survival_times_no_censored_rows(tmp_path):
    checkpoint = tmp_path / "step-0.ckpt"
    tiny_survival_checkpoint(checkpoint)
    X = np.random.default_rng(4).normal(size=(5, 2)).astype(np.float32)
    t = np.arange(1.0, 6.0, dtype=np.float32)
    delta = np.ones(5, dtype=np.float32)

    result = impute_censored_survival_times(
        checkpoint,
        X,
        t,
        delta,
        n_soft_samples=2,
    )

    assert result.censored_indices.size == 0
    assert result.hard_times.size == 0
    assert result.soft_times.shape == (0, 2)
    assert np.allclose(result.completed_hard_times, t)
    assert np.allclose(result.completed_soft_times, t[:, None])
