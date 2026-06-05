"""Tests for parallel on-the-fly survival prior generation."""

from __future__ import annotations

from types import SimpleNamespace

import torch

import survival_prior
from survival_prior import SurvivalPriorDataset


def _make_dataset(*, n_jobs: int = 1) -> SurvivalPriorDataset:
    return SurvivalPriorDataset(
        batch_size=4,
        batch_size_per_gp=2,
        min_features=2,
        max_features=4,
        min_seq_len=32,
        max_seq_len=32,
        prior_type="mlp_scm",
        model_type="ph",
        baseline_types=["weibull"],
        baseline_mode="weibull",
        min_event_rate=0.4,
        max_event_rate=0.9,
        censoring_strategy="target_event_rate",
        calibration_scope="context",
        n_jobs=n_jobs,
        device="cpu",
    )


def test_parallel_survival_prior_generation_is_finite():
    dataset = _make_dataset(n_jobs=2)

    X, t, delta, t_event, d, seq_lens, train_sizes = dataset.get_batch()

    assert dataset.prior.n_jobs == 2
    assert X.shape == (4, 32, 4)
    assert t.shape == delta.shape == t_event.shape == (4, 32)
    assert d.shape == seq_lens.shape == train_sizes.shape == (4,)
    assert torch.isfinite(X).all()
    assert torch.isfinite(t).all()
    assert torch.isfinite(t_event).all()
    assert ((delta == 0) | (delta == 1)).all()


def test_dataloader_workers_get_independent_survival_rngs(monkeypatch):
    first = _make_dataset()
    monkeypatch.setattr(
        survival_prior,
        "get_worker_info",
        lambda: SimpleNamespace(seed=123),
    )
    iter(first)
    first_draw = first.prior._survival_rng.integers(0, 2**31, size=4)
    first_rng = first.prior._survival_rng
    iter(first)
    assert first.prior._survival_rng is first_rng

    second = _make_dataset()
    monkeypatch.setattr(
        survival_prior,
        "get_worker_info",
        lambda: SimpleNamespace(seed=124),
    )
    iter(second)
    second_draw = second.prior._survival_rng.integers(0, 2**31, size=4)

    assert not (first_draw == second_draw).all()
