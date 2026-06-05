"""Tests for parallel on-the-fly survival prior generation."""

from __future__ import annotations

from types import SimpleNamespace

import torch

import survival_prior
import tabicl.prior._survival as survival_module
from tabicl.prior._mlp_scm import MLPSCM
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


def test_mlp_scm_marks_infinite_draws_invalid(monkeypatch):
    model = MLPSCM(seq_len=4, num_features=2, num_layers=2, hidden_dim=5)
    bad_X = torch.tensor([
        [1.0, float("inf")],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
    ])
    y = torch.arange(4.0).unsqueeze(-1)
    monkeypatch.setattr(model, "handle_outputs", lambda causes, outputs: (bad_X.clone(), y.clone()))

    X_out, y_out = model()

    assert (X_out == 0.0).all()
    assert (y_out == -100.0).all()


def test_survival_prior_regenerates_nonfinite_processed_features(monkeypatch):
    prior = _make_dataset().prior
    calls = 0

    class FakeMLPSCM:
        def __init__(self, **kwargs):
            self.seq_len = kwargs["seq_len"]
            self.num_features = kwargs["num_features"]

        def __call__(self):
            nonlocal calls
            calls += 1
            X = torch.arange(
                self.seq_len * self.num_features, dtype=torch.float32,
            ).reshape(self.seq_len, self.num_features)
            if calls == 1:
                X[0, 0] = float("inf")
            y = torch.arange(self.seq_len, dtype=torch.float32)
            return X, y

    monkeypatch.setattr(survival_module, "MLPSCM", FakeMLPSCM)
    monkeypatch.setattr(
        prior.ph_sampler,
        "sample",
        lambda **kwargs: (
            torch.arange(1.0, 33.0),
            torch.tensor([0.0, 1.0] * 16),
            torch.arange(1.0, 33.0),
        ),
    )
    params = {
        **prior.fixed_hp,
        "prior_type": "mlp_scm",
        "seq_len": 32,
        "train_size": 32,
        "max_features": 4,
        "num_features": 4,
        "device": "cpu",
        "cat_prob": 0.0,
        "permute_features": False,
        "sampler_type": "ph",
        "baseline_type": "weibull",
        "baseline_params": {},
        "censor_scale": 1.0,
        "target_event_rate": 0.5,
        "min_event_rate": 0.4,
        "max_event_rate": 0.9,
        "_rng": prior._survival_rng if hasattr(prior, "_survival_rng") else None,
    }

    X, _, _, _, _ = prior.generate_dataset(params)

    assert calls == 2
    assert torch.isfinite(X).all()


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
