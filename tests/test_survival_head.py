"""Tests for survival head, NLL loss, and quantile stability."""

from __future__ import annotations

import torch
import pytest

from tabicl.survival import TimeBinner, discrete_survival_nll


# ---------------------------------------------------------------------------
# discrete_survival_nll
# ---------------------------------------------------------------------------


def test_nll_event_closed_form():
    """Single-bin event: NLL = -log(h)."""
    h_raw = torch.tensor([[0.0]])  # sigmoid(0) = 0.5
    bin_idx = torch.tensor([0])
    delta = torch.tensor([1.0])
    loss = discrete_survival_nll(h_raw, bin_idx, delta)
    # h = 0.5, S_km1 = 1.0, NLL = -log(0.5) = 0.6931
    assert torch.isfinite(loss)
    assert torch.allclose(loss, torch.tensor(0.6931), atol=1e-3)


def test_nll_censored_closed_form():
    """Single-bin censored: NLL = -log(1-h)."""
    h_raw = torch.tensor([[0.0]])  # sigmoid(0) = 0.5
    bin_idx = torch.tensor([0])
    delta = torch.tensor([0.0])
    loss = discrete_survival_nll(h_raw, bin_idx, delta)
    # h = 0.5, S_k = 1-h = 0.5, NLL = -log(0.5) = 0.6931
    assert torch.isfinite(loss)
    assert torch.allclose(loss, torch.tensor(0.6931), atol=1e-3)


def test_nll_finite_under_large_logits():
    """NLL stays finite even with extreme logits (float32)."""
    h_raw = torch.tensor([[-100.0, 100.0, -50.0]], dtype=torch.float32)
    bin_idx = torch.tensor([0])
    delta = torch.tensor([1.0])
    loss = discrete_survival_nll(h_raw, bin_idx, delta)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_nll_finite_under_float16():
    """NLL stays finite under float16 inputs (computed in float32)."""
    h_raw = torch.tensor([[-10.0, -10.0, -10.0]], dtype=torch.float16)
    bin_idx = torch.tensor([0])
    delta = torch.tensor([1.0])
    loss = discrete_survival_nll(h_raw.float(), bin_idx, delta.float())
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_nll_event_ignores_infinite_censoring_branch():
    """An impossible censoring branch must not create 0 * inf = nan."""
    h_raw = torch.tensor([[float("inf"), float("inf")]])
    loss = discrete_survival_nll(
        h_raw,
        bin_idx=torch.tensor([0]),
        delta=torch.tensor([1.0]),
    )
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0)


def test_nll_censoring_ignores_infinite_event_branch():
    """An impossible event branch must not create 0 * inf = nan."""
    h_raw = torch.tensor([[-float("inf"), -float("inf")]])
    loss = discrete_survival_nll(
        h_raw,
        bin_idx=torch.tensor([0]),
        delta=torch.tensor([0.0]),
    )
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0)


def test_nll_multi_bin_event():
    """Event at bin 2: includes hazard of bins 0,1 in survival term."""
    h_raw = torch.tensor([[0.0, 0.0, 0.0]])  # all h = 0.5
    bin_idx = torch.tensor([2])
    delta = torch.tensor([1.0])
    loss = discrete_survival_nll(h_raw, bin_idx, delta)
    # h_2 = 0.5, S_before_2 = (1-0.5)*(1-0.5) = 0.25
    # NLL = -log(0.5 * 0.25) = -log(0.125) = 2.079
    assert torch.allclose(loss, torch.tensor(2.0794), atol=1e-3)


def test_nll_multi_bin_censored():
    """Censored at bin 1: only survival up to and including bin 1."""
    h_raw = torch.tensor([[0.0, 0.0, 0.0]])  # all h = 0.5
    bin_idx = torch.tensor([1])
    delta = torch.tensor([0.0])
    loss = discrete_survival_nll(h_raw, bin_idx, delta)
    # log S(τ_1) = log(0.5) + log(0.5) = -1.386
    assert torch.allclose(loss, torch.tensor(1.3863), atol=1e-3)


# ---------------------------------------------------------------------------
# quantile_at stability
# ---------------------------------------------------------------------------


def test_quantile_at_unreached_returns_z_max():
    """When CDF never reaches the requested probability, return z_max."""
    binner = TimeBinner.from_standardized_range(num_bins=10, z_min=-6.0, z_max=6.0)
    # All hazard logits very negative → CDF ≈ 0 everywhere
    h_raw = torch.tensor([[-100.0] * 10], dtype=torch.float32)
    probs = torch.tensor([0.5])
    q = binner.quantile_at(h_raw, probs)
    assert torch.isfinite(q).all()
    assert q.item() == 6.0  # z_max


def test_quantile_at_reached_interpolates():
    """When CDF reaches the probability, interpolate correctly."""
    binner = TimeBinner.from_standardized_range(num_bins=10, z_min=-6.0, z_max=6.0)
    # All hazards ~0.5 → eventual CDF near 1.0
    h_raw = torch.tensor([[0.0] * 10], dtype=torch.float32)
    probs = torch.tensor([0.5])
    q = binner.quantile_at(h_raw, probs)
    assert torch.isfinite(q).all()
    assert -6.0 <= q.item() <= 6.0


def test_quantile_at_clamped_to_bin_range():
    """Quantiles stay inside [z_min, z_max] for any input."""
    binner = TimeBinner.from_standardized_range(num_bins=10, z_min=-6.0, z_max=6.0)
    for logit_val in [-100.0, -10.0, 0.0, 10.0, 100.0]:
        h_raw = torch.tensor([[logit_val] * 10], dtype=torch.float32)
        for p in [0.1, 0.5, 0.9]:
            q = binner.quantile_at(h_raw, torch.tensor([p]))
            assert torch.isfinite(q).all()
            assert -6.0 <= q.item() <= 6.0, f"logit={logit_val}, p={p}, q={q.item()}"


def test_quantile_at_rejects_invalid_probs():
    """Validate probabilities are finite and in (0, 1)."""
    binner = TimeBinner.from_standardized_range(num_bins=10)
    h_raw = torch.tensor([[0.0] * 10])
    with pytest.raises(ValueError, match="finite"):
        binner.quantile_at(h_raw, torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        binner.quantile_at(h_raw, torch.tensor([0.0]))
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        binner.quantile_at(h_raw, torch.tensor([1.0]))


# ---------------------------------------------------------------------------
# standardized_time_summary
# ---------------------------------------------------------------------------


def test_standardized_time_summary_includes_tail():
    """Capped expectation includes residual mass at z_max."""
    binner = TimeBinner.from_standardized_range(num_bins=10, z_min=-6.0, z_max=6.0)
    # All hazards very negative → survival stays near 1 → tail mass ≈ 1.0
    h_raw = torch.tensor([[-100.0] * 10], dtype=torch.float32)
    s = binner.standardized_time_summary(h_raw)
    assert torch.isfinite(s).all()
    # Should be close to z_max since nearly all mass is in the tail
    assert s.item() > 5.0


def test_expected_time_is_deprecated_alias():
    """expected_time warns and delegates to standardized_time_summary."""
    binner = TimeBinner.from_standardized_range(num_bins=10)
    h_raw = torch.tensor([[0.0] * 10])
    with pytest.warns(DeprecationWarning, match="expected_time is deprecated"):
        result_dep = binner.expected_time(h_raw)
    result_new = binner.standardized_time_summary(h_raw)
    assert torch.allclose(result_dep, result_new)


# ---------------------------------------------------------------------------
# hazard / survival / CDF / event_prob_mass
# ---------------------------------------------------------------------------


def test_hazard_probs_in_01():
    binner = TimeBinner.from_standardized_range(num_bins=10)
    h = binner.hazard_probs(torch.randn(5, 10))
    assert (h > 0).all() and (h < 1).all()


def test_survival_monotonic():
    binner = TimeBinner.from_standardized_range(num_bins=10)
    S = binner.survival(torch.randn(3, 10))
    assert (S[:, 0] == 1.0).all()
    diff = S[:, 1:] - S[:, :-1]
    assert (diff <= 0).all()


def test_cdf_monotonic():
    binner = TimeBinner.from_standardized_range(num_bins=10)
    F = binner.cdf(torch.randn(3, 10))
    assert (F[:, 0] >= 0).all()
    diff = F[:, 1:] - F[:, :-1]
    assert (diff >= 0).all()


def test_event_prob_mass_sums_le_one():
    binner = TimeBinner.from_standardized_range(num_bins=10)
    p = binner.event_prob_mass(torch.randn(3, 10))
    assert (p >= 0).all()
    assert (p.sum(dim=-1) <= 1.0 + 1e-5).all()


# ---------------------------------------------------------------------------
# quantile_at vectorized
# ---------------------------------------------------------------------------


def test_quantile_at_unbatched_shape():
    """Unbatched h_raw (K,) should return (Q,), not (1, Q)."""
    binner = TimeBinner.from_standardized_range(num_bins=10)
    h_raw = torch.randn(10)
    probs = torch.tensor([0.1, 0.5, 0.9])
    q = binner.quantile_at(h_raw, probs)
    assert q.shape == (3,), f"Expected (3,), got {tuple(q.shape)}"


def test_quantile_at_batched_shape():
    """Batched h_raw (B, K) should return (B, Q)."""
    binner = TimeBinner.from_standardized_range(num_bins=10)
    h_raw = torch.randn(4, 10)
    probs = torch.tensor([0.1, 0.5, 0.9])
    q = binner.quantile_at(h_raw, probs)
    assert q.shape == (4, 3)


def test_quantile_at_rejects_multidim_probs():
    """Multidimensional probs must be rejected."""
    binner = TimeBinner.from_standardized_range(num_bins=10)
    h_raw = torch.randn(3, 10)
    probs = torch.tensor([[0.1, 0.5], [0.3, 0.7]])
    with pytest.raises(ValueError, match="1D"):
        binner.quantile_at(h_raw, probs)
