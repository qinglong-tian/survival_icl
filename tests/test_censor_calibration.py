from __future__ import annotations

import torch

from tabicl.prior._survival import (
    MIN_RAW_TIME,
    SurvivalSCMPrior,
    WeibullHazard,
    WeibullAFT,
    ProportionalHazardSampler,
    AcceleratedFailureTimeSampler,
    calibrate_censor_scale_by_quantile,
)


def test_exact_k_events_achieved():
    t_event = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    c_base = torch.ones_like(t_event)
    scale, diag = calibrate_censor_scale_by_quantile(t_event, c_base, 0.6)
    # Target 0.6: 3/5 events = 0.6. Scale must be ≤ the 4th value
    # since s = unique_vals[3] = 4.0 gives exactly 3 events under strict <.
    assert abs(diag["achieved"] - 0.6) < 0.01
    assert 3.0 < scale <= 4.0


def test_below_smallest_yields_zero():
    t_event = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    c_base = torch.ones_like(t_event)
    scale, diag = calibrate_censor_scale_by_quantile(t_event, c_base, 0.01)
    assert diag["achieved"] == 0.0


def test_above_largest_yields_all():
    t_event = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    c_base = torch.ones_like(t_event)
    scale, diag = calibrate_censor_scale_by_quantile(t_event, c_base, 0.99)
    assert diag["achieved"] == 1.0


def test_ties_correct_handling():
    t_event = torch.tensor([2.0, 2.0, 2.0, 4.0, 4.0])
    c_base = torch.ones_like(t_event)
    scale, diag = calibrate_censor_scale_by_quantile(t_event, c_base, 0.3)
    # ratios: [2,2,2,4,4], s < 2 → 0, 2 < s < 4 → 3/5=0.6, s > 4 → 1.0
    # target 0.3, closest achievable is 0 with error 0.3
    assert diag["achieved"] == 0.0


def test_monotonic():
    t_event = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
    c_base = torch.ones_like(t_event)
    targets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    achieved = []
    for p in targets:
        scale, diag = calibrate_censor_scale_by_quantile(t_event, c_base, p)
        achieved.append(diag["achieved"])
    # Should be non-decreasing
    for i in range(len(achieved) - 1):
        assert achieved[i] <= achieved[i + 1]


def test_scale_clamped_to_eps():
    t_event = torch.tensor([0.0, 0.0, 0.0, 0.0, 5.0])
    c_base = torch.tensor([1e10, 1e10, 1e10, 1e10, 1.0])
    scale, diag = calibrate_censor_scale_by_quantile(t_event, c_base, 0.5, eps=1e-12)
    assert scale >= 1e-12


def test_survival_sanity_accepts_sanitized_time_floor():
    t = torch.tensor([MIN_RAW_TIME, 0.1, 1.0, 10.0])
    delta = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert SurvivalSCMPrior._survival_sanity_check(t, delta)


def test_survival_sanity_rejects_time_below_sanitized_floor():
    t = torch.tensor([0.0, 0.1, 1.0, 10.0])
    delta = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert not SurvivalSCMPrior._survival_sanity_check(t, delta)


def test_survival_sanity_retains_independent_variance_guard():
    t = torch.full((4,), MIN_RAW_TIME)
    delta = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert not SurvivalSCMPrior._survival_sanity_check(t, delta)


def test_context_calibration_invariant_to_query_rows():
    """Changing query rows must not change censoring scale."""
    import numpy as np

    # PH sampler
    baseline_ph = {"weibull": WeibullHazard()}
    sampler_ph = ProportionalHazardSampler(baseline_ph, beta=1.0, max_time=1e30, u_eps=1e-6)
    sampler_ph._calibration_prefix = 5

    y = torch.randn(10)
    rng = np.random.default_rng(42)
    params = {"k": 1.0}

    # Two batches: same context (first 5), different query (last 5)
    y1 = y.clone()
    y2 = y1.clone()
    y2[5:] = torch.randn(5) * 100  # wildly different query rows, same context

    # Reset torch RNG so that uniform draws inside sample() are identical
    torch.manual_seed(42)
    t1, d1, te1 = sampler_ph.sample(
        y1, "weibull", params, rng, device="cpu",
        censoring_strategy="target_event_rate",
        target_event_rate=0.5,
    )
    torch.manual_seed(42)
    t2, d2, te2 = sampler_ph.sample(
        y2, "weibull", params, rng, device="cpu",
        censoring_strategy="target_event_rate",
        target_event_rate=0.5,
    )

    # Context rows (first 5) must have identical t_obs and delta
    assert torch.allclose(t1[:5], t2[:5])
    assert torch.equal(d1[:5], d2[:5])

    # AFT sampler
    baseline_aft = {"weibull": WeibullAFT()}
    sampler_aft = AcceleratedFailureTimeSampler(
        baseline_aft, beta=1.0, max_time=1e30, u_eps=1e-6,
    )
    sampler_aft._calibration_prefix = 5

    torch.manual_seed(43)
    t3, d3, te3 = sampler_aft.sample(
        y1, "weibull", params, rng, device="cpu",
        censoring_strategy="target_event_rate",
        target_event_rate=0.5,
    )
    torch.manual_seed(43)
    t4, d4, te4 = sampler_aft.sample(
        y2, "weibull", params, rng, device="cpu",
        censoring_strategy="target_event_rate",
        target_event_rate=0.5,
    )
    assert torch.allclose(t3[:5], t4[:5])
    assert torch.equal(d3[:5], d4[:5])


if __name__ == "__main__":
    test_exact_k_events_achieved()
    test_below_smallest_yields_zero()
    test_above_largest_yields_all()
    test_ties_correct_handling()
    test_monotonic()
    test_scale_clamped_to_eps()
    test_survival_sanity_accepts_sanitized_time_floor()
    test_survival_sanity_rejects_time_below_sanitized_floor()
    test_survival_sanity_retains_independent_variance_guard()
    test_context_calibration_invariant_to_query_rows()
    print("All 10 tests passed.")
