from __future__ import annotations

import torch

from tabicl.prior._survival import calibrate_censor_scale_by_quantile


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
    # ratioss: [2,2,2,4,4], s < 2 → 0, 2 < s < 4 → 3/5=0.6, s > 4 → 1.0
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


if __name__ == "__main__":
    test_exact_k_events_achieved()
    test_below_smallest_yields_zero()
    test_above_largest_yields_all()
    test_ties_correct_handling()
    test_monotonic()
    test_scale_clamped_to_eps()
    print("All 6 tests passed.")
