from __future__ import annotations

import torch

from tabicl.survival import SurvivalTimeScaler, TimeBinner


def test_survival_time_scaler_matches_median_iqr_formula():
    t_context = torch.exp(torch.tensor([0.0, 1.0, 3.0]))
    scaler = SurvivalTimeScaler().fit(t_context)

    log_t = torch.log(t_context)
    q25, q75 = torch.quantile(log_t, torch.tensor([0.25, 0.75]))
    expected_loc = log_t.median()
    expected_scale = ((q75 - q25) / 1.349).clamp_min(0.1)

    assert torch.allclose(scaler.loc, expected_loc)
    assert torch.allclose(scaler.scale, expected_scale)


def test_survival_time_scaler_is_scale_invariant_and_round_trips_unclipped_times():
    t_context = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
    t_query = torch.tensor([2.0, 4.0, 8.0])

    scaler = SurvivalTimeScaler().fit(t_context)
    scaled_scaler = SurvivalTimeScaler().fit(t_context * 365.0)

    z = scaler.transform_time(t_query)
    z_scaled = scaled_scaler.transform_time(t_query * 365.0)

    assert torch.allclose(z, z_scaled, atol=1e-5)
    assert torch.allclose(scaler.inverse_time(z), t_query, atol=1e-5)


def test_survival_time_scaler_handles_extremes_and_administrative_censoring():
    scaler = SurvivalTimeScaler().fit(torch.tensor([1.0, 2.0, 4.0]))

    z = scaler.transform_time(torch.tensor([1e-12, 1e6, 1e30]))
    assert torch.isfinite(z).all()
    assert ((z >= -6.0) & (z <= 6.0)).all()

    z_obs, delta_obs = scaler.transform_observed(
        torch.tensor([1e30, 2.0]), torch.tensor([1.0, 1.0])
    )
    assert z_obs[0].item() == 6.0
    assert delta_obs.tolist() == [0.0, 1.0]


def test_time_binner_from_standardized_range():
    binner = TimeBinner.from_standardized_range(num_bins=50, z_min=-6.0, z_max=6.0)

    assert binner.bin_edges.shape == (51,)
    assert binner.bin_means.shape == (50,)
    assert binner.bin_edges[0].item() == -6.0
    assert binner.bin_edges[-1].item() == 6.0

    idx = binner.bin_index(torch.tensor([-100.0, -6.0, 0.0, 6.0, 100.0]))
    assert idx[0].item() == 0
    assert idx[1].item() == 0
    assert 0 <= idx[2].item() < 50
    assert idx[3].item() == 49
    assert idx[4].item() == 49


def test_scaler_is_context_only_and_scale_invariant():
    """Per-dataset scaler fit on context only: identical z-values across
    scale-shifted datasets, administrative censoring of beyond-horizon test
    observations."""
    scaler_kwargs = dict(eps=1e-8, min_scale=0.1, z_min=-6.0, z_max=6.0)

    # Dataset  a: small raw times              Dataset b: times × 365
    t_ctx_a = torch.tensor([1.0, 2.0, 4.0])
    t_ctx_b = torch.tensor([365.0, 730.0, 1460.0])
    t_qry_a = torch.tensor([8.0, 1e30])
    t_qry_b = torch.tensor([2920.0, 365.0e30])

    scaler_a = SurvivalTimeScaler(**scaler_kwargs).fit(t_ctx_a)
    scaler_b = SurvivalTimeScaler(**scaler_kwargs).fit(t_ctx_b)

    # Context: scale-invariant z-values
    z_ctx_a, _ = scaler_a.transform_observed(t_ctx_a, torch.ones_like(t_ctx_a))
    z_ctx_b, _ = scaler_b.transform_observed(t_ctx_b, torch.ones_like(t_ctx_b))
    assert torch.allclose(z_ctx_a, z_ctx_b, atol=1e-5)

    # Query: first element is clean, second is beyond horizon
    z_qry_a, d_qry_a = scaler_a.transform_observed(t_qry_a, torch.ones_like(t_qry_a))
    z_qry_b, d_qry_b = scaler_b.transform_observed(t_qry_b, torch.ones_like(t_qry_b))

    assert torch.allclose(z_qry_a[0], z_qry_b[0], atol=1e-5)
    assert d_qry_a.tolist() == [1.0, 0.0]  # second obs administratively censored
    assert d_qry_b.tolist() == [1.0, 0.0]

    # Output range
    for z in (z_ctx_a, z_ctx_b, z_qry_a, z_qry_b):
        assert torch.isfinite(z).all()
        assert ((z >= -6.0) & (z <= 6.0)).all()

