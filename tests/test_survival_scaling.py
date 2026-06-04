from __future__ import annotations

import torch

from tabicl.survival import (
    SurvivalTimeScaler,
    TimeBinner,
    km_quantiles,
    standardize_survival_micro_batch,
)


def test_km_quantiles_no_censoring_use_step_convention():
    log_y = torch.log(torch.tensor([1.0, 2.0, 4.0, 8.0]))
    delta = torch.ones(4)

    q25, q50, q75 = km_quantiles(log_y, delta)

    expected = torch.log(torch.tensor([1.0, 2.0, 4.0]))
    assert torch.allclose(torch.stack([q25, q50, q75]), expected)


def test_km_quantiles_heavy_censoring_returns_nan_for_unreached_quantiles():
    log_y = torch.log(torch.tensor([1.0, 2.0, 4.0]))
    delta = torch.tensor([1.0, 0.0, 0.0])

    q25, q50, q75 = km_quantiles(log_y, delta)

    assert torch.isfinite(q25)
    assert torch.isnan(q50)
    assert torch.isnan(q75)


def test_km_quantiles_ties_events_jump_before_risk_set_removal():
    log_y = torch.log(torch.tensor([1.0, 1.0, 2.0, 2.0, 4.0]))
    delta = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0])

    q25, q50, q75 = km_quantiles(log_y, delta)

    expected = torch.log(torch.tensor([1.0, 2.0, 4.0]))
    assert torch.allclose(torch.stack([q25, q50, q75]), expected)


def test_survival_time_scaler_uses_km_median_iqr_when_available():
    t_context = torch.tensor([1.0, 2.0, 4.0, 8.0])
    delta = torch.ones_like(t_context)
    scaler = SurvivalTimeScaler().fit(t_context, delta)

    expected_loc = torch.log(torch.tensor(2.0))
    expected_scale = (torch.log(torch.tensor(4.0)) - torch.log(torch.tensor(1.0))) / 1.349

    assert torch.allclose(scaler.loc, expected_loc)
    assert torch.allclose(scaler.scale, expected_scale)
    assert scaler.metadata["location_source"] == "km"
    assert scaler.metadata["scale_source"] == "km"


def test_survival_time_scaler_falls_back_when_km_quantiles_unavailable():
    t_context = torch.tensor([1.0, 2.0, 4.0])
    delta = torch.tensor([1.0, 0.0, 0.0])
    scaler = SurvivalTimeScaler().fit(t_context, delta)

    log_t = torch.log(t_context)
    q25, q50, q75 = torch.quantile(log_t, torch.tensor([0.25, 0.5, 0.75]))
    expected_scale = ((q75 - q25) / 1.349).clamp_min(0.1)

    assert torch.allclose(scaler.loc, q50)
    assert torch.allclose(scaler.scale, expected_scale)
    assert scaler.metadata["location_source"] == "observed_fallback"
    assert scaler.metadata["scale_source"] == "observed_fallback"


def test_survival_time_scaler_is_scale_invariant_and_round_trips_unclipped_times():
    t_context = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
    t_query = torch.tensor([2.0, 4.0, 8.0])

    delta = torch.ones_like(t_context)
    scaler = SurvivalTimeScaler().fit(t_context, delta)
    scaled_scaler = SurvivalTimeScaler().fit(t_context * 365.0, delta)

    z = scaler.transform_time(t_query)
    z_scaled = scaled_scaler.transform_time(t_query * 365.0)

    assert torch.allclose(z, z_scaled, atol=1e-5)
    assert torch.allclose(scaler.inverse_time(z), t_query, atol=1e-5)


def test_preprocessing_helper_scales_context_only():
    """Context z-values must be invariant to extreme query times.

    Calls ``standardize_survival_micro_batch`` with identical context but
    wildly different query values — standardized context must match, proving
    the helper fits exclusively on context.
    """
    scaler_kwargs = dict(eps=1e-8, min_scale=0.1, z_min=-6.0, z_max=6.0)

    ctx = torch.tensor([1.0, 2.0, 4.0])
    delta_ctx = torch.ones_like(ctx)
    # Pad to batch dim and add one fake test position for the helper's shape
    t_train = ctx.unsqueeze(0)             # (1, 3)
    d_train = delta_ctx.unsqueeze(0)

    # Two extreme query sets
    q1 = torch.tensor([[8.0, 16.0]])       # benign
    q2 = torch.tensor([[1e10, 1e-10]])     # extreme — must not affect scaling

    # Helper splits at train_sizes_ds = seq_len // 2 internally,
    # but the full row is passed as t_train + remainder as t_test.
    # We pass the full row as both t_train and t_test (train_sizes=3,
    # query_sizes=2) so the scaler fits on the first 3 positions only.
    out1 = standardize_survival_micro_batch(
        t_train, d_train, q1, torch.ones_like(q1), q1,
        train_sizes_ds=torch.tensor([3]), query_sizes_ds=torch.tensor([2]),
        scaler_kwargs=scaler_kwargs,
    )
    out2 = standardize_survival_micro_batch(
        t_train, d_train, q2, torch.ones_like(q2), q2,
        train_sizes_ds=torch.tensor([3]), query_sizes_ds=torch.tensor([2]),
        scaler_kwargs=scaler_kwargs,
    )
    z_ctx1 = out1[0][0, :3]  # context positions (unmasked)
    z_ctx2 = out2[0][0, :3]
    assert torch.allclose(z_ctx1, z_ctx2, atol=1e-5)

    # Context-only fit reference
    ref = SurvivalTimeScaler(**scaler_kwargs).fit(ctx, delta_ctx)
    z_ref, _ = ref.transform_observed(ctx, delta_ctx)
    assert torch.allclose(z_ctx1, z_ref, atol=1e-5)


def test_survival_time_scaler_handles_extremes_and_administrative_censoring():
    scaler = SurvivalTimeScaler().fit(torch.tensor([1.0, 2.0, 4.0]), torch.ones(3))

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

    delta_ctx = torch.ones_like(t_ctx_a)
    scaler_a = SurvivalTimeScaler(**scaler_kwargs).fit(t_ctx_a, delta_ctx)
    scaler_b = SurvivalTimeScaler(**scaler_kwargs).fit(t_ctx_b, delta_ctx)

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


def test_survival_preprocessing_helper_uses_context_delta_and_stays_in_range():
    scaler_kwargs = {
        "eps": 1e-8,
        "min_scale": 0.1,
        "z_min": -6.0,
        "z_max": 6.0,
    }

    t_train = torch.tensor([[1.0, 2.0, 4.0], [365.0, 730.0, 1460.0]])
    delta_train = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    t_test = torch.tensor([[8.0, 1e30], [2920.0, 365.0e30]])
    delta_test = torch.ones_like(t_test)
    t_event_test = t_test.clone()
    train_sizes_ds = torch.tensor([3, 3])
    query_sizes_ds = torch.tensor([2, 2])

    outputs = standardize_survival_micro_batch(
        t_train, delta_train, t_test, delta_test, t_event_test,
        train_sizes_ds, query_sizes_ds, scaler_kwargs,
    )
    t_train_z, delta_train_z, t_test_z, delta_test_z, t_event_test_z, t_event_in_range = outputs

    for tensor in (t_train_z, t_test_z, t_event_test_z):
        assert torch.isfinite(tensor).all()
    # t_event_test_z is unclipped (transform_event_target returns raw_z)
    # — only context/query observed are clamped to [z_min, z_max]
    for tensor in (t_train_z, t_test_z):
        assert ((tensor >= -6.0) & (tensor <= 6.0)).all()

    assert torch.allclose(t_train_z[0], t_train_z[1], atol=1e-5)
    assert torch.allclose(t_test_z[0, 0], t_test_z[1, 0], atol=1e-5)
    assert delta_train_z.tolist() == delta_train.tolist()
    assert delta_test_z[:, 1].tolist() == [0.0, 0.0]
