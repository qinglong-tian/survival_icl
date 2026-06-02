"""Generate a few survival batches, apply KM-hybrid scaling, and report z-distribution."""

from __future__ import annotations

import time
import numpy as np
import torch

from survival_prior import SurvivalPriorDataset
from tabicl.survival import SurvivalTimeScaler

torch.set_printoptions(precision=4, sci_mode=False)


def collect_z(dataset, n_batches: int, label: str):
    """Generate n_batches, scale each dataset, collect z-values."""
    z_context_all, z_query_all = [], []
    t_ctx_all, t_qry_all = [], []
    d_ctx_all, d_qry_all = [], []
    n_km, n_fallback, n_scale_clamped = 0, 0, 0
    start = time.time()

    print(f"\n{'='*56}")
    print(f"  {label}  ({n_batches} batches)")
    print(f"{'='*56}")

    for b in range(n_batches):
        X, t, delta, t_event, d, seq_lens, train_sizes = dataset.get_batch()
        B = X.shape[0]

        if t.is_nested:
            t = t.to_padded_tensor(padding=0.0)
            delta = delta.to_padded_tensor(padding=0.0).float()

        for ds in range(B):
            seq = seq_lens[ds].item()
            ctx_n = seq // 2

            t_ds = t[ds, :seq]
            d_ds = delta[ds, :seq]

            t_ctx = t_ds[:ctx_n]
            d_ctx = d_ds[:ctx_n]
            t_qry = t_ds[ctx_n:seq]
            d_qry = d_ds[ctx_n:seq]

            scaler = SurvivalTimeScaler().fit(t_ctx, d_ctx)

            if scaler.metadata["location_source"] == "km":
                n_km += 1
            else:
                n_fallback += 1
            if scaler.metadata.get("scale_was_lower_bounded", False):
                n_scale_clamped += 1

            z_ctx, _ = scaler.transform_observed(t_ctx, d_ctx)
            z_qry, _ = scaler.transform_observed(t_qry, d_qry)

            z_context_all.append(z_ctx.numpy())
            z_query_all.append(z_qry.numpy())
            t_ctx_all.append(t_ctx.numpy())
            t_qry_all.append(t_qry.numpy())
            d_ctx_all.append(d_ctx.numpy())
            d_qry_all.append(d_qry.numpy())

    elapsed = time.time() - start
    z_c = np.concatenate(z_context_all)
    z_q = np.concatenate(z_query_all)
    t_c = np.concatenate(t_ctx_all)
    t_q = np.concatenate(t_qry_all)
    d_c = np.concatenate(d_ctx_all)
    d_q = np.concatenate(d_qry_all)

    print(f"  {len(z_c):,} context + {len(z_q):,} query obs in {elapsed:.1f}s")
    print(f"  KM used: {n_km}/{B*n_batches}, fallback: {n_fallback}, scale-clamped: {n_scale_clamped}")

    return z_c, z_q, t_c, t_q, d_c, d_q


def summarize(label: str, z: np.ndarray, t_raw: np.ndarray, d: np.ndarray):
    print(f"\n{'─'*56}")
    print(f"  {label}  (n={len(z):,})")
    print(f"{'─'*56}")
    evt = z[d > 0.5] if (d > 0.5).any() else z[:0]
    cen = z[d < 0.5] if (d < 0.5).any() else z[:0]
    print(f"  events: {len(evt):,}   censored: {len(cen):,}")
    print(f"  event rate (before admin censoring): {d.mean():.3f}")

    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    qv = np.quantile(z, qs)

    print(f"  {'':>6s}  {'all z':>10s}  {'events':>10s}  {'censored':>10s}  {'raw t':>12s}")
    print(f"  {'min':>6s}  {z.min():>10.4f}  {evt.min() if len(evt) else float('nan'):>10.4f}  {cen.min() if len(cen) else float('nan'):>10.4f}  {t_raw.min():>12.4f}")
    print(f"  {'mean':>6s}  {z.mean():>10.4f}  {evt.mean() if len(evt) else float('nan'):>10.4f}  {cen.mean() if len(cen) else float('nan'):>10.4f}  {t_raw.mean():>12.4f}")
    print(f"  {'std':>6s}  {z.std():>10.4f}  {evt.std() if len(evt) else float('nan'):>10.4f}  {cen.std() if len(cen) else float('nan'):>10.4f}  {t_raw.std():>12.4f}")
    for q, v in zip(qs, qv):
        ve = np.quantile(evt, q) if len(evt) else float('nan')
        vc = np.quantile(cen, q) if len(cen) else float('nan')
        print(f"  {int(q*100):>3d}%   {v:>10.4f}  {ve:>10.4f}  {vc:>10.4f}")
    print(f"  {'max':>6s}  {z.max():>10.4f}  {evt.max() if len(evt) else float('nan'):>10.4f}  {cen.max() if len(cen) else float('nan'):>10.4f}  {t_raw.max():>12.4f}")

    # Bounds check
    below = (z < -6.0).sum()
    above = (z > 6.0).sum()
    at_n6 = (np.isclose(z, -6.0)).sum()
    at_p6 = (np.isclose(z, 6.0)).sum()
    if below or above or at_n6 or at_p6:
        print(f"  bounds:  <-6: {below}  ==-6: {at_n6}  ==+6: {at_p6}  >+6: {above}")


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    N = 4   # batches
    bs = 128  # datasets per batch

    ph = SurvivalPriorDataset(
        batch_size=bs, max_seq_len=512, max_features=30,
        prior_type="mlp_scm", model_type="ph", beta=1.0,
        baseline_types=["weibull", "gompertz", "loglogistic", "lognormal"],
        baseline_mode="mix",
        min_censor_scale=1.0, max_censor_scale=5.0,
        min_event_rate=0.40, max_event_rate=1.0,
        device="cpu", n_jobs=4,
        min_train_size=1.0, max_train_size=1.0,
    )

    aft = SurvivalPriorDataset(
        batch_size=bs, max_seq_len=512, max_features=30,
        prior_type="mlp_scm", model_type="aft", beta=1.0,
        baseline_types=["weibull", "loglogistic", "lognormal"],
        baseline_mode="mix",
        min_censor_scale=1.0, max_censor_scale=5.0,
        min_event_rate=0.40, max_event_rate=1.0,
        device="cpu", n_jobs=4,
        min_train_size=1.0, max_train_size=1.0,
    )

    zc_ph, zq_ph, tc_ph, tq_ph, dc_ph, dq_ph = collect_z(ph, N, "PH")
    zc_aft, zq_aft, tc_aft, tq_aft, dc_aft, dq_aft = collect_z(aft, N, "AFT")

    print(f"\n{'='*56}")
    print(f"  STANDARDIZED Z-VALUE DISTRIBUTIONS")
    print(f"{'='*56}")

    summarize("PH  context", zc_ph, tc_ph, dc_ph)
    summarize("PH  query",   zq_ph, tq_ph, dq_ph)
    summarize("AFT context", zc_aft, tc_aft, dc_aft)
    summarize("AFT query",   zq_aft, tq_aft, dq_aft)

    print(f"\n{'─'*56}")
    print(f"  z_min={-6.0}  z_max={6.0}  (default SurvivalTimeScaler bounds)")
    print(f"  All z-values should fall in [{-6.0}, {6.0}].")


if __name__ == "__main__":
    main()
