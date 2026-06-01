"""Analyze the distribution of uncensored event times (t_event) from the survival prior.

Generates a large number of datasets using SurvivalPriorDataset across both PH and
AFT model types and reports five-number summary + high quantiles (0.95, 0.99, 0.999).
"""

from __future__ import annotations

import time
import numpy as np
import torch

from survival_prior import SurvivalPriorDataset

torch.set_printoptions(precision=4, sci_mode=False)


def collect_event_times(dataset: SurvivalPriorDataset, n_batches: int, label: str) -> np.ndarray:
    """Generate n_batches from dataset and return all t_event values as a flat numpy array."""
    all_t_event = []
    total_obs = 0
    total_ds = 0
    start = time.time()

    print(f"\n{'='*64}")
    print(f"{label}")
    print(f"  generating {n_batches} batches ...")
    print(f"{'='*64}")

    for i in range(n_batches):
        X, t, delta, t_event, d, seq_lens, train_sizes = dataset.get_batch()
        B = X.shape[0]

        if isinstance(t_event, torch.Tensor) and not t_event.is_nested:
            # Regular stacked tensor: (B, seq)
            t_flat = t_event.cpu().numpy().ravel()
        else:
            # Nested tensor (variable-length)
            t_flat = t_event.cpu().to_padded_tensor(nan=float('nan')).numpy().ravel()
            t_flat = t_flat[~np.isnan(t_flat)]

        all_t_event.append(t_flat)
        total_obs += len(t_flat)
        total_ds += B

        if (i + 1) % max(1, n_batches // 10) == 0:
            elapsed = time.time() - start
            print(f"    batch {i+1:6d}/{n_batches}  |  {total_obs:>10,} observations  |  {elapsed:.1f}s")

    elapsed = time.time() - start
    combined = np.concatenate(all_t_event)
    print(f"    done in {elapsed:.1f}s: {total_ds:,} datasets, {len(combined):,} total event times")
    return combined


def summarize(name: str, data: np.ndarray):
    """Print distribution summary for a flat array of event times."""
    print(f"\n{'─'*64}")
    print(f"  {name}")
    print(f"  n = {len(data):,}")

    qs = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
    quantiles = np.quantile(data, qs)

    print(f"  {'':>6s}  {'value':>12s}")
    print(f"  {'min':>6s}  {data.min():>12.6f}")
    print(f"  {'mean':>6s}  {data.mean():>12.4f}")
    print(f"  {'std':>6s}  {data.std():>12.4f}")
    for q, v in zip(qs, quantiles):
        qpct = int(q * 100) if q >= 0.10 else f"{q*100:g}%"
        label = f"{qpct:>6s}" if isinstance(qpct, str) else f"{qpct:>3d}%  "
        print(f"  {label}  {v:>12.6f}")
    print(f"  {'max':>6s}  {data.max():>12.6f}")


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    # Number of batches to generate per configuration
    N = 20

    # Use a moderate batch size — faster generation
    batch_size = 128
    max_seq = 512

    # ── PH, mixed baselines ──────────────────────────────────────────────
    ph = SurvivalPriorDataset(
        batch_size=batch_size,
        max_seq_len=max_seq,
        max_features=30,
        prior_type="mlp_scm",
        model_type="ph",
        beta=1.0,
        baseline_types=["weibull", "gompertz", "loglogistic", "lognormal"],
        baseline_mode="mix",
        max_time=1e12,  # disable the 100-clip so we see true event times
        min_censor_scale=1.0,
        max_censor_scale=5.0,
        min_event_rate=0.40,
        max_event_rate=1.0,
        device="cpu",
        n_jobs=4,
        min_train_size=1.0,
        max_train_size=1.0,
    )
    ph_data = collect_event_times(ph, N, "PH — mixed baselines")

    # ── AFT, mixed baselines ─────────────────────────────────────────────
    aft = SurvivalPriorDataset(
        batch_size=batch_size,
        max_seq_len=max_seq,
        max_features=30,
        prior_type="mlp_scm",
        model_type="aft",
        beta=1.0,
        baseline_types=["weibull", "loglogistic", "lognormal"],
        baseline_mode="mix",
        max_time=1e12,  # disable the 100-clip so we see true event times
        min_censor_scale=1.0,
        max_censor_scale=5.0,
        min_event_rate=0.40,
        max_event_rate=1.0,
        device="cpu",
        n_jobs=4,
        min_train_size=1.0,
        max_train_size=1.0,
    )
    aft_data = collect_event_times(aft, N, "AFT — mixed baselines")

    # ── Both combined ───────────────────────────────────────────────────
    combined = np.concatenate([ph_data, aft_data])

    # ── Report ──────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  DISTRIBUTION OF EVENT TIMES (t_event, uncensored)")
    print(f"{'='*64}")

    summarize("PH — Weibull + Gompertz + LogLogistic + LogNormal", ph_data)
    summarize("AFT — Weibull + LogLogistic + LogNormal", aft_data)
    summarize("COMBINED (PH + AFT)", combined)

    # ── Histogram bins ──────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print("  Histogram (log-spaced bins, uncensored)")
    print(f"{'─'*64}")

    # Log-spaced bins up to 99.9% quantile (skip extreme outliers for binning)
    cap = np.quantile(combined, 0.999)
    bins = np.geomspace(max(combined.min(), 1e-4), cap, 30)
    hist, edges = np.histogram(combined, bins=bins)
    max_bar = 60
    max_count = hist.max()
    for i in range(len(hist)):
        if hist[i] > 0:
            bar_len = int(hist[i] / max_count * max_bar)
            bar = "█" * bar_len
            print(f"  [{edges[i]:>10.2f}, {edges[i+1]:>10.2f})  {hist[i]:>10,}  {bar}")
        else:
            print(f"  [{edges[i]:>10.2f}, {edges[i+1]:>10.2f})           0  ")

    above_cap = (combined > cap).sum()
    print(f"  (above 99.9% quantile {cap:.1f}): {above_cap:,} observations {above_cap/len(combined)*100:.4f}%)")


if __name__ == "__main__":
    main()
