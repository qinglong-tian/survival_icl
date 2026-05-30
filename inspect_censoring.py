from __future__ import annotations

import time
import numpy as np
import torch

from survival_prior import SurvivalPriorDataset

torch.set_printoptions(precision=4, sci_mode=False)


def inspect_batch(dataset, model_type, label):
    print(f"\n{'='*72}")
    print(f"{'='*72}")
    print(f"{label}")
    print(f"{'='*72}")
    print(dataset)
    print()

    X, t, delta, d, seq_lens, train_sizes = dataset.get_batch()
    B = X.shape[0]

    t_np = t.cpu().numpy()
    delta_np = delta.cpu().numpy()

    for ds_idx in range(min(B, 8)):
        d_val = d[ds_idx].item()
        seq_len = seq_lens[ds_idx].item()
        train_size = train_sizes[ds_idx].item()
        t_ds = t_np[ds_idx]
        delta_ds = delta_np[ds_idx]
        event_rate = delta_ds.mean()
        n_events = delta_ds.sum()
        n_censored = seq_len - n_events

        print(f"Dataset {ds_idx:3d}:  N={seq_len:5d}  "
              f"train={train_size:5d}  features={d_val:3d}  "
              f"events={int(n_events):5d}  censored={int(n_censored):5d}  "
              f"event_rate={event_rate:.3f}  "
              f"t_mean={t_ds.mean():.3f}  t_std={t_ds.std():.3f}  "
              f"t_min={t_ds.min():.3f}  t_max={t_ds.max():.3f}")

    # Summary across batch
    event_rates = delta_np.mean(axis=1)
    print(f"\n--- Batch summary ({B} datasets) ---")
    print(f"  Event rates:    mean={event_rates.mean():.3f}  "
          f"std={event_rates.std():.3f}  "
          f"min={event_rates.min():.3f}  max={event_rates.max():.3f}")
    print(f"  t (all obs):    mean={t_np.mean():.3f}  "
          f"std={t_np.std():.3f}  "
          f"min={t_np.min():.6f}  max={t_np.max():.3f}")

    # Per-baseline breakdown
    print(f"  Event rate histogram:")
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(event_rates, bins=bins)
    for i in range(len(bins) - 1):
        bar = "█" * hist[i]
        print(f"    [{bins[i]:.1f}-{bins[i+1]:.1f}): {hist[i]:4d}  {bar}")

    # Fraction of datasets rejected (we can't directly observe this from
    # get_batch since it only returns valid ones, but we can inspect the
    # delta distribution to see how many are near the bounds)
    near_lower = (event_rates >= 0.50).sum()
    near_upper = (event_rates <= 0.97).sum()
    print(f"  within [0.40, 1.0]: {int(near_lower & near_upper)}/{B}")


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    # === PH: mixed baselines (Weibull + Gompertz + LogLogistic + LogNormal) ===
    ph = SurvivalPriorDataset(
        batch_size=32,
        max_seq_len=512,
        max_features=30,
        prior_type="mix_scm",
        model_type="ph",
        beta=1.0,
        baseline_types=["weibull", "gompertz", "loglogistic", "lognormal"],
        baseline_mode="mix",
        min_censor_scale=1.0,
        max_censor_scale=5.0,
        min_event_rate=0.40,
        max_event_rate=1.0,
        log_seq_len=True,
        device="cpu",
        n_jobs=4,
        min_train_size=100,
        max_train_size=400,
    )
    inspect_batch(ph, "ph", "PH — Mixed baselines (Weibull+Gompertz+LogLogistic+LogNormal)")

    # === PH: fixed Weibull only ===
    ph_wei = SurvivalPriorDataset(
        batch_size=32,
        max_seq_len=512,
        max_features=30,
        prior_type="mix_scm",
        model_type="ph",
        beta=1.0,
        baseline_types=["weibull"],
        baseline_mode="weibull",
        min_censor_scale=1.0,
        max_censor_scale=5.0,
        min_event_rate=0.40,
        max_event_rate=1.0,
        log_seq_len=True,
        device="cpu",
        n_jobs=4,
        min_train_size=100,
        max_train_size=400,
    )
    inspect_batch(ph_wei, "ph_weibull", "PH — Weibull only")

    # === PH: fixed LogNormal only ===
    ph_ln = SurvivalPriorDataset(
        batch_size=32,
        max_seq_len=512,
        max_features=30,
        prior_type="mix_scm",
        model_type="ph",
        beta=1.0,
        baseline_types=["lognormal"],
        baseline_mode="lognormal",
        min_censor_scale=1.0,
        max_censor_scale=5.0,
        min_event_rate=0.40,
        max_event_rate=1.0,
        log_seq_len=True,
        device="cpu",
        n_jobs=4,
        min_train_size=100,
        max_train_size=400,
    )
    inspect_batch(ph_ln, "ph_lognormal", "PH — LogNormal only")

    # === AFT: mixed baselines (Weibull + LogLogistic + LogNormal) ===
    aft = SurvivalPriorDataset(
        batch_size=32,
        max_seq_len=512,
        max_features=30,
        prior_type="mix_scm",
        model_type="aft",
        beta=1.0,
        baseline_types=["weibull", "loglogistic", "lognormal"],
        baseline_mode="mix",
        min_censor_scale=1.0,
        max_censor_scale=5.0,
        min_event_rate=0.40,
        max_event_rate=1.0,
        log_seq_len=True,
        device="cpu",
        n_jobs=4,
        min_train_size=100,
        max_train_size=400,
    )
    inspect_batch(aft, "aft", "AFT — Mixed baselines (Weibull+LogLogistic+LogNormal)")

    # === Speed benchmark ===
    print(f"\n{'='*72}")
    print("Speed benchmark: generating 200 PH datasets sequentially")
    print(f"{'='*72}")
    ph_bench = SurvivalPriorDataset(
        batch_size=200,
        max_seq_len=1024,
        max_features=30,
        prior_type="mix_scm",
        model_type="ph",
        baseline_types=["weibull", "gompertz", "loglogistic", "lognormal"],
        baseline_mode="mix",
        min_censor_scale=1.0,
        max_censor_scale=5.0,
        min_event_rate=0.40,
        max_event_rate=1.0,
        device="cpu",
        n_jobs=4,
        min_train_size=100,
        max_train_size=900,
    )
    start = time.time()
    X, t, delta, d, seq_lens, train_sizes = ph_bench.get_batch()
    elapsed = time.time() - start
    event_rates = delta.float().mean(dim=1)
    print(f"  {X.shape[0]} datasets generated in {elapsed:.2f}s "
          f"({X.shape[0]/elapsed:.0f} ds/s)")
    print(f"  Overall event rate: {event_rates.mean():.3f} ± {event_rates.std():.3f}")

    # Verify that delta actually varies (not all ones)
    unique_deltas = delta.unique().tolist()
    print(f"  delta values observed: {unique_deltas}")
    assert 0.0 in unique_deltas, "No censored observations generated!"
    assert 1.0 in unique_deltas, "No events generated!"
    print("  ✓ delta contains both 0s (censored) and 1s (events)")


if __name__ == "__main__":
    main()
