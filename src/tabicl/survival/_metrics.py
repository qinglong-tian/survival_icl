"""Evaluation metrics for fixed synthetic survival tasks."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from tabicl.survival._head import TimeBinner
from tabicl.survival._loss import discrete_survival_nll


def harrell_c_index(event_time: Tensor, risk: Tensor) -> float:
    """Harrell concordance for uncensored oracle event times."""
    event_time = event_time.detach().float().cpu()
    risk = risk.detach().float().cpu()
    left, right = torch.triu_indices(len(event_time), len(event_time), offset=1)
    comparable = event_time[left] != event_time[right]
    if not comparable.any():
        return float("nan")
    earlier_left = event_time[left] < event_time[right]
    risk_diff = risk[left] - risk[right]
    concordant = torch.where(earlier_left, risk_diff > 0, risk_diff < 0).float()
    ties = (risk_diff == 0).float() * 0.5
    return float((concordant[comparable] + ties[comparable]).mean())


def oracle_integrated_brier(
    survival: Tensor,
    event_time_z: Tensor,
    time_grid_z: Tensor,
) -> float:
    """Integrated Brier score against uncensored oracle event times."""
    targets = (event_time_z[:, None] > time_grid_z[None, :]).float()
    brier = ((survival.float() - targets) ** 2).mean(dim=0)
    width = float(time_grid_z[-1] - time_grid_z[0])
    if width <= 0:
        return float("nan")
    return float(torch.trapz(brier, time_grid_z.float()) / width)


def task_metrics(
    hazard_logits: Tensor,
    survival: Tensor,
    quantiles_z: Tensor,
    quantile_levels: Tensor,
    t_obs: Tensor,
    delta: Tensor,
    t_event: Tensor,
    scaler,
    binner: TimeBinner,
) -> dict[str, float]:
    """Compute all evaluation metrics for one synthetic query set."""
    event_z, in_range = scaler.transform_event_target(t_event.float())
    observed_z, observed_delta = scaler.transform_observed(t_obs.float(), delta.float())
    event_bins = binner.bin_index(event_z)
    observed_bins = binner.bin_index(observed_z)
    oracle_nll = discrete_survival_nll(
        hazard_logits.float(), event_bins, torch.ones_like(event_z),
    )
    observed_nll = discrete_survival_nll(
        hazard_logits.float(), observed_bins, observed_delta,
    )
    median_idx = int(torch.argmin(torch.abs(quantile_levels - 0.5)))
    c_index = harrell_c_index(event_z, -quantiles_z[:, median_idx])
    ibs = oracle_integrated_brier(survival, event_z, binner.bin_edges.cpu())

    eligible = in_range.bool()
    if eligible.any():
        diff = event_z[eligible, None] - quantiles_z[eligible]
        tau = quantile_levels.view(1, -1)
        pinball = torch.maximum(tau * diff, (tau - 1.0) * diff).mean()
        coverage = (event_z[eligible, None] <= quantiles_z[eligible]).float().mean(dim=0)
    else:
        pinball = torch.tensor(float("nan"))
        coverage = torch.full_like(quantile_levels, float("nan"))

    metrics = {
        "oracle_event_nll": float(oracle_nll),
        "observed_nll": float(observed_nll),
        "oracle_c_index": c_index,
        "oracle_ibs": ibs,
        "oracle_pinball": float(pinball),
        "event_in_horizon_fraction": float(eligible.float().mean()),
        "nonfinite_prediction_count": float(
            (~torch.isfinite(hazard_logits)).sum()
            + (~torch.isfinite(survival)).sum()
            + (~torch.isfinite(quantiles_z)).sum()
        ),
    }
    for idx, level in enumerate(quantile_levels.tolist()):
        metrics[f"coverage_{level:g}"] = float(coverage[idx])
    return metrics


def macro_means(rows: list[dict], metric_names: list[str]) -> dict[str, float]:
    return {
        metric: float(np.nanmean([row[metric] for row in rows]))
        for metric in metric_names
    }


def group_bootstrap_ci(
    rows: list[dict],
    metric: str,
    *,
    samples: int = 2000,
    seed: int = 20260605,
) -> list[float]:
    """Bootstrap a macro metric by resampling related four-task GP groups."""
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["group_id"]), []).append(float(row[metric]))
    group_means = np.asarray(
        [np.nanmean(values) for values in grouped.values()], dtype=float,
    )
    group_means = group_means[np.isfinite(group_means)]
    if len(group_means) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = rng.choice(group_means, size=(samples, len(group_means)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def paired_group_bootstrap_ci(
    rows: list[dict],
    metric: str,
    *,
    checkpoint_key: str,
    baseline_key: str,
    samples: int = 2000,
    seed: int = 20260605,
) -> list[float]:
    """Bootstrap paired checkpoint-minus-baseline group-level metric differences."""
    grouped: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        grouped.setdefault(str(row["checkpoint"]), {}).setdefault(
            int(row["group_id"]), []
        ).append(float(row[metric]))
    current = {
        group: float(np.nanmean(values))
        for group, values in grouped.get(checkpoint_key, {}).items()
    }
    baseline = {
        group: float(np.nanmean(values))
        for group, values in grouped.get(baseline_key, {}).items()
    }
    differences = np.asarray([
        current[group] - baseline[group]
        for group in sorted(current.keys() & baseline.keys())
        if math.isfinite(current[group]) and math.isfinite(baseline[group])
    ])
    if len(differences) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        differences, size=(samples, len(differences)), replace=True,
    ).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def all_metrics_finite(row: dict, metric_names: list[str]) -> bool:
    return all(math.isfinite(float(row[name])) for name in metric_names)
