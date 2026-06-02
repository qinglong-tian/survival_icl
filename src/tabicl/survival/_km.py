"""Kaplan-Meier utilities for survival time preprocessing."""

from __future__ import annotations

import torch
from torch import Tensor


def km_quantiles(log_y: Tensor, delta: Tensor, probs=(0.25, 0.5, 0.75)) -> Tensor:
    """Estimate event-time quantiles from right-censored log observed times.

    Parameters
    ----------
    log_y : Tensor, shape ``(n,)``
        Log observed survival times.

    delta : Tensor, shape ``(n,)``
        Event indicators where 1 means observed event and 0 means censored.

    probs : iterable of float, default=(0.25, 0.5, 0.75)
        CDF probability levels for requested quantiles.

    Returns
    -------
    Tensor
        Quantiles on the log-time scale.  Unavailable quantiles are ``nan``.
    """
    log_y = log_y.detach().float().reshape(-1)
    delta = delta.detach().float().reshape(-1)
    if log_y.shape != delta.shape:
        raise ValueError(f"log_y and delta must have the same shape, got {log_y.shape} and {delta.shape}")

    valid = torch.isfinite(log_y) & torch.isfinite(delta)
    log_y = log_y[valid]
    delta = delta[valid]
    if log_y.numel() == 0:
        return torch.full((len(tuple(probs)),), torch.nan, dtype=torch.float32, device=log_y.device)

    order = torch.argsort(log_y)
    times = log_y[order]
    events = (delta[order] > 0.5).float()
    _, counts = torch.unique_consecutive(times, return_counts=True)
    event_counts = torch.stack([chunk.sum() for chunk in torch.split(events, counts.tolist())])
    unique_times = times[torch.cumsum(counts, dim=0) - 1]

    targets = 1.0 - torch.tensor(tuple(probs), dtype=torch.float32, device=times.device)
    out = torch.full((targets.numel(),), torch.nan, dtype=torch.float32, device=times.device)

    survival = torch.tensor(1.0, dtype=torch.float32, device=times.device)
    at_risk = torch.tensor(float(times.numel()), dtype=torch.float32, device=times.device)

    for t, n_at_time, d_at_time in zip(unique_times, counts.float(), event_counts):
        if at_risk > 0 and d_at_time > 0:
            survival = survival * (1.0 - d_at_time / at_risk)
            newly_reached = torch.isnan(out) & (survival <= targets)
            out = torch.where(newly_reached, t.to(dtype=torch.float32), out)
        at_risk = at_risk - n_at_time

    return out
