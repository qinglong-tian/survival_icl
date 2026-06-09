"""Discrete-time survival NLL and oracle query-event pinball loss.

Provides:

- :func:`discrete_survival_nll`: numerically stable negative log-likelihood
  for discrete-time survival data ``(t_obs, delta)``.
- :func:`oracle_query_pinball_loss`: pinball loss on standardized oracle query
  event times for all valid query rows, using quantiles extracted from the
  discrete survival CDF.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from tabicl.survival._head import TimeBinner

# Default quantile levels for imputation pinball loss (9 deciles).
_DEFAULT_TAU_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ---------------------------------------------------------------------------
# Discrete survival NLL
# ---------------------------------------------------------------------------


def discrete_survival_nll(
    h_raw: Tensor,
    bin_idx: Tensor,
    delta: Tensor,
) -> Tensor:
    """Discrete-time survival negative log-likelihood (scalar, mean over obs).

    For observation ``i`` at bin ``k_i``:

    - If ``delta_i = 1`` (event): ``L_i = -[log(h_{i,k}) + Σ_{j<k} log(1 - h_{i,j})]``
    - If ``delta_i = 0`` (censored): ``L_i = -Σ_{j≤k} log(1 - h_{i,j})``

    Computed in logit-space for numerical stability.

    Parameters
    ----------
    h_raw : Tensor, shape ``(N, K)``
        Raw hazard logits per observation (before sigmoid).

    bin_idx : Tensor, shape ``(N,)``, dtype ``long``
        0-indexed bin index for each observation's ``t_obs``, values in ``[0, K-1]``.

    delta : Tensor, shape ``(N,)``
        Event indicator: 1 = event observed, 0 = censored.

    Returns
    -------
    Tensor
        Scalar loss (mean over all ``N`` observations).
    """
    K = h_raw.shape[-1]
    device = h_raw.device

    # log(1 - σ(x)) = -softplus(x)
    # log σ(x) = -softplus(-x)
    log_h = F.logsigmoid(h_raw)  # (N, K): log(h_k) per bin
    log_1mh = F.logsigmoid(-h_raw)  # (N, K): log(1 - h_k) per bin

    # Cumulative log survival: log S(τ_k) = Σ_{j=1}^{k} log(1 - h_j)
    # Pad with a zero column at the front for k=0: S(τ_0) = 1 → log S(τ_0) = 0
    zeros = torch.zeros(h_raw.shape[0], 1, device=device, dtype=h_raw.dtype)
    log_S_cum = torch.cat([zeros, torch.cumsum(log_1mh, dim=-1)], dim=-1)  # (N, K+1)

    # log S up to bin k (inclusive): columns 1..K of log_S_cum → index k
    # log S up to bin k-1: columns 0..K-1 of log_S_cum → index k
    # Gather at each observation's bin index
    idx = bin_idx.clamp(0, K - 1)  # (N,)

    # log S(τ_{k_i}): column k_i of log_S (which is index k_i+1 in log_S_cum)
    log_S_k = log_S_cum[torch.arange(log_S_cum.shape[0], device=device), idx + 1]

    # log S(τ_{k_i-1}): column k_i of log_S (which is index k_i in log_S_cum)
    log_S_km1 = log_S_cum[torch.arange(log_S_cum.shape[0], device=device), idx]

    # log h_{k_i}
    log_h_k = log_h[torch.arange(log_h.shape[0], device=device), idx]

    # Per-observation NLL
    # Event:   NLL = -log_h_k - log_S_km1
    # Censored: NLL = -log_S_k = -(log_1mh_k + log_S_km1) = -log_S_k
    nll_event = -(log_h_k + log_S_km1)
    nll_cens = -log_S_k

    is_event = delta > 0.5
    nll = torch.where(is_event, nll_event, nll_cens)

    return nll.mean()


# ---------------------------------------------------------------------------
# Oracle query-event pinball loss
# ---------------------------------------------------------------------------


def oracle_query_pinball_loss(
    h_raw: Tensor,
    t_event_z: Tensor,
    binner: TimeBinner,
    in_range: Tensor,
    valid_mask: Tensor | None = None,
    tau_levels: Tensor | None = None,
) -> Tensor:
    """Pinball loss on oracle query event times for ALL valid query rows.

    This function:

    - Applies to **every** valid query row (all are unconditional events).
    - Uses unclipped ``t_event_z`` from :meth:`SurvivalTimeScaler.transform_event_target`.
    - Excludes targets where ``in_range`` is ``False`` (outside ``[z_min, z_max]``)
      from pinball because predicted quantiles are capped at the horizon.
    - Does NOT clip targets — out-of-range rows are excluded, not repaired.

    Parameters
    ----------
    h_raw : Tensor, shape ``(N, K)``
        Raw hazard logits per observation.

    t_event_z : Tensor, shape ``(N,)``
        Unclipped standardized log-times ``raw_z(t_event)`` from the
        context-fitted scaler.  May contain values outside ``[z_min, z_max]``.

    binner : TimeBinner
        Provides ``quantile_at()`` for extracting predicted quantiles from
        the discrete survival CDF.

    in_range : Tensor, shape ``(N,)``, bool
        ``z_min <= t_event_z <= z_max``.  Produced by
        ``SurvivalTimeScaler.transform_event_target``.

    valid_mask : Tensor, optional, shape ``(N,)``, bool
        Padding mask.  Row ``i`` is evaluated only when ``valid_mask[i]`` is
        ``True``.  If ``None``, all rows are assumed valid.

    tau_levels : Tensor, optional, shape ``(Q,)``
        Quantile probability levels.  Default: ``[0.1, 0.2, ..., 0.9]``.

    Returns
    -------
    Tensor
        Scalar pinball loss, averaged over selected τ levels and valid
        in-range rows.  Returns a differentiable zero **on the correct
        device** when there are no valid in-range rows.
    """
    if tau_levels is None:
        tau_levels = torch.tensor(_DEFAULT_TAU_LEVELS, device=h_raw.device, dtype=h_raw.dtype)

    if valid_mask is not None:
        pinball_mask = valid_mask.bool() & in_range
    else:
        pinball_mask = in_range

    n_eligible = pinball_mask.sum()

    if n_eligible == 0:
        # Return a differentiable zero on the same device/dtype as h_raw
        # so that the computation graph is preserved.
        return (h_raw.sum() * 0.0).to(dtype=h_raw.dtype)

    # Extract quantiles Q(τ) for the eligible observations
    h_eligible = h_raw[pinball_mask]          # (N_eligible, K)
    t_eligible = t_event_z[pinball_mask]      # (N_eligible,)

    quantiles = binner.quantile_at(h_eligible, tau_levels)  # (N_eligible, Q)

    # Pinball loss: ρ_τ(t - Q(τ)) = max(τ·(t - Q), (τ-1)·(t - Q))
    diff = t_eligible.unsqueeze(-1) - quantiles  # (N_eligible, Q)
    tau = tau_levels.view(1, -1)                  # (1, Q)
    pinball = torch.maximum(tau * diff, (tau - 1.0) * diff)  # (N_eligible, Q)

    # Mean over quantile levels, then mean over eligible observations
    return pinball.mean()

