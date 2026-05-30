"""Discrete-time survival NLL and hybrid loss with privileged ``t_event`` supervision.

Provides:

- :func:`discrete_survival_nll`: numerically stable negative log-likelihood
  for discrete-time survival data ``(t_obs, delta)``.
- :func:`censored_pinball_loss`: pinball loss on ``t_event`` for censored
  observations only, using quantiles extracted from the discrete survival CDF.
- :class:`HybridSurvivalLoss`: combines both with a cosine-decaying weight
  ``α(step)`` on the imputation term.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

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

    delta_f = delta.float()
    nll = delta_f * nll_event + (1.0 - delta_f) * nll_cens

    return nll.mean()


# ---------------------------------------------------------------------------
# Censored-only pinball imputation loss
# ---------------------------------------------------------------------------


def censored_pinball_loss(
    h_raw: Tensor,
    t_event: Tensor,
    delta: Tensor,
    binner: TimeBinner,
    tau_levels: Optional[Tensor] = None,
) -> Tensor:
    """Pinball (quantile) loss on ``t_event`` for **censored observations only**.

    For each quantile level τ ∈ ``tau_levels``, the discrete survival CDF
    is interpolated to obtain ``Q(τ)``, and the pinball loss
    ``ρ_τ(t_event - Q(τ))`` is computed.  The result is averaged over all
    censored observations and all τ levels.  If no censored observations
    exist in the batch, returns 0 (scalar on the correct device).

    ``t_event`` values exceeding the last bin boundary are clipped to ``τ_K``
    before the loss is computed (per design: only the imputation target is
    capped; ``t_obs`` and ``delta`` are used unmodified).

    Parameters
    ----------
    h_raw : Tensor, shape ``(N, K)``
        Raw hazard logits.

    t_event : Tensor, shape ``(N,)``
        Ground-truth counterfactual event time.

    delta : Tensor, shape ``(N,)``
        Event indicator.  Loss is only applied where ``delta == 0``.

    binner : TimeBinner
        Provides ``quantile_at()`` for extracting quantiles from the CDF.

    tau_levels : Tensor, optional, shape ``(Q,)``
        Quantile probability levels.  Default: ``[0.1, 0.2, ..., 0.9]``.

    Returns
    -------
    Tensor
        Scalar loss, averaged over censored observations and τ levels.
        Returns 0.0 if there are no censored observations.
    """
    if tau_levels is None:
        tau_levels = torch.tensor(_DEFAULT_TAU_LEVELS, device=h_raw.device, dtype=h_raw.dtype)

    mask = (delta == 0).float()
    n_censored = mask.sum()

    if n_censored == 0:
        return torch.tensor(0.0, device=h_raw.device, dtype=h_raw.dtype)

    # Clip t_event to τ_K for the imputation loss
    t_max = binner.bin_edges[-1].to(device=h_raw.device, dtype=h_raw.dtype)
    t_event_clipped = torch.clamp(t_event, max=t_max)

    # Extract quantiles Q(τ) for all observations at all τ levels
    quantiles = binner.quantile_at(h_raw, tau_levels)  # (N, Q)

    # Pinball loss per τ level
    # diff shape: (N, Q) — broadcasting t_event against quantiles
    diff = t_event_clipped.unsqueeze(-1) - quantiles  # (N, Q)
    tau = tau_levels.view(1, -1)  # (1, Q)
    pinball = torch.maximum(tau * diff, (tau - 1.0) * diff)  # (N, Q)

    # Average over quantile levels, then mask by censored
    per_sample = pinball.mean(dim=-1)  # (N,)
    return (per_sample * mask).sum() / n_censored


# ---------------------------------------------------------------------------
# Hybrid loss
# ---------------------------------------------------------------------------


class HybridSurvivalLoss:
    """Combined survival NLL + imputation loss with cosine-decay weight.

    .. math::
        L = L_{surv}(h_raw, t_obs, delta) + α(step) · L_{impute}(h_raw, t_event, delta)

    where ``α(step)`` follows a cosine schedule from ``alpha_start`` to
    ``alpha_floor`` over ``max_steps``.

    Parameters
    ----------
    alpha_start : float, default=3.0
        Initial weight on the imputation loss at step 0.

    alpha_floor : float, default=0.05
        Minimum weight on the imputation loss after fully decaying.

    max_steps : int, default=10_000
        Total training steps over which to decay.
    """

    def __init__(
        self,
        alpha_start: float = 3.0,
        alpha_floor: float = 0.05,
        max_steps: int = 10_000,
    ) -> None:
        self.alpha_start = alpha_start
        self.alpha_floor = alpha_floor
        self.max_steps = max_steps

    def alpha(self, step: int) -> float:
        """Cosine-decayed imputation weight at a given step."""
        if step >= self.max_steps:
            return self.alpha_floor
        import math

        progress = step / self.max_steps
        cos_val = math.cos(math.pi * progress)
        return self.alpha_floor + (self.alpha_start - self.alpha_floor) * 0.5 * (1.0 + cos_val)

    def __call__(
        self,
        h_raw: Tensor,
        t_obs: Tensor,
        delta: Tensor,
        t_event: Tensor,
        binner: TimeBinner,
        step: int,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute the total hybrid loss.

        Parameters
        ----------
        h_raw : Tensor, shape ``(N, K)``
            Raw hazard logits from the survival head.

        t_obs : Tensor, shape ``(N,)``
            Observed time = min(event_time, censoring_time).

        delta : Tensor, shape ``(N,)``
            Event indicator: 1 = event, 0 = censored.

        t_event : Tensor, shape ``(N,)``
            Counterfactual event time (available for ALL observations in
            synthetic data).  Only used for censored observations via the
            imputation term.

        binner : TimeBinner
            Bin boundaries and representative times.

        step : int
            Current training step (0-indexed) for α scheduling.

        Returns
        -------
        total : Tensor
            Scalar ``L_surv + α(step) · L_impute``.

        breakdown : dict
            ``{"surv_nll": float, "impute": float, "alpha": float}``.
        """
        # Map t_obs to bins
        bin_idx = binner.bin_index(t_obs)  # (N,), 0-indexed

        # Survival NLL
        surv_loss = discrete_survival_nll(h_raw, bin_idx, delta)

        # Imputation loss (censored only)
        impute_loss = censored_pinball_loss(h_raw, t_event, delta, binner)

        a = self.alpha(step)
        total = surv_loss + a * impute_loss

        breakdown = {
            "surv_nll": surv_loss.item(),
            "impute": impute_loss.item(),
            "alpha": a,
        }
        return total, breakdown
