"""Discrete-time survival prediction head and time binning.

Provides:
- :class:`TimeBinner`: constructs quantile-based time bins, maps times to bins,
  and computes survival curves / CDF / expected times / quantiles from hazard
  logits.
- :class:`DiscreteTimeSurvivalHead`: a drop-in replacement for TabICL's decoder
  MLP that outputs K raw hazard logits (one per time bin).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# TimeBinner
# ---------------------------------------------------------------------------


class TimeBinner:
    """Convert continuous times into discrete bins and compute survival quantities.

    Parameters
    ----------
    bin_edges : Tensor, shape ``(K+1,)``
        Bin boundaries ``τ_0 < τ_1 < ... < τ_K``.  ``τ_0`` is typically 0.

    bin_means : Tensor, shape ``(K,)``
        Representative time ``m_k`` for each bin (conditional mean of event
        times observed in that bin).  Used when computing expected event time.
    """

    def __init__(self, bin_edges: Tensor, bin_means: Tensor) -> None:
        if bin_edges.dim() != 1:
            raise ValueError(f"bin_edges must be 1D, got shape {bin_edges.shape}")
        if bin_means.dim() != 1 or bin_means.shape[0] != len(bin_edges) - 1:
            raise ValueError(
                f"bin_means must be 1D of length K={len(bin_edges) - 1}, "
                f"got shape {bin_means.shape}"
            )
        self.bin_edges = bin_edges  # (K+1,)
        self.bin_means = bin_means  # (K,)
        self.num_bins = len(bin_means)

    # --- factory ----------------------------------------------------------

    @classmethod
    def from_event_times(
        cls,
        t_event: Tensor,
        num_bins: int = 50,
        *,
        headroom: float = 0.05,
    ) -> "TimeBinner":
        """Build quantile-based bins from a sample of event times.

        Bin boundaries are placed at equal quantile levels of the event time
        distribution so every bin contains roughly the same number of events.
        The representative time per bin is the conditional mean of event times
        that fall in that bin.

        Parameters
        ----------
        t_event : Tensor
            Event times (uncensored), any shape — flattened internally.
            For synthetic data this is the counterfactual ``t_event``.

        num_bins : int, default=50
            Number of time bins ``K``.

        headroom : float, default=0.05
            Fractional padding added beyond ``max(t_event)`` for the last bin
            boundary so observations at the maximum time are not out-of-range.

        Returns
        -------
        TimeBinner
        """
        t = t_event.detach().cpu().float().reshape(-1)
        t = t[t.isfinite()]

        if t.numel() < num_bins:
            raise ValueError(
                f"Need at least {num_bins} finite event times, got {t.numel()}"
            )

        t_sorted = torch.sort(t).values
        t_max = t_sorted[-1].item()

        # Quantile-based interior edges
        q_levels = torch.linspace(0.0, 1.0, num_bins + 1, dtype=torch.float64)
        edges = torch.quantile(t_sorted.to(torch.float64), q_levels).float()

        # First edge is min(t), force to 0
        edges[0] = 0.0
        # Last edge: extend slightly beyond max so max time falls in last bin
        edges[-1] = t_max * (1.0 + headroom) + 1e-6

        # Compute conditional mean per bin
        bin_means = torch.zeros(num_bins, dtype=torch.float32)
        bin_indices = torch.searchsorted(edges, t_sorted, right=True) - 1
        bin_indices = bin_indices.clamp(0, num_bins - 1)

        for k in range(num_bins):
            mask = bin_indices == k
            if mask.any():
                bin_means[k] = t_sorted[mask].float().mean()
            else:
                # Should not happen with quantile bins, but fall back to midpoint
                bin_means[k] = (edges[k] + edges[k + 1]) * 0.5

        return cls(bin_edges=edges, bin_means=bin_means)

    @classmethod
    def from_standardized_range(
        cls,
        num_bins: int = 50,
        z_min: float = -6.0,
        z_max: float = 6.0,
    ) -> "TimeBinner":
        """Build fixed bins on the standardized log-time axis."""
        if num_bins < 1:
            raise ValueError(f"num_bins must be positive, got {num_bins}")
        if z_max <= z_min:
            raise ValueError(f"z_max must be greater than z_min, got {z_min}, {z_max}")

        edges = torch.linspace(z_min, z_max, num_bins + 1, dtype=torch.float32)
        bin_means = (edges[:-1] + edges[1:]) * 0.5
        return cls(bin_edges=edges, bin_means=bin_means)

    # --- bin assignment ---------------------------------------------------

    def bin_index(self, t: Tensor) -> Tensor:
        """Map continuous times to 0-indexed bin indices.

        Times ``<= τ_0`` map to bin 0; times ``> τ_K`` are clamped to bin K-1.

        Parameters
        ----------
        t : Tensor, any shape

        Returns
        -------
        Tensor, same shape as ``t``, dtype ``torch.long``, values in ``[0, K-1]``.
        """
        edges = self.bin_edges.to(device=t.device)
        idx = torch.searchsorted(edges, t)  # values in [0, K]
        return (idx - 1).clamp(min=0, max=self.num_bins - 1)

    # --- survival quantities from hazard logits ---------------------------

    def hazard_probs(self, h_raw: Tensor) -> Tensor:
        """Convert raw logits to conditional hazard probabilities.

        Parameters
        ----------
        h_raw : Tensor, shape ``(..., K)``

        Returns
        -------
        Tensor, shape ``(..., K)``, values in ``(0, 1)``.
        """
        return torch.sigmoid(h_raw)

    def log_survival(self, h_raw: Tensor) -> Tensor:
        """Cumulative log survival ``log S(τ_k)`` for each bin edge.

        ``log S(τ_k) = Σ_{j=1}^{k} log(1 - h_j)``

        Parameters
        ----------
        h_raw : Tensor, shape ``(..., K)``

        Returns
        -------
        Tensor, shape ``(..., K+1)``, where column 0 is 0 (S(0) = 1).
        """
        log_1mh = nn.functional.logsigmoid(-h_raw)  # log(1 - σ(x))
        # Pad at the front so index k gives S(τ_k) for k=0..K
        zeros = torch.zeros(*log_1mh.shape[:-1], 1, device=h_raw.device, dtype=h_raw.dtype)
        return torch.cat([zeros, torch.cumsum(log_1mh, dim=-1)], dim=-1)

    def survival(self, h_raw: Tensor) -> Tensor:
        """Survival function ``S(τ_k)`` at each bin edge.

        Parameters
        ----------
        h_raw : Tensor, shape ``(..., K)``

        Returns
        -------
        Tensor, shape ``(..., K+1)``, values in ``[0, 1]``, with S(τ_0)=1.
        """
        return torch.exp(self.log_survival(h_raw))

    def cdf(self, h_raw: Tensor) -> Tensor:
        """CDF ``F(τ_k) = 1 - S(τ_k)`` at each bin edge.

        Returns
        -------
        Tensor, shape ``(..., K+1)``.
        """
        return 1.0 - self.survival(h_raw)

    def event_prob_mass(self, h_raw: Tensor) -> Tensor:
        """Probability of the event occurring in each bin.

        ``p_k = h_k · S(τ_{k-1})``

        Parameters
        ----------
        h_raw : Tensor, shape ``(..., K)``

        Returns
        -------
        Tensor, shape ``(..., K)``, sums to ``≤ 1`` across the last dim.
        """
        h = self.hazard_probs(h_raw)
        log_S = self.log_survival(h_raw)  # (..., K+1)
        # S(τ_{k-1}) for each bin k: columns 0..K-1 of log_S
        S_km1 = torch.exp(log_S[..., : self.num_bins])
        return h * S_km1

    def standardized_time_summary(self, h_raw: Tensor) -> Tensor:
        """Capped standardized-time summary with residual tail mass at ``z_max``.

        Computes ``E[min(Z, z_max)] = Σ_k m_k · p_k + z_max · S(τ_K)``
        where ``S(τ_K)`` is the probability mass beyond the last bin edge
        (residual survival).  This is a proper restricted mean on the
        standardized log-time axis, not a raw event time.

        For raw-time quantities, apply each task's ``SurvivalTimeScaler.inverse_time``.

        Parameters
        ----------
        h_raw : Tensor, shape ``(..., K)``

        Returns
        -------
        Tensor, shape ``(...)`` (one fewer dim than input), in standardized
        log-time units.
        """
        p = self.event_prob_mass(h_raw.float())  # (..., K)
        means = self.bin_means.to(device=h_raw.device, dtype=torch.float32)  # (K,)
        e_trunc = (p * means).sum(dim=-1)  # (...)
        # Add residual tail: z_max times probability mass beyond τ_K
        log_S = self.log_survival(h_raw.float())
        S_K = torch.exp(log_S[..., -1])  # (...), S(τ_K)
        z_max = self.bin_edges[-1].to(device=h_raw.device, dtype=torch.float32)
        return (e_trunc + z_max * S_K).to(dtype=h_raw.dtype)

    def expected_time(self, h_raw: Tensor) -> Tensor:
        """Deprecated alias for :meth:`standardized_time_summary`.

        The name is misleading because the returned values are standardized
        log-times, not raw event times.  Use :meth:`standardized_time_summary`
        directly in new code.
        """
        import warnings

        warnings.warn(
            "expected_time is deprecated — use standardized_time_summary for "
            "capped standardized log-times.  For raw-time quantiles, apply "
            "SurvivalTimeScaler.inverse_time.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.standardized_time_summary(h_raw)

    def quantile_at(self, h_raw: Tensor, probs: Tensor) -> Tensor:
        """Compute specified quantiles via linear interpolation of the CDF.

        All internal computations are performed in float32 for numerical
        stability.  Quantiles not reached by the finite-horizon CDF return
        ``bin_edges[-1]`` (i.e. ``z_max``).

        No GPU synchronizations occur after the initial validation.

        Parameters
        ----------
        h_raw : Tensor, shape ``(..., K)``

        probs : Tensor, shape ``(Q,)``
            Probability levels.  All entries must be finite and in (0, 1).

        Returns
        -------
        Tensor, shape ``(..., Q)`` — quantile time for each requested level,
        always inside ``[bin_edges[0], bin_edges[-1]]``.
        """
        if probs.ndim != 1:
            raise ValueError(f"probs must be 1D (Q,), got ndim={probs.ndim}")
        if not torch.isfinite(probs).all():
            raise ValueError("probs must be finite")
        if not ((probs > 0) & (probs < 1)).all():
            raise ValueError("probs must be in (0, 1)")

        # Force float32 for CDF to avoid inf/nan in float16
        F = self.cdf(h_raw.float())  # (..., K+1), float32
        edges = self.bin_edges.to(device=h_raw.device, dtype=torch.float32)  # (K+1,)
        z_max = edges[-1]  # scalar, float32
        p = probs.to(device=h_raw.device, dtype=torch.float32)  # (Q,)

        # Vectorize across all probabilities: expand F for broadcast
        # F: (..., K+1) -> (..., K+1, 1) broadcasts with p: (Q,) -> (..., K+1, Q)
        F_exp = F.unsqueeze(-1)  # (..., K+1, 1)

        # First bin where F >= p for each probability
        above = F_exp >= p  # (..., K+1, Q)
        any_above = above.any(dim=-2)  # (..., Q)

        k = above.float().argmax(dim=-2)  # (..., Q)
        k = torch.where(any_above, k, torch.full_like(k, self.num_bins))
        k = k.clamp(min=1, max=self.num_bins)  # (..., Q)

        # Gather F_lo, F_hi along the K+1 dimension for each probability.
        km1_idx = (k - 1).unsqueeze(-2)  # (..., 1, Q)
        k_idx = k.unsqueeze(-2)  # (..., 1, Q)
        F_expanded = F_exp.expand(*F_exp.shape[:-1], k.shape[-1])  # (..., K+1, Q)
        F_lo = F_expanded.gather(-2, km1_idx).squeeze(-2)  # (..., Q)
        F_hi = F_expanded.gather(-2, k_idx).squeeze(-2)  # (..., Q)

        t_lo = edges[(k - 1).clamp(min=0)]  # (..., Q)
        t_hi = edges[k.clamp(max=self.num_bins)]  # (..., Q)

        # Unreached quantile: return z_max
        q = torch.where(any_above, torch.zeros_like(t_lo), torch.full_like(t_lo, z_max))

        # Interpolate unconditionally; select with torch.where
        denom = (F_hi - F_lo).clamp(min=1e-10)
        frac = ((p - F_lo) / denom).clamp(0.0, 1.0)
        q_interp = t_lo + frac * (t_hi - t_lo)
        q = torch.where(any_above, q_interp, q)
        q = q.clamp(edges[0], z_max)

        return q.to(dtype=h_raw.dtype)  # (..., Q)

    def to(self, device: torch.device) -> "TimeBinner":
        """Return a copy with tensors moved to ``device``."""
        return TimeBinner(
            bin_edges=self.bin_edges.to(device),
            bin_means=self.bin_means.to(device),
        )

    def __repr__(self) -> str:
        return (
            f"TimeBinner(num_bins={self.num_bins}, "
            f"t_range=[{self.bin_edges[0].item():.2f}, {self.bin_edges[-1].item():.2f}])"
        )


# ---------------------------------------------------------------------------
# DiscreteTimeSurvivalHead
# ---------------------------------------------------------------------------


class DiscreteTimeSurvivalHead(nn.Module):
    """Drop-in replacement for TabICL's decoder MLP, outputting raw hazard logits.

    Architecture matches the upstream ``ICLearning.decoder`` pattern::

        Linear(d_model, 2*d_model) → GELU → Linear(2*d_model, num_bins)

    The output is a raw score per time bin (no activation applied).  Sigmoid
    is applied downstream via :class:`TimeBinner.hazard_probs`.

    Parameters
    ----------
    d_model : int
        Input dimension from the ICLearning transformer (``embed_dim × row_num_cls``).

    num_bins : int, default=50
        Number of time bins ``K``.
    """

    def __init__(self, d_model: int, num_bins: int = 50) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_bins = num_bins
        self.head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, num_bins),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Project transformer output to raw hazard logits.

        Parameters
        ----------
        z : Tensor, shape ``(..., d_model)``
            Transformer output (ICLearning's penultimate representation).

        Returns
        -------
        Tensor, shape ``(..., num_bins)`` — raw logits ``h_raw``.
        """
        return self.head(z)
