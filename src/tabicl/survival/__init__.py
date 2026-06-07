"""Survival prediction head, discrete NLL loss, and inference helpers for TabICL.

Provides per-task log-time scaling, a discrete-time survival prediction head
(DiscreteTimeSurvivalHead and TimeBinner), a discrete survival negative
log-likelihood (NLL), and a batched context-scaling helper for inference.

Training uses NLL only.  Inference fits one SurvivalTimeScaler per support
dataset via ``scale_survival_context``, passes standardized ``(z, delta)`` to
the model, and interprets hazard logits via ``TimeBinner``.

Usage sketch::

    from tabicl.survival import scale_survival_context, TimeBinner, discrete_survival_nll

    # Context-only KM-hybrid scaling (always float32)
    z_context, delta_adj, scalers = scale_survival_context(
        t_context.float(), delta_context.float()
    )
    binner = TimeBinner.from_standardized_range(num_bins=50)

    # Model returns raw hazard logits over standardized bins
    h_raw = model(X, y_train=z_context, delta_train=delta_adj)
    loss = discrete_survival_nll(h_raw, bin_idx, delta_query)
"""

from __future__ import annotations

from tabicl.survival._head import TimeBinner, DiscreteTimeSurvivalHead
from tabicl.survival._km import km_quantiles
from tabicl.survival._loss import (
    HybridSurvivalLoss,
    censored_pinball_loss,
    discrete_survival_nll,
    oracle_query_pinball_loss,
)
from tabicl.survival._scaler import SurvivalTimeScaler, standardize_survival_micro_batch
from tabicl.survival._checkpoint import load_survival_checkpoint, validate_survival_metadata
from tabicl.survival._inference import SurvivalPrediction, TabICLSurvivalPredictor
from tabicl.survival._sklearn import TabICLSurvivalEstimator

__all__ = [
    "TimeBinner",
    "DiscreteTimeSurvivalHead",
    "HybridSurvivalLoss",
    "censored_pinball_loss",
    "discrete_survival_nll",
    "oracle_query_pinball_loss",
    "km_quantiles",
    "SurvivalTimeScaler",
    "standardize_survival_micro_batch",
    "scale_survival_context",
    "load_survival_checkpoint",
    "validate_survival_metadata",
    "SurvivalPrediction",
    "TabICLSurvivalPredictor",
    "TabICLSurvivalEstimator",
]


def scale_survival_context(
    t_context,
    delta_context,
    *,
    eps=1e-8,
    min_scale=0.1,
    z_min=-6.0,
    z_max=6.0,
):
    """Fit one per-dataset scaler and return standardized context times.

    Parameters
    ----------
    t_context : Tensor, shape ``(B, train_size)``
        Context observed times per dataset.
    delta_context : Tensor, shape ``(B, train_size)``
        Context event indicators per dataset.
    eps, min_scale, z_min, z_max :
        Passed to :class:`SurvivalTimeScaler`.

    Returns
    -------
    z_context : Tensor, shape ``(B, train_size)``
        Standardized log-times for context.
    delta_context_out : Tensor, shape ``(B, train_size)``, float32
        Adjusted event indicators (administratively censored above ``z_max``).
    scalers : list of SurvivalTimeScaler
        One fitted scaler per batch element, for interpreting predictions.
    """
    import math as _math
    import torch

    t_context = t_context.float()
    delta_context = delta_context.float()

    if t_context.dim() != 2 or delta_context.dim() != 2:
        raise ValueError(
            f"t_context and delta_context must be 2D (B, train_size), "
            f"got shapes {tuple(t_context.shape)} and {tuple(delta_context.shape)}. "
            f"For a single dataset, unsqueeze(0) to add a batch dimension."
        )
    if t_context.shape != delta_context.shape:
        raise ValueError(
            f"t_context and delta_context must have the same shape, "
            f"got {tuple(t_context.shape)} vs {tuple(delta_context.shape)}"
        )
    if not (_math.isfinite(eps) and eps > 0):
        raise ValueError(f"eps must be finite and > 0, got {eps}")
    if not (_math.isfinite(min_scale) and min_scale > 0):
        raise ValueError(f"min_scale must be finite and > 0, got {min_scale}")
    if not (_math.isfinite(z_min) and _math.isfinite(z_max)):
        raise ValueError(f"z_min and z_max must be finite, got z_min={z_min}, z_max={z_max}")
    if z_min >= z_max:
        raise ValueError(f"z_min must be < z_max, got z_min={z_min}, z_max={z_max}")

    B = t_context.shape[0]
    z_context = torch.empty_like(t_context)
    delta_out = torch.empty_like(delta_context, dtype=torch.float32)
    scalers = []

    for b in range(B):
        scaler = SurvivalTimeScaler(eps=eps, min_scale=min_scale, z_min=z_min, z_max=z_max)
        scaler.fit(t_context[b], delta_context[b])
        z_b, d_b = scaler.transform_observed(t_context[b], delta_context[b])
        z_context[b] = z_b.float()
        delta_out[b] = d_b.float()
        scalers.append(scaler)

    return z_context, delta_out, scalers
