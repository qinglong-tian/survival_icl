"""Survival head and hybrid loss for TabICL fine-tuning with time-to-event data.

Provides per-task log-time scaling, a discrete-time survival prediction head, and
a hybrid loss function that combines the discrete survival negative log-likelihood
(on observables `(t_obs, delta)`) with a pinball imputation loss on the privileged
counterfactual event times `t_event` (available only at training time in synthetic
data).  The model is trained on standardized log-times; future inference code
should fit :class:`SurvivalTimeScaler` on the in-context observed outcomes
``(t_obs, delta)``, transform support labels before calling the model, interpret
hazards on the standardized ``TimeBinner``, and convert requested outputs back
with ``inverse_time``.

Usage sketch::

    from tabicl.survival import SurvivalTimeScaler, TimeBinner, DiscreteTimeSurvivalHead, HybridSurvivalLoss

    # Fit on context observed outcomes only, then use fixed standardized bins.
    scaler = SurvivalTimeScaler().fit(t_context, delta_context)
    z_context, delta_context = scaler.transform_observed(t_context, delta_context)
    binner = TimeBinner.from_standardized_range(num_bins=50)

    # Load a pretrained TabICL checkpoint, then swap the decoder head
    model = TabICL(max_classes=0)  # regression path
    model.load_state_dict(ckpt, strict=False)
    head = DiscreteTimeSurvivalHead(d_model=embed_dim * row_num_cls, num_bins=50)
    model.icl_predictor.decoder = head

    # Compute hybrid loss
    loss_fn = HybridSurvivalLoss(alpha_start=3.0, alpha_floor=0.05, max_steps=10_000)
    z = model(X, y_train)          # (B, T_test, d_model) — but we need the raw decoder
    h_raw = head(z)                # (B, T_test, K) — raw hazard logits
    loss, breakdown = loss_fn(h_raw, t_obs, delta, t_event, binner=binner, step=cur_step)
"""

from __future__ import annotations

from tabicl.survival._head import TimeBinner, DiscreteTimeSurvivalHead
from tabicl.survival._km import km_quantiles
from tabicl.survival._loss import HybridSurvivalLoss, censored_pinball_loss
from tabicl.survival._scaler import SurvivalTimeScaler, standardize_survival_micro_batch

__all__ = [
    "TimeBinner",
    "DiscreteTimeSurvivalHead",
    "HybridSurvivalLoss",
    "censored_pinball_loss",
    "km_quantiles",
    "SurvivalTimeScaler",
    "standardize_survival_micro_batch",
]
