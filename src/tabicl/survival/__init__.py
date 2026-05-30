"""Survival head and hybrid loss for TabICL fine-tuning with time-to-event data.

Provides a discrete-time survival prediction head and a hybrid loss function that
combines the discrete survival negative log-likelihood (on observables `(t_obs, delta)`)
with a pinball imputation loss on the privileged counterfactual event times `t_event`
(available only at training time in synthetic data).

Usage sketch::

    from tabicl.survival import TimeBinner, DiscreteTimeSurvivalHead, HybridSurvivalLoss

    # Build bin boundaries from training event times
    binner = TimeBinner.from_event_times(t_event_all, num_bins=50)

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
from tabicl.survival._loss import HybridSurvivalLoss, censored_pinball_loss
