"""Per-task log-time scaling for survival in-context learning."""

from __future__ import annotations

import torch
from torch import Tensor

from tabicl.survival._km import km_quantiles


class SurvivalTimeScaler:
    """KM-hybrid per-task scaler for survival times.

    The scaler is fit on context observed outcomes only.  It maps raw positive
    times to standardized log-time values using KM median/IQR when available
    and observed log-time quantiles as fallback values:

    ``z = clamp((log(max(t, eps)) - loc) / scale, z_min, z_max)``

    At inference, fit one scaler per support set using the support
    ``(t_obs, delta)``, feed standardized support times to the model, interpret
    hazards on the standardized ``TimeBinner``, and map requested standardized
    times back with :meth:`inverse_time`.
    """

    def __init__(
        self,
        *,
        eps: float = 1e-8,
        min_scale: float = 0.1,
        z_min: float = -6.0,
        z_max: float = 6.0,
    ) -> None:
        self.eps = eps
        self.min_scale = min_scale
        self.z_min = z_min
        self.z_max = z_max
        self.loc: Tensor | None = None
        self.scale: Tensor | None = None
        self.metadata: dict[str, object] = {}

    def fit(
        self,
        t_context: Tensor,
        delta_context: Tensor,
        valid_mask: Tensor | None = None,
    ) -> "SurvivalTimeScaler":
        """Fit scaling parameters from context observed outcomes."""
        if t_context.shape != delta_context.shape:
            raise ValueError(
                f"t_context and delta_context must have the same shape, got {t_context.shape} and {delta_context.shape}"
            )
        if valid_mask is not None:
            mask = valid_mask.bool()
            t = t_context[mask]
            delta = delta_context[mask]
        else:
            t = t_context.reshape(-1)
            delta = delta_context.reshape(-1)

        log_t = torch.log(t.clamp_min(self.eps))
        finite = torch.isfinite(log_t)
        finite = finite & torch.isfinite(delta)
        delta = delta[finite]
        log_t = log_t[finite]
        if log_t.numel() == 0:
            raise ValueError("SurvivalTimeScaler.fit requires at least one finite context time.")
        if not (((delta == 0) | (delta == 1)).all()):
            raise ValueError("delta_context must contain only 0/1 event indicators.")

        log_t_f = log_t.float()
        q25_obs, q50_obs, q75_obs = torch.quantile(
            log_t_f, torch.tensor([0.25, 0.5, 0.75], device=log_t_f.device)
        )

        q25_km, q50_km, q75_km = km_quantiles(log_t_f, delta.float())

        if torch.isfinite(q50_km):
            loc = q50_km
            location_source = "km"
        else:
            loc = q50_obs
            location_source = "observed_fallback"

        if torch.isfinite(q25_km) and torch.isfinite(q75_km) and q75_km > q25_km:
            q25_used, q75_used = q25_km, q75_km
            scale_source = "km"
        else:
            q25_used, q75_used = q25_obs, q75_obs
            scale_source = "observed_fallback"

        scale_raw = (q75_used - q25_used) / 1.349
        scale = scale_raw.clamp_min(self.min_scale)

        self.loc = loc.to(dtype=torch.float32)
        self.scale = scale.to(dtype=torch.float32)
        self.metadata = {
            "method": "km_hybrid",
            "location_source": location_source,
            "scale_source": scale_source,
            "q25": float(q25_used.detach().cpu()),
            "q50": float(loc.detach().cpu()),
            "q75": float(q75_used.detach().cpu()),
            "q25_km": float(q25_km.detach().cpu()),
            "q50_km": float(q50_km.detach().cpu()),
            "q75_km": float(q75_km.detach().cpu()),
            "q25_obs": float(q25_obs.detach().cpu()),
            "q50_obs": float(q50_obs.detach().cpu()),
            "q75_obs": float(q75_obs.detach().cpu()),
            "scale_raw": float(scale_raw.detach().cpu()),
            "scale_was_lower_bounded": bool((scale_raw < self.min_scale).detach().cpu()),
        }
        return self

    def _check_fitted(self) -> tuple[Tensor, Tensor]:
        if self.loc is None or self.scale is None:
            raise RuntimeError("SurvivalTimeScaler must be fit before transforming times.")
        return self.loc, self.scale

    def raw_z(self, t: Tensor) -> Tensor:
        """Return unclipped standardized log-time."""
        loc, scale = self._check_fitted()
        loc = loc.to(device=t.device, dtype=t.dtype)
        scale = scale.to(device=t.device, dtype=t.dtype)
        return (torch.log(t.clamp_min(self.eps)) - loc) / scale

    def transform_time(self, t: Tensor) -> Tensor:
        """Transform raw times to clipped standardized log-time."""
        z = self.raw_z(t)
        z = torch.nan_to_num(z, nan=self.z_max, posinf=self.z_max, neginf=self.z_min)
        return z.clamp(self.z_min, self.z_max)

    def transform_observed(self, t_obs: Tensor, delta: Tensor) -> tuple[Tensor, Tensor]:
        """Transform observed labels and administratively censor above ``z_max``."""
        z_raw = self.raw_z(t_obs)
        above_horizon = (~torch.isfinite(z_raw)) | (z_raw > self.z_max)
        z = torch.nan_to_num(z_raw, nan=self.z_max, posinf=self.z_max, neginf=self.z_min)
        z = z.clamp(self.z_min, self.z_max)
        delta_out = torch.where(above_horizon, torch.zeros_like(delta.float()), delta.float())
        return z, delta_out

    def transform_event(self, t_event: Tensor) -> Tensor:
        """Transform privileged event times for synthetic imputation loss."""
        return self.transform_time(t_event)

    def transform_event_target(self, t_event: Tensor) -> tuple[Tensor, Tensor]:
        """Transform oracle query event times for loss supervision.

        Returns unclipped standardized log-times ``z_raw`` so that
        out-of-range targets can be identified.  Unlike
        :meth:`transform_observed`, this does *not* administratively
        censor any observation — every valid target remains an event.

        Parameters
        ----------
        t_event : Tensor
            True event times (must be finite and > 0).

        Returns
        -------
        z_raw : Tensor
            ``raw_z(t_event)`` — may be outside ``[z_min, z_max]``.
            Underflow/overflow values are still valid event-NLL targets
            (mapped to edge bins by ``TimeBinner.bin_index``).

        in_range : Tensor, bool
            ``z_min <= z_raw <= z_max``.  Out-of-range targets are
            excluded from pinball loss because predicted quantiles are
            capped at the finite horizon.

        Raises
        ------
        ValueError
            If any ``t_event`` is non-finite or ≤ 0.
        """
        if not torch.all(torch.isfinite(t_event) & (t_event > 0)):
            raise ValueError(
                "t_event targets must be finite and strictly positive."
            )
        z = self.raw_z(t_event)
        in_range = (z >= self.z_min) & (z <= self.z_max)
        return z, in_range

    def inverse_time(self, z: Tensor) -> Tensor:
        """Map standardized log-time back to raw time units."""
        loc, scale = self._check_fitted()
        loc = loc.to(device=z.device, dtype=z.dtype)
        scale = scale.to(device=z.device, dtype=z.dtype)
        return torch.exp(z * scale + loc)


def standardize_survival_micro_batch(
    t_train: Tensor,
    delta_train: Tensor,
    t_test: Tensor,
    delta_test: Tensor,
    t_event_test: Tensor | None,
    train_sizes_ds: Tensor,
    query_sizes_ds: Tensor,
    scaler_kwargs: dict[str, float],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
    """Apply context-only KM-hybrid scaling to a survival micro-batch.

    Fit scaler on ``(t_train, delta_train)`` (context only), then transform
    context, query, and optionally ``t_event_test`` with the same scaler.
    Query times never influence ``loc`` or ``scale``.

    When ``t_event_test`` is ``None`` (NLL-only training), the fifth and
    sixth return elements are ``None``.

    Returns
    -------
    t_train_z, delta_train_z, t_test_z, delta_test_z : Tensor
        Standardized context and query tensors.
    t_event_z_raw : Tensor or None
        Unclipped standardized event times (from ``transform_event_target``).
        May contain values outside ``[z_min, z_max]``.
    t_event_in_range : Tensor or None, bool
        Mask where ``z_min <= t_event_z_raw <= z_max``.  Out-of-range
        targets are excluded from pinball loss.
    """
    t_train_z = torch.empty_like(t_train)
    delta_train_z = torch.empty_like(delta_train, dtype=torch.float32)
    t_test_z = torch.empty_like(t_test)
    delta_test_z = torch.empty_like(delta_test, dtype=torch.float32)
    t_event_test_z = torch.empty_like(t_event_test) if t_event_test is not None else None
    t_event_in_range = torch.empty_like(t_event_test, dtype=torch.bool) if t_event_test is not None else None

    context_pos = torch.arange(t_train.shape[1], device=t_train.device)
    query_pos = torch.arange(t_test.shape[1], device=t_test.device)

    for ds_idx in range(t_train.shape[0]):
        context_mask = context_pos < train_sizes_ds[ds_idx]
        if not context_mask.any():
            context_mask = context_mask.clone()
            context_mask[0] = True

        scaler = SurvivalTimeScaler(**scaler_kwargs).fit(
            t_train[ds_idx], delta_train[ds_idx], valid_mask=context_mask,
        )

        ctx_t, ctx_delta = scaler.transform_observed(t_train[ds_idx], delta_train[ds_idx])
        ctx_t = torch.where(context_mask, ctx_t, torch.zeros_like(ctx_t))
        ctx_delta = torch.where(context_mask, ctx_delta, torch.zeros_like(ctx_delta))
        t_train_z[ds_idx] = ctx_t
        delta_train_z[ds_idx] = ctx_delta

        query_mask = query_pos < query_sizes_ds[ds_idx]
        q_t, q_delta = scaler.transform_observed(t_test[ds_idx], delta_test[ds_idx])
        t_test_z[ds_idx] = torch.where(query_mask, q_t, torch.zeros_like(q_t))
        delta_test_z[ds_idx] = torch.where(query_mask, q_delta, torch.zeros_like(q_delta))
        if t_event_test is not None and t_event_test_z is not None and t_event_in_range is not None:
            # Mask padding positions (zero-valued) before validation in
            # transform_event_target.  Padding keeps zero/in-range=False.
            t_ev_ds = t_event_test[ds_idx]
            q_event_z = torch.zeros_like(t_ev_ds)
            q_in_range = torch.zeros_like(t_ev_ds, dtype=torch.bool)
            if query_mask.any():
                q_event_z[query_mask], q_in_range[query_mask] = \
                    scaler.transform_event_target(t_ev_ds[query_mask])
            t_event_test_z[ds_idx] = q_event_z
            t_event_in_range[ds_idx] = q_in_range

    return t_train_z, delta_train_z, t_test_z, delta_test_z, t_event_test_z, t_event_in_range
