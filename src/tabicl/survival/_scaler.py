"""Per-task log-time scaling for survival in-context learning."""

from __future__ import annotations

import torch
from torch import Tensor


class SurvivalTimeScaler:
    """Robust per-task scaler for survival times.

    The scaler is fit on context observed times only.  It maps raw positive
    times to standardized log-time values:

    ``z = clamp((log(max(t, eps)) - loc) / scale, z_min, z_max)``

    where ``loc`` is the median context log-time and ``scale`` is
    ``IQR(log_t) / 1.349`` clamped below by ``min_scale``.  At inference, fit
    one scaler per support set, feed standardized support times to the model,
    interpret hazards on the standardized ``TimeBinner``, and map requested
    standardized times back with :meth:`inverse_time`.
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

    def fit(self, t_context: Tensor, valid_mask: Tensor | None = None) -> "SurvivalTimeScaler":
        """Fit scaling parameters from context observed times."""
        if valid_mask is not None:
            t = t_context[valid_mask.bool()]
        else:
            t = t_context.reshape(-1)

        log_t = torch.log(t.clamp_min(self.eps))
        log_t = log_t[torch.isfinite(log_t)]
        if log_t.numel() == 0:
            raise ValueError("SurvivalTimeScaler.fit requires at least one finite context time.")

        log_t_f = log_t.float()
        q25, q75 = torch.quantile(log_t_f, torch.tensor([0.25, 0.75], device=log_t_f.device))
        self.scale = ((q75 - q25) / 1.349).clamp_min(self.min_scale)
        self.loc = log_t_f.median()
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

    def inverse_time(self, z: Tensor) -> Tensor:
        """Map standardized log-time back to raw time units."""
        loc, scale = self._check_fitted()
        loc = loc.to(device=z.device, dtype=z.dtype)
        scale = scale.to(device=z.device, dtype=z.dtype)
        return torch.exp(z * scale + loc)

