"""Reusable checkpoint inference for TabICL survival models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from tabicl.survival._checkpoint import load_survival_checkpoint
from tabicl.survival._scaler import SurvivalTimeScaler


@dataclass
class SurvivalPrediction:
    """Predictions and task-specific transformations for a survival prompt."""

    hazard_logits: Tensor
    survival_probabilities: Tensor
    standardized_time_grid: Tensor
    raw_time_grid: Tensor
    standardized_quantiles: Tensor
    raw_quantiles: Tensor
    quantile_levels: Tensor
    scalers: list[SurvivalTimeScaler]


class TabICLSurvivalPredictor:
    """Load a survival checkpoint and predict from in-context survival prompts."""

    def __init__(self, model, binner, scaler_config, *, device="cpu", checkpoint=None):
        self.model = model
        self.binner = binner
        self.scaler_config = dict(scaler_config)
        self.device = torch.device(device)
        self.checkpoint = checkpoint

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, *, device="cpu"):
        model, binner, scaler_config, checkpoint = load_survival_checkpoint(
            checkpoint_path, device=device,
        )
        return cls(
            model, binner, scaler_config, device=device, checkpoint=checkpoint,
        )

    def predict(
        self,
        X_context: Tensor,
        t_context: Tensor,
        delta_context: Tensor,
        X_query: Tensor,
        *,
        quantile_levels: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
    ) -> SurvivalPrediction:
        """Predict query survival distributions without using query outcomes."""
        if X_context.ndim != 3 or X_query.ndim != 3:
            raise ValueError("X_context and X_query must have shape (B, rows, features).")
        if t_context.ndim != 2 or delta_context.ndim != 2:
            raise ValueError("t_context and delta_context must have shape (B, rows).")
        if t_context.shape != delta_context.shape:
            raise ValueError("t_context and delta_context must have matching shapes.")
        if X_context.shape[:2] != t_context.shape:
            raise ValueError("Context feature rows must match context survival outcomes.")
        if X_context.shape[0] != X_query.shape[0] or X_context.shape[2] != X_query.shape[2]:
            raise ValueError("Context and query features must share batch and feature dimensions.")

        t_context_cpu = t_context.detach().float().cpu()
        delta_context_cpu = delta_context.detach().float().cpu()
        z_context = torch.empty_like(t_context_cpu)
        delta_adjusted = torch.empty_like(delta_context_cpu)
        scalers: list[SurvivalTimeScaler] = []
        for task_idx in range(t_context_cpu.shape[0]):
            scaler = SurvivalTimeScaler(**self.scaler_config).fit(
                t_context_cpu[task_idx], delta_context_cpu[task_idx],
            )
            z_context[task_idx], delta_adjusted[task_idx] = scaler.transform_observed(
                t_context_cpu[task_idx], delta_context_cpu[task_idx],
            )
            scalers.append(scaler)

        X = torch.cat([X_context, X_query], dim=1).to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            hazard_logits = self.model(
                X,
                z_context.to(self.device),
                delta_train=delta_adjusted.to(self.device),
            ).float()
            survival = self.binner.survival(hazard_logits)
            levels = torch.as_tensor(
                quantile_levels, device=self.device, dtype=torch.float32,
            )
            standardized_quantiles = self.binner.quantile_at(hazard_logits, levels)

        hazard_cpu = hazard_logits.cpu()
        survival_cpu = survival.cpu()
        quantiles_cpu = standardized_quantiles.cpu()
        standardized_grid = self.binner.bin_edges.detach().float().cpu()
        raw_grid = torch.stack([
            scaler.inverse_time(standardized_grid.double()) for scaler in scalers
        ])
        raw_quantiles = torch.stack([
            scaler.inverse_time(quantiles_cpu[idx].double())
            for idx, scaler in enumerate(scalers)
        ])
        return SurvivalPrediction(
            hazard_logits=hazard_cpu,
            survival_probabilities=survival_cpu,
            standardized_time_grid=standardized_grid,
            raw_time_grid=raw_grid,
            standardized_quantiles=quantiles_cpu,
            raw_quantiles=raw_quantiles,
            quantile_levels=levels.cpu(),
            scalers=scalers,
        )
