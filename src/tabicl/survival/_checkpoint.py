"""Survival checkpoint loading and metadata validation."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from tabicl._model.tabicl import TabICL
from tabicl.survival._head import TimeBinner


def validate_survival_metadata(
    metadata: dict,
    *,
    model: TabICL | None = None,
    device: str | torch.device = "cpu",
    require_time_scaler: bool = False,
) -> tuple[TimeBinner, dict[str, float]]:
    """Validate modern standardized-time metadata and reconstruct helpers."""
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint is missing survival_metadata.")
    if metadata.get("task") != "survival":
        raise ValueError("Checkpoint survival_metadata must declare task='survival'.")
    if metadata.get("time_scale") != "km_hybrid_log":
        raise ValueError("Checkpoint is not a modern km_hybrid_log survival checkpoint.")

    try:
        edges = metadata["binner_edges"].detach().float().cpu()
        means = metadata["binner_means"].detach().float().cpu()
    except (KeyError, AttributeError) as exc:
        raise ValueError("Checkpoint survival_metadata is missing tensor binner fields.") from exc

    num_bins = int(metadata.get("num_bins", len(means)))
    if edges.ndim != 1 or means.ndim != 1:
        raise ValueError("Checkpoint binner edges and means must be one-dimensional.")
    if len(edges) != num_bins + 1 or len(means) != num_bins:
        raise ValueError(
            f"Checkpoint binner shapes do not match K={num_bins}: "
            f"edges={len(edges)}, means={len(means)}."
        )
    if not torch.isfinite(edges).all() or not torch.isfinite(means).all():
        raise ValueError("Checkpoint binner contains non-finite values.")
    if not (edges.diff() > 0).all():
        raise ValueError("Checkpoint binner edges must be strictly increasing.")
    if not ((means >= edges[:-1]) & (means <= edges[1:])).all():
        raise ValueError("Checkpoint binner mean is outside bin boundaries.")

    if model is not None:
        if not getattr(model, "survival", False):
            raise ValueError("Checkpoint model is not configured for survival.")
        if int(model.num_quantiles) != num_bins:
            raise ValueError(
                f"Checkpoint survival_metadata K={num_bins} != model K={model.num_quantiles}."
            )

    raw_scaler = metadata.get("time_scaler")
    if raw_scaler is None:
        if require_time_scaler:
            raise ValueError("Checkpoint survival_metadata is missing time_scaler.")
        scaler = {
            "eps": 1e-8,
            "min_scale": 0.1,
            "z_min": float(edges[0]),
            "z_max": float(edges[-1]),
        }
    elif isinstance(raw_scaler, dict):
        try:
            scaler = {
                "eps": float(raw_scaler["eps"]),
                "min_scale": float(raw_scaler["min_scale"]),
                "z_min": float(raw_scaler["z_min"]),
                "z_max": float(raw_scaler["z_max"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Checkpoint time_scaler is missing required numeric fields.") from exc
    else:
        raise ValueError("Checkpoint time_scaler must be a dictionary.")

    if not math.isfinite(scaler["eps"]) or scaler["eps"] <= 0:
        raise ValueError(f"Checkpoint scaler has invalid eps={scaler['eps']}.")
    if not math.isfinite(scaler["min_scale"]) or scaler["min_scale"] <= 0:
        raise ValueError(f"Checkpoint scaler has invalid min_scale={scaler['min_scale']}.")
    if not math.isfinite(scaler["z_min"]) or not math.isfinite(scaler["z_max"]):
        raise ValueError("Checkpoint scaler bounds must be finite.")
    if not math.isclose(scaler["z_min"], float(edges[0]), rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("Scaler z_min does not match the first binner edge.")
    if not math.isclose(scaler["z_max"], float(edges[-1]), rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("Scaler z_max does not match the last binner edge.")

    return TimeBinner(edges, means).to(torch.device(device)), scaler


def load_survival_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[TabICL, TimeBinner, dict[str, float], dict]:
    """Strict-load a modern survival checkpoint on the requested device."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a checkpoint dictionary.")
    config = checkpoint.get("config")
    state = checkpoint.get("state_dict")
    if not isinstance(config, dict) or state is None:
        raise ValueError("Checkpoint must contain saved model config and state_dict.")
    if not bool(config.get("survival", False)):
        raise ValueError("Checkpoint config is not a survival model.")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint state_dict must be a dictionary.")
    stripped_state = {}
    for key, value in state.items():
        stripped_key = key.removeprefix("_orig_mod.")
        if stripped_key in stripped_state:
            raise ValueError(f"Checkpoint state_dict has duplicate key after prefix stripping: {stripped_key}")
        stripped_state[stripped_key] = value

    model = TabICL(**config)
    model.load_state_dict(stripped_state, strict=True)
    binner, scaler_config = validate_survival_metadata(
        checkpoint.get("survival_metadata"),
        model=model,
        device=device,
        require_time_scaler=True,
    )
    model.to(device)
    model.eval()
    return model, binner, scaler_config, checkpoint
