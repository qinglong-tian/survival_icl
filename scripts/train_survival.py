#!/usr/bin/env python
"""Fine-tune a pretrained TabICL checkpoint with a discrete-time survival head.

Loads a TabICL regressor checkpoint, swaps the output decoder for a
:class:`DiscreteTimeSurvivalHead`, and trains with the
:class:`HybridSurvivalLoss` on pre-generated survival prior ``.pt`` files.

Usage::

    # 1. Generate a small test batch first
    python survival_prior.py --model_type ph --num_batches 1 --batch_size 4 \\
        --prior_type mlp_scm --baseline_types weibull --baseline_mode weibull \\
        --save_dir test_survival_data

    # 2. Train
    python scripts/train_survival.py --data_dir test_survival_data --max_steps 100

Data format expected per ``.pt`` file (as saved by ``SaveSurvivalPriorDataset``):
    ``{"X": ..., "t": ..., "delta": ..., "t_event": ..., "d": ..., "seq_lens": ..., "train_sizes": ..., "batch_size": B}``
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_survival_batch(batch_path: Path, device: torch.device) -> dict:
    """Load a single ``batch_XXXXXX.pt`` file to device.

    Returns all tensors in the file.  Sparse ``X`` is densified if nested.
    """
    batch = torch.load(batch_path, map_location=device, weights_only=True)

    X = batch["X"]
    t_tensor = batch["t"]
    delta_tensor = batch["delta"]
    t_event_tensor = batch.get("t_event", None)
    d = batch["d"]
    seq_lens = batch["seq_lens"]
    train_sizes = batch["train_sizes"]

    if X.is_nested:
        X_dense = X.to_padded_tensor(padding=0.0)
    elif X.dim() == 1:
        # Flattened sparse from dense2sparse (the typical save format)
        from tabicl.prior._genload import sparse2dense
        B = len(d)
        T = int(seq_lens[0].item())
        X_dense = sparse2dense(X, d.repeat_interleave(T), dtype=torch.float32).view(B, T, -1)
    else:
        X_dense = X

    return {
        "X": X_dense,
        "t": t_tensor,
        "delta": delta_tensor,
        "t_event": t_event_tensor,
        "d": d,
        "seq_lens": seq_lens,
        "train_sizes": train_sizes,
    }


def _split_ctx_query(
    B: int,
    seq_len: int,
    ctx_ratio: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> tuple[int, int]:
    """Return ``(ctx_size, query_size)`` for a dataset of ``seq_len`` samples.

    Uses a deterministic or random split at ``ctx_ratio``.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    ctx_size = max(2, int(seq_len * ctx_ratio))
    query_size = seq_len - ctx_size
    return ctx_size, query_size


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(args):
    # ---- Device ----------------------------------------------------------
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s", device)

    # ---- Load data files ------------------------------------------------
    data_dir = Path(args.data_dir)
    pt_files = sorted(data_dir.glob("batch_*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No batch_*.pt files found in {data_dir}")

    logger.info("Found %d batch files in %s", len(pt_files), data_dir)

    # Load a few batches to collect t_event for TimeBinner fitting
    sample_t_events = []
    for f in pt_files[: min(4, len(pt_files))]:
        b = _load_survival_batch(f, torch.device("cpu"))
        if b["t_event"] is not None:
            sample_t_events.append(b["t_event"].reshape(-1))
    if not sample_t_events:
        raise RuntimeError("No t_event found in data files — this script requires synthetic data with counterfactuals.")
    all_t_event = torch.cat(sample_t_events)

    # ---- Build TimeBinner ------------------------------------------------
    from tabicl.survival import TimeBinner

    binner = TimeBinner.from_event_times(all_t_event, num_bins=args.num_bins, headroom=0.05)
    binner = binner.to(device)
    logger.info("TimeBinner: %r", binner)

    # ---- Load pretrained TabICL ------------------------------------------
    from tabicl._sklearn.regressor import TabICLRegressor

    reg = TabICLRegressor(
        n_estimators=1,
        model_path=args.model_path,
        allow_auto_download=args.allow_auto_download,
        checkpoint_version=args.checkpoint_version,
        device=device,
        random_state=args.seed,
    )
    reg._resolve_device()
    reg._load_model()
    model = reg.model_

    # ---- Swap decoder with survival head ---------------------------------
    from tabicl.survival import DiscreteTimeSurvivalHead

    icl_dim = model.embed_dim * model.row_num_cls
    d_model = icl_dim  # the dimension fed to the decoder
    survival_head = DiscreteTimeSurvivalHead(d_model=d_model, num_bins=args.num_bins).to(device)

    # Replace the decoder. The upstream decoder is
    #   nn.Sequential(Linear(d_model, 2*d_model), GELU(), Linear(2*d_model, out_dim))
    model.icl_predictor.decoder = survival_head
    logger.info("Replaced decoder with DiscreteTimeSurvivalHead(d_model=%d, num_bins=%d)", d_model, args.num_bins)

    # Update out_dim on the ICLearning and InferenceManager so shape checks pass
    model.icl_predictor.inference_mgr._out_dim = args.num_bins  # type: ignore[attr-defined]
    # The `out_dim` attribute in ICLearning __init__ is `out_dim` (stored as attribute but not stored).
    # We just need the model.forward to work — it calls decoder then slices to test.
    # decoder now outputs (B, T, K) → sliced to (B, test_size, K).  That's fine.

    model.train()
    model.to(device)

    # Freeze upstreams if requested
    if args.freeze_col:
        for p in model.col_embedder.parameters():
            p.requires_grad = False
    if args.freeze_row:
        for p in model.row_interactor.parameters():
            p.requires_grad = False
    if args.freeze_icl:
        for p in model.icl_predictor.parameters():
            p.requires_grad = False

    # Only train the survival head unless we unfroze things
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        # If everything is frozen, at least train the head
        for p in survival_head.parameters():
            p.requires_grad = True
        trainable_params = list(survival_head.parameters())
    logger.info("Trainable parameters: %d", sum(p.numel() for p in trainable_params))

    # ---- Optimizer -------------------------------------------------------
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # ---- Loss function ---------------------------------------------------
    from tabicl.survival import HybridSurvivalLoss

    loss_fn = HybridSurvivalLoss(
        alpha_start=args.alpha_start,
        alpha_floor=args.alpha_floor,
        max_steps=args.max_steps,
    )

    # ---- Training loop ---------------------------------------------------
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))
    rng = np.random.default_rng(args.seed)
    step = 0
    batch_files_cycle = pt_files[:]
    file_idx = 0

    t_start = time.time()
    total_surv = 0.0
    total_impute = 0.0
    log_every = max(1, args.max_steps // 20)

    # Scan for max_features across a sample of files
    max_features_global = 0
    for f in pt_files[: min(8, len(pt_files))]:
        b = _load_survival_batch(f, torch.device("cpu"))
        h = b["X"].shape[-1]
        if h > max_features_global:
            max_features_global = h
    logger.info("Global max_features: %d", max_features_global)

    while step < args.max_steps:
        # Cycle through files
        if file_idx >= len(batch_files_cycle):
            rng.shuffle(batch_files_cycle)
            file_idx = 0

        batch_path = batch_files_cycle[file_idx]
        file_idx += 1

        batch = _load_survival_batch(batch_path, device)
        X_full = batch["X"]  # (B, seq_len, H)
        t_obs = batch["t"]  # (B, seq_len)
        delta = batch["delta"]  # (B, seq_len)
        t_event = batch["t_event"]  # (B, seq_len) or None
        d = batch["d"]  # (B,)

        B, seq_len_total, H = X_full.shape

        # Pad X to global max_features for consistent tensor shapes
        if H < max_features_global:
            pad = torch.zeros(B, seq_len_total, max_features_global - H, device=device, dtype=X_full.dtype)
            X_full = torch.cat([X_full, pad], dim=-1)
        elif H > max_features_global:
            max_features_global = H

        # Process each dataset within the batch
        for ds_idx in range(B):
            seq_len = int(batch["seq_lens"][ds_idx].item())
            if seq_len < 4:
                continue  # too small

            # Take only the active samples (ignore padding)
            X_ds = X_full[ds_idx : ds_idx + 1, :seq_len]  # (1, seq_len, H)
            t_obs_ds = t_obs[ds_idx : ds_idx + 1, :seq_len]  # (1, seq_len)
            delta_ds = delta[ds_idx : ds_idx + 1, :seq_len]  # (1, seq_len)
            if t_event is not None:
                t_event_ds = t_event[ds_idx : ds_idx + 1, :seq_len]  # (1, seq_len)
            else:
                t_event_ds = t_obs_ds  # fallback: no privileged signal

            # Split into context / query
            ctx_size, query_size = _split_ctx_query(1, seq_len, ctx_ratio=args.ctx_ratio, rng=rng)
            if query_size == 0:
                continue

            # Permute rows so context isn't always first N
            perm = rng.permutation(seq_len)
            ctx_idx = perm[:ctx_size]
            qry_idx = perm[ctx_size : ctx_size + query_size]

            X_ctx = X_ds[:, ctx_idx]  # (1, ctx_size, H)
            X_qry = X_ds[:, qry_idx]  # (1, query_size, H)

            # y_train for ICL is t_obs for context samples (the model sees times as labels)
            y_train = t_obs_ds[:, ctx_idx]  # (1, ctx_size)

            # Targets for loss: query samples
            t_obs_qry = t_obs_ds[:, qry_idx].squeeze(0)  # (query_size,)
            delta_qry = delta_ds[:, qry_idx].squeeze(0)  # (query_size,)
            t_event_qry = t_event_ds[:, qry_idx].squeeze(0)  # (query_size,)

            # Concatenate context + query for TabICL.forward
            X_input = torch.cat([X_ctx, X_qry], dim=1)  # (1, ctx_size + query_size, H)

            # Forward pass
            h_raw = model(X_input, y_train)  # (1, query_size, K)
            h_raw = h_raw.squeeze(0)  # (query_size, K)

            # Compute loss
            total, comps = loss_fn(h_raw, t_obs_qry, delta_qry, t_event_qry, binner, step)
            total_surv += comps["surv_nll"]
            total_impute += comps["impute"]

            # Backward
            optimizer.zero_grad()
            if args.amp and device.type == "cuda":
                scaler.scale(total).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                optimizer.step()

            step += 1

            if step % log_every == 0 or step == 1:
                elapsed = time.time() - t_start
                avg_surv = total_surv / log_every
                avg_impute = total_impute / log_every
                logger.info(
                    "step %5d/%d | surv=%.4f impute=%.4f α=%.3f | %.1f min elapsed",
                    step,
                    args.max_steps,
                    avg_surv,
                    avg_impute,
                    comps["alpha"],
                    elapsed / 60,
                )
                total_surv = 0.0
                total_impute = 0.0

            if step >= args.max_steps:
                break

    # ---- Save checkpoint -------------------------------------------------
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_dir / "survival_finetuned.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "binner_edges": binner.bin_edges,
                "binner_means": binner.bin_means,
                "d_model": d_model,
                "num_bins": args.num_bins,
                "step": step,
            },
            ckpt_path,
        )
        logger.info("Saved checkpoint to %s", ckpt_path)

    logger.info("Training complete.  Final step: %d", step)
    return model, binner


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune TabICL with a discrete-time survival head",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with batch_*.pt survival files")
    parser.add_argument("--ctx_ratio", type=float, default=0.5, help="Fraction of each dataset used as context")

    # Model loading
    parser.add_argument("--model_path", type=str, default=None, help="Path to TabICL checkpoint (None = HuggingFace)")
    parser.add_argument("--allow_auto_download", action="store_true", default=True, help="Auto-download checkpoint")
    parser.add_argument(
        "--checkpoint_version", type=str, default="tabicl-regressor-v2-20260212.ckpt", help="HF checkpoint version"
    )

    # Survival head
    parser.add_argument("--num_bins", type=int, default=50, help="Number of time bins K")

    # Loss
    parser.add_argument("--alpha_start", type=float, default=3.0, help="Initial imputation loss weight")
    parser.add_argument("--alpha_floor", type=float, default=0.05, help="Minimum imputation loss weight after decay")

    # Training
    parser.add_argument("--max_steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max gradient norm (0 to disable)")
    parser.add_argument("--amp", action="store_true", default=True, help="Use AMP on CUDA")
    parser.add_argument("--device", type=str, default=None, help="Device (auto if None)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Freezing
    parser.add_argument("--freeze_col", action="store_true", default=False)
    parser.add_argument("--freeze_row", action="store_true", default=False)
    parser.add_argument("--freeze_icl", action="store_true", default=False)

    # Output
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for final checkpoint")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    train(args)


if __name__ == "__main__":
    main()
