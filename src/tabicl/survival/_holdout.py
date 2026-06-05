"""Immutable synthetic survival holdout artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from tabicl.prior._genload import dense2sparse, sparse2dense


HOLDOUT_SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def save_holdout_slice(
    path: str | Path,
    batch,
    *,
    task_offset: int,
    group_offset: int,
    group_size: int = 4,
) -> dict:
    """Save one fixed-length holdout slice in compact sparse form."""
    X, t_obs, delta, t_event, d, seq_lens, _ = batch
    if X.is_nested or len(seq_lens.unique()) != 1:
        raise ValueError("Holdout slices require one fixed sequence length.")
    batch_size, seq_len, max_features = X.shape
    if group_size < 1 or batch_size % group_size != 0:
        raise ValueError("Holdout slice batch size must be divisible by group_size.")
    X_sparse = dense2sparse(
        X.reshape(-1, max_features),
        d.repeat_interleave(seq_len),
        dtype=torch.float32,
    )
    group_ids = torch.arange(batch_size, dtype=torch.long) // group_size + group_offset
    payload = {
        "X_sparse": X_sparse.cpu(),
        "t_obs": t_obs.cpu().float(),
        "delta": delta.cpu().float(),
        "t_event": t_event.cpu().float(),
        "d": d.cpu().long(),
        "seq_lens": seq_lens.cpu().long(),
        "task_ids": torch.arange(task_offset, task_offset + batch_size, dtype=torch.long),
        "group_ids": group_ids,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "max_features": max_features,
        "group_size": group_size,
    }
    torch.save(payload, path)
    return {
        "task_count": batch_size,
        "seq_len": seq_len,
        "max_features": max_features,
        "group_size": group_size,
        "sha256": sha256_file(path),
    }


def verify_holdout(holdout_dir: str | Path) -> dict:
    """Verify manifest schema and every immutable data-file hash."""
    holdout_dir = Path(holdout_dir)
    manifest_path = holdout_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing holdout manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != HOLDOUT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported holdout schema_version={manifest.get('schema_version')}."
        )
    slices = manifest.get("slices")
    if not isinstance(slices, list) or not slices:
        raise ValueError("Holdout manifest must contain at least one slice.")
    task_count = 0
    for entry in slices:
        path = holdout_dir / entry["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing holdout slice: {path}")
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise ValueError(f"Holdout slice hash mismatch: {path}")
        task_count += int(entry["task_count"])
    if "task_count" in manifest and task_count != int(manifest["task_count"]):
        raise ValueError("Holdout manifest task_count does not match its slices.")
    expected_hash = manifest.get("holdout_hash")
    unsigned = {key: value for key, value in manifest.items() if key != "holdout_hash"}
    if expected_hash != canonical_hash(unsigned):
        raise ValueError("Holdout manifest hash mismatch.")
    return manifest


def load_holdout_slice(path: str | Path) -> dict:
    """Load one verified fixed-length holdout slice and reconstruct dense X."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    row_lengths = payload["d"].repeat_interleave(payload["seq_len"])
    payload["X"] = sparse2dense(
        payload.pop("X_sparse"),
        row_lengths,
        max_len=payload["max_features"],
        dtype=torch.float32,
    ).view(payload["batch_size"], payload["seq_len"], payload["max_features"])
    return payload
