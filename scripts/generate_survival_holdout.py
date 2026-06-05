#!/usr/bin/env python
"""Generate the immutable Stage 1 synthetic survival holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from survival_prior import SurvivalPriorDataset
from tabicl.survival._holdout import (
    HOLDOUT_SCHEMA_VERSION,
    canonical_hash,
    save_holdout_slice,
)


DIAGNOSTICS = [
    ("ph_weibull", "ph", "weibull"),
    ("ph_gompertz", "ph", "gompertz"),
    ("ph_loglogistic", "ph", "loglogistic"),
    ("ph_lognormal", "ph", "lognormal"),
    ("aft_weibull", "aft", "weibull"),
    ("aft_loglogistic", "aft", "loglogistic"),
    ("aft_lognormal", "aft", "lognormal"),
]


def source_fingerprint(repo_root: Path) -> str:
    digest = hashlib.sha256()
    paths = [
        Path(__file__).resolve(),
        repo_root / "survival_prior.py",
        repo_root / "src/tabicl/survival/_holdout.py",
    ]
    paths.extend(sorted((repo_root / "src/tabicl/prior").glob("*.py")))
    for path in paths:
        digest.update(str(path.relative_to(repo_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prior_kwargs(batch_size: int, model_type: str, baseline_mode: str) -> dict:
    return {
        "batch_size": batch_size,
        "batch_size_per_gp": 4,
        "min_features": 2,
        "max_features": 100,
        "min_seq_len": 1024,
        "max_seq_len": 1024,
        "min_train_size": 1.0,
        "max_train_size": 1.0,
        "prior_type": "mlp_scm",
        "model_type": model_type,
        "beta_sampling": "log_uniform",
        "min_beta": 0.25,
        "max_beta": 2.0,
        "baseline_param_prior": "broad",
        "time_scale_sampling": "log_uniform",
        "min_time_scale": 0.2,
        "max_time_scale": 5.0,
        "baseline_types": ["weibull", "gompertz", "loglogistic", "lognormal"],
        "baseline_mode": baseline_mode,
        "censoring_strategy": "target_event_rate",
        "min_event_rate": 0.40,
        "max_event_rate": 0.90,
        "calibration_scope": "context",
        "n_jobs": 1,
        "device": "cpu",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="survival_holdouts/stage1_v1")
    parser.add_argument("--np-seed", type=int, default=20260605)
    parser.add_argument("--torch-seed", type=int, default=20260606)
    parser.add_argument("--id-tasks", type=int, default=512)
    parser.add_argument("--diagnostic-tasks", type=int, default=32)
    args = parser.parse_args()
    if args.id_tasks <= 0 or args.id_tasks % 4 != 0:
        raise ValueError("--id-tasks must be positive and divisible by 4.")
    if args.diagnostic_tasks <= 0 or args.diagnostic_tasks % 4 != 0:
        raise ValueError("--diagnostic-tasks must be positive and divisible by 4.")

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite immutable holdout directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    repo_root = REPO_ROOT

    slices = [("id", "mix", "mix", args.id_tasks), *[
        (name, mechanism, baseline, args.diagnostic_tasks)
        for name, mechanism, baseline in DIAGNOSTICS
    ]]
    manifest_slices = []
    task_offset = 0
    group_offset = 0
    for slice_index, (name, mechanism, baseline, count) in enumerate(slices):
        np_seed = args.np_seed + slice_index
        torch_seed = args.torch_seed + slice_index
        np.random.seed(np_seed)
        torch.manual_seed(torch_seed)
        kwargs = prior_kwargs(count, mechanism, baseline)
        batch = SurvivalPriorDataset(**kwargs).get_batch()
        filename = f"{slice_index:02d}_{name}.pt"
        saved = save_holdout_slice(
            output_dir / filename,
            batch,
            task_offset=task_offset,
            group_offset=group_offset,
        )
        manifest_slices.append({
            "name": name,
            "suite": "id" if name == "id" else "diagnostic",
            "mechanism": mechanism,
            "baseline": baseline,
            "filename": filename,
            "np_seed": np_seed,
            "torch_seed": torch_seed,
            "generation_args": kwargs,
            **saved,
        })
        task_offset += count
        group_offset += count // 4

    manifest = {
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "holdout_id": output_dir.name,
        "context_rows": 512,
        "query_rows": 512,
        "task_count": task_offset,
        "base_np_seed": args.np_seed,
        "base_torch_seed": args.torch_seed,
        "source_fingerprint": source_fingerprint(repo_root),
        "slices": manifest_slices,
    }
    manifest["holdout_hash"] = canonical_hash(manifest)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Created immutable holdout {output_dir} ({task_offset} tasks)")


if __name__ == "__main__":
    main()
