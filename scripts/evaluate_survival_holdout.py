#!/usr/bin/env python
"""Evaluate survival checkpoints on an immutable synthetic holdout."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from tabicl.survival import TabICLSurvivalPredictor
from tabicl.survival._holdout import load_holdout_slice, sha256_file, verify_holdout
from tabicl.survival._metrics import (
    all_metrics_finite,
    group_bootstrap_ci,
    macro_means,
    paired_group_bootstrap_ci,
    task_metrics,
)


QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
METRICS = [
    "oracle_event_nll",
    "observed_nll",
    "oracle_c_index",
    "oracle_ibs",
    "oracle_pinball",
    "event_in_horizon_fraction",
    "nonfinite_prediction_count",
    *[f"coverage_{level:g}" for level in QUANTILES],
]


def event_rate_band(rate: float) -> str:
    if rate < 0.55:
        return "low_[0.40,0.55)"
    if rate < 0.75:
        return "medium_[0.55,0.75)"
    return "high_[0.75,0.90]"


def feature_band(features: int) -> str:
    if features <= 10:
        return "small_2-10"
    if features <= 50:
        return "medium_11-50"
    return "large_51-100"


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summarize_slices(rows: list[dict]) -> dict:
    summaries = {}
    selectors = {
        "id": lambda row: row["suite"] == "id",
        **{
            f"diagnostic/{name}": lambda row, name=name: row["slice"] == name
            for name in sorted({row["slice"] for row in rows if row["suite"] == "diagnostic"})
        },
        **{
            f"mechanism/{name}": lambda row, name=name: row["mechanism"] == name
            for name in sorted({row["mechanism"] for row in rows if row["suite"] == "diagnostic"})
        },
        **{
            f"baseline/{name}": lambda row, name=name: row["baseline"] == name
            for name in sorted({row["baseline"] for row in rows if row["suite"] == "diagnostic"})
        },
        **{
            f"event_rate/{name}": lambda row, name=name: row["event_rate_band"] == name
            for name in sorted({row["event_rate_band"] for row in rows})
        },
        **{
            f"features/{name}": lambda row, name=name: row["feature_band"] == name
            for name in sorted({row["feature_band"] for row in rows})
        },
    }
    for name, selector in selectors.items():
        selected = [row for row in rows if selector(row)]
        if selected:
            summaries[name] = {
                "task_count": len(selected),
                "metrics": macro_means(selected, METRICS),
                "confidence_intervals": {
                    metric: group_bootstrap_ci(selected, metric)
                    for metric in METRICS
                },
            }
    return summaries


def evaluate_checkpoint(checkpoint_path, holdout_dir, manifest, device, task_batch_size):
    predictor = TabICLSurvivalPredictor.from_checkpoint(checkpoint_path, device=device)
    checkpoint = predictor.checkpoint
    step = checkpoint.get("curr_step")
    if not isinstance(step, int) or step < 0:
        raise ValueError(f"Checkpoint {checkpoint_path} is missing a valid curr_step.")
    rows = []
    context_rows = int(manifest["context_rows"])
    for entry in manifest["slices"]:
        payload = load_holdout_slice(Path(holdout_dir) / entry["filename"])
        for start in range(0, payload["batch_size"], task_batch_size):
            stop = min(start + task_batch_size, payload["batch_size"])
            X = payload["X"][start:stop]
            batch_t_obs = payload["t_obs"][start:stop]
            batch_delta = payload["delta"][start:stop]
            batch_t_event = payload["t_event"][start:stop]
            prediction = predictor.predict(
                X[:, :context_rows],
                batch_t_obs[:, :context_rows],
                batch_delta[:, :context_rows],
                X[:, context_rows:],
                quantile_levels=QUANTILES,
            )
            for local_idx in range(stop - start):
                idx = start + local_idx
                metrics = task_metrics(
                    prediction.hazard_logits[local_idx],
                    prediction.survival_probabilities[local_idx],
                    prediction.standardized_quantiles[local_idx],
                    prediction.quantile_levels,
                    batch_t_obs[local_idx, context_rows:],
                    batch_delta[local_idx, context_rows:],
                    batch_t_event[local_idx, context_rows:],
                    prediction.scalers[local_idx],
                    predictor.binner,
                )
                metrics["nonfinite_prediction_count"] += float(
                    (~torch.isfinite(prediction.raw_time_grid[local_idx])).sum()
                    + (~torch.isfinite(prediction.raw_quantiles[local_idx])).sum()
                )
                context_rate = float(
                    batch_delta[local_idx, :context_rows].float().mean()
                )
                features = int(payload["d"][idx])
                rows.append({
                    "checkpoint_step": step,
                    "checkpoint": str(checkpoint_path),
                    "task_id": int(payload["task_ids"][idx]),
                    "group_id": int(payload["group_ids"][idx]),
                    "suite": entry["suite"],
                    "slice": entry["name"],
                    "mechanism": entry["mechanism"],
                    "baseline": entry["baseline"],
                    "context_event_rate": context_rate,
                    "event_rate_band": event_rate_band(context_rate),
                    "active_features": features,
                    "feature_band": feature_band(features),
                    **metrics,
                })
    id_rows = [row for row in rows if row["suite"] == "id"]
    id_summary = macro_means(id_rows, METRICS)
    finite = all(all_metrics_finite(row, METRICS) for row in rows)
    finite = finite and all(row["nonfinite_prediction_count"] == 0.0 for row in rows)
    return {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "step": step,
        "eligible": finite,
        "id_metrics": id_summary,
        "id_confidence_intervals": {
            metric: group_bootstrap_ci(id_rows, metric)
            for metric in METRICS
        },
        "slices": summarize_slices(rows),
    }, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout-dir", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.task_batch_size < 1:
        raise ValueError("--task-batch-size must be positive.")

    manifest = verify_holdout(args.holdout_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output_dir}")

    checkpoint_summaries = []
    all_rows = []
    for checkpoint in args.checkpoints:
        summary, rows = evaluate_checkpoint(
            checkpoint, args.holdout_dir, manifest, args.device, args.task_batch_size,
        )
        checkpoint_summaries.append(summary)
        all_rows.extend(rows)

    eligible = [item for item in checkpoint_summaries if item["eligible"]]
    winner = min(eligible, key=lambda item: item["id_metrics"]["oracle_event_nll"]) if eligible else None
    baseline = min(checkpoint_summaries, key=lambda item: item["step"])
    id_rows = [row for row in all_rows if row["suite"] == "id"]
    for item in checkpoint_summaries:
        item["paired_delta_vs_step0_confidence_intervals"] = {
            metric: paired_group_bootstrap_ci(
                id_rows,
                metric,
                checkpoint_key=item["path"],
                baseline_key=baseline["path"],
            )
            for metric in METRICS
        }

    summary = {
        "holdout_dir": str(args.holdout_dir),
        "holdout_hash": manifest["holdout_hash"],
        "selection_metric": "id/oracle_event_nll",
        "winner": None if winner is None else {
            "path": winner["path"],
            "step": winner["step"],
            "oracle_event_nll": winner["id_metrics"]["oracle_event_nll"],
        },
        "checkpoints": checkpoint_summaries,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False)
    )
    with open(output_dir / "per_task.jsonl", "w") as handle:
        for row in all_rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True, allow_nan=False) + "\n")
    with open(output_dir / "comparison.csv", "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "path", "eligible", *METRICS],
        )
        writer.writeheader()
        for item in sorted(checkpoint_summaries, key=lambda value: value["step"]):
            writer.writerow({
                "step": item["step"],
                "path": item["path"],
                "eligible": item["eligible"],
                **item["id_metrics"],
            })
    print(json.dumps(summary["winner"], indent=2))


if __name__ == "__main__":
    main()
