#!/usr/bin/env python
"""Inspect a training checkpoint for non-finite tensor state."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence

import torch


def iter_tensors(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix or "<root>", value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from iter_tensors(child, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            yield from iter_tensors(child, f"{prefix}[{index}]")


def summarize(name, value):
    tensor_count = 0
    element_count = 0
    nonfinite_count = 0
    bad_paths = []

    for path, tensor in iter_tensors(value):
        tensor_count += 1
        element_count += tensor.numel()
        if tensor.is_floating_point() or tensor.is_complex():
            bad = (~torch.isfinite(tensor)).sum().item()
            nonfinite_count += bad
            if bad and len(bad_paths) < 10:
                bad_paths.append(f"{path} ({bad})")

    print(
        f"{name}: tensors={tensor_count}, elements={element_count}, "
        f"nonfinite={nonfinite_count}"
    )
    for path in bad_paths:
        print(f"  bad: {path}")
    return nonfinite_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a checkpoint dictionary.")

    print(f"checkpoint: {args.checkpoint}")
    print(f"curr_step: {checkpoint.get('curr_step', 'unknown')}")
    print(f"scaler_state: {checkpoint.get('scaler_state', {})}")

    nonfinite = 0
    nonfinite += summarize("model", checkpoint.get("state_dict", {}))
    nonfinite += summarize("optimizer", checkpoint.get("optimizer_state", {}))

    if nonfinite:
        raise SystemExit(f"FAILED: checkpoint contains {nonfinite} non-finite value(s).")
    print("PASS: checkpoint model and optimizer tensors are finite.")


if __name__ == "__main__":
    main()
