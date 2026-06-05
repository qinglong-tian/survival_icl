"""Tests for survival checkpoint inference, holdouts, and evaluation metrics."""

from __future__ import annotations

import inspect
import json
import math
import subprocess
import sys
import types

import pytest
import torch

from tabicl._model.tabicl import TabICL
from tabicl.survival import TabICLSurvivalPredictor, TimeBinner
from tabicl.survival._holdout import (
    HOLDOUT_SCHEMA_VERSION,
    canonical_hash,
    load_holdout_slice,
    save_holdout_slice,
    verify_holdout,
)
from tabicl.survival._metrics import (
    harrell_c_index,
    oracle_integrated_brier,
    task_metrics,
)


def tiny_survival_checkpoint(path):
    config = {
        "max_classes": 0,
        "num_quantiles": 8,
        "embed_dim": 16,
        "col_num_blocks": 1,
        "col_nhead": 2,
        "col_num_inds": 4,
        "row_num_blocks": 1,
        "row_nhead": 2,
        "row_num_cls": 2,
        "icl_num_blocks": 1,
        "icl_nhead": 2,
        "ff_factor": 2,
        "survival": True,
    }
    model = TabICL(**config)
    binner = TimeBinner.from_standardized_range(num_bins=8)
    torch.save({
        "config": config,
        "state_dict": model.state_dict(),
        "curr_step": 0,
        "survival_metadata": {
            "task": "survival",
            "time_scale": "km_hybrid_log",
            "num_bins": 8,
            "binner_edges": binner.bin_edges,
            "binner_means": binner.bin_means,
            "time_scaler": {"eps": 1e-8, "min_scale": 0.1, "z_min": -6.0, "z_max": 6.0},
        },
    }, path)
    return model


def test_checkpoint_predictor_matches_direct_forward_and_returns_monotonic_survival(tmp_path):
    checkpoint_path = tmp_path / "step-0.ckpt"
    model = tiny_survival_checkpoint(checkpoint_path)
    predictor = TabICLSurvivalPredictor.from_checkpoint(checkpoint_path)
    X_context = torch.randn(1, 4, 3)
    X_query = torch.randn(1, 3, 3)
    t_context = torch.tensor([[1.0, 2.0, 3.0, 5.0]])
    delta_context = torch.tensor([[1.0, 0.0, 1.0, 1.0]])

    prediction = predictor.predict(X_context, t_context, delta_context, X_query)
    scaler = prediction.scalers[0]
    z_context, delta_adjusted = scaler.transform_observed(t_context[0], delta_context[0])
    model.eval()
    with torch.inference_mode():
        direct = model(
            torch.cat([X_context, X_query], dim=1),
            z_context.unsqueeze(0),
            delta_train=delta_adjusted.unsqueeze(0),
        )

    assert torch.allclose(prediction.hazard_logits, direct, atol=1e-5)
    assert prediction.raw_quantiles.shape == (1, 3, 5)
    assert prediction.raw_quantiles.dtype == torch.float64
    assert torch.isfinite(prediction.raw_quantiles).all()
    assert (prediction.survival_probabilities.diff(dim=-1) <= 1e-6).all()
    assert {
        "t_query", "delta_query", "t_event_query",
    }.isdisjoint(inspect.signature(predictor.predict).parameters)


def test_checkpoint_predictor_rejects_non_survival_checkpoint(tmp_path):
    path = tmp_path / "regression.ckpt"
    torch.save({"config": {"survival": False}, "state_dict": {}}, path)
    with pytest.raises(ValueError, match="not a survival"):
        TabICLSurvivalPredictor.from_checkpoint(path)


def test_checkpoint_predictor_strips_compiled_prefixes(tmp_path):
    path = tmp_path / "step-0.ckpt"
    tiny_survival_checkpoint(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["state_dict"] = {
        f"_orig_mod.{key}": value for key, value in checkpoint["state_dict"].items()
    }
    torch.save(checkpoint, path)

    predictor = TabICLSurvivalPredictor.from_checkpoint(path)
    assert predictor.model.survival


def test_checkpoint_predictor_requires_modern_scaler_metadata(tmp_path):
    path = tmp_path / "step-0.ckpt"
    tiny_survival_checkpoint(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["survival_metadata"].pop("time_scaler")
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="missing time_scaler"):
        TabICLSurvivalPredictor.from_checkpoint(path)


def test_holdout_round_trip_and_hash_validation(tmp_path):
    output = tmp_path / "holdout"
    output.mkdir()
    X = torch.randn(4, 8, 5)
    t = torch.rand(4, 8) + 0.1
    delta = torch.randint(0, 2, (4, 8)).float()
    t_event = t + torch.rand(4, 8)
    d = torch.tensor([2, 3, 4, 5])
    seq_lens = torch.full((4,), 8)
    train_sizes = seq_lens.clone()
    entry = save_holdout_slice(
        output / "00_id.pt",
        (X, t, delta, t_event, d, seq_lens, train_sizes),
        task_offset=0,
        group_offset=0,
    )
    manifest = {
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "holdout_id": "tiny",
        "slices": [{"filename": "00_id.pt", **entry}],
    }
    manifest["holdout_hash"] = canonical_hash(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest))

    assert verify_holdout(output)["holdout_id"] == "tiny"
    loaded = load_holdout_slice(output / "00_id.pt")
    for idx, active in enumerate(d):
        assert torch.allclose(loaded["X"][idx, :, :active], X[idx, :, :active])
        assert not loaded["X"][idx, :, active:].any()
    with open(output / "00_id.pt", "ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_holdout(output)


def test_holdout_slice_serialization_is_byte_reproducible(tmp_path):
    X = torch.randn(4, 8, 5)
    t = torch.rand(4, 8) + 0.1
    delta = torch.randint(0, 2, (4, 8)).float()
    t_event = t + torch.rand(4, 8)
    d = torch.tensor([2, 3, 4, 5])
    seq_lens = torch.full((4,), 8)
    batch = (X, t, delta, t_event, d, seq_lens, seq_lens)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    save_holdout_slice(first / "00_id.pt", batch, task_offset=0, group_offset=0)
    save_holdout_slice(second / "00_id.pt", batch, task_offset=0, group_offset=0)

    assert (first / "00_id.pt").read_bytes() == (second / "00_id.pt").read_bytes()


def test_oracle_metrics_perfect_predictions():
    times = torch.tensor([1.0, 2.0, 3.0])
    assert harrell_c_index(times, -times) == pytest.approx(1.0)
    grid = torch.tensor([0.0, 1.0, 2.0])
    event_time = torch.tensor([0.5, 1.5])
    perfect_survival = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    assert oracle_integrated_brier(perfect_survival, event_time, grid) == pytest.approx(0.0)


def test_oracle_metrics_cover_ties_censoring_and_out_of_horizon_events():
    class IdentityScaler:
        z_min = -1.0
        z_max = 1.0

        def transform_event_target(self, event_time):
            return event_time, (event_time >= self.z_min) & (event_time <= self.z_max)

        def transform_observed(self, observed_time, delta):
            return observed_time, delta

    binner = TimeBinner(
        torch.tensor([-1.0, 0.0, 1.0]),
        torch.tensor([-0.5, 0.5]),
    )
    hazard_logits = torch.zeros(2, 2)
    survival = binner.survival(hazard_logits)
    levels = torch.tensor([0.5])
    quantiles = binner.quantile_at(hazard_logits, levels)
    metrics = task_metrics(
        hazard_logits,
        survival,
        quantiles,
        levels,
        torch.tensor([-0.5, 0.5]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([-0.5, 2.0]),
        IdentityScaler(),
        binner,
    )

    assert harrell_c_index(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0])) == 0.5
    assert metrics["oracle_event_nll"] == pytest.approx(1.5 * math.log(2))
    assert metrics["observed_nll"] == pytest.approx(1.5 * math.log(2))
    assert metrics["event_in_horizon_fraction"] == 0.5
    assert metrics["coverage_0.5"] == 1.0


def test_initial_checkpoint_request_saves_only_for_new_run(tmp_path):
    from tabicl.train._run import Trainer

    trainer = Trainer.__new__(Trainer)
    trainer.config = types.SimpleNamespace(
        save_initial_checkpoint=True, checkpoint_dir=str(tmp_path),
    )
    trainer.curr_step = 0
    trainer.master_process = True
    trainer.ddp = False
    saved = []
    trainer.save_checkpoint = saved.append

    trainer.save_initial_checkpoint_if_requested()
    assert saved == ["step-0.ckpt"]
    (tmp_path / "step-0.ckpt").touch()
    trainer.save_initial_checkpoint_if_requested()
    assert saved == ["step-0.ckpt"]


def test_stage1_pilot_script_defaults_and_chunk_seed_derivation():
    curriculum = open("scripts/train_survival_curriculum.sh").read()
    nibi = open("scripts/train_survival_stage1_nibi.sh").read()
    evaluation = open("scripts/evaluate_survival_stage1_nibi.sh").read()
    vulcan = open("scripts/train_survival_stage1_vulcan.sh").read()
    vulcan_evaluation = open("scripts/evaluate_survival_stage1_vulcan.sh").read()
    assert 'RUN_STAGES="${RUN_STAGES-1}"' in curriculum
    assert 'STAGE1_STEPS="${STAGE1_STEPS:-5000}"' in curriculum
    assert 'STAGE1_SCHEDULER_STEPS="${STAGE1_SCHEDULER_STEPS:-100000}"' in curriculum
    assert 'STAGE1_MICRO_BATCH_SIZE="${STAGE1_MICRO_BATCH_SIZE:-4}"' in curriculum
    assert "TRAIN_FLAGS <<EOF" in curriculum
    assert "TRAIN_FLAGS <<'EOF'" not in curriculum
    assert "--save_initial_checkpoint True" in curriculum
    assert "--save_perm_steps 500,1000,2000" in curriculum
    assert 'STAGE1_TARGET_STEPS="${STAGE1_TARGET_STEPS:-5000}"' in nibi
    assert "NP_SEED=$((BASE_NP_SEED + CURRENT_STEP))" in nibi
    assert "TORCH_SEED=$((BASE_TORCH_SEED + CURRENT_STEP))" in nibi
    assert 'EVAL_STEPS="${EVAL_STEPS:-0,500,1000,2000,5000}"' in evaluation
    assert "#SBATCH --account=aip-qltian" in vulcan
    assert "#SBATCH --gres=gpu:4" in vulcan
    assert "#SBATCH --cpus-per-task=16" in vulcan
    assert "#SBATCH --mem=64G" in vulcan
    assert 'STAGE1_MICRO_BATCH_SIZE="${STAGE1_MICRO_BATCH_SIZE:-2}"' in vulcan
    assert "NP_SEED=$((BASE_NP_SEED + CURRENT_STEP))" in vulcan
    assert "TORCH_SEED=$((BASE_TORCH_SEED + CURRENT_STEP))" in vulcan
    assert 'export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"' in vulcan
    assert "pip install" not in vulcan
    assert "#SBATCH --gres=gpu:1" in vulcan_evaluation
    assert 'TASK_BATCH_SIZE="${TASK_BATCH_SIZE:-2}"' in vulcan_evaluation
    assert "--task-batch-size \"$TASK_BATCH_SIZE\"" in vulcan_evaluation
    assert 'export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"' in vulcan_evaluation
    assert "pip install" not in vulcan_evaluation


def test_exact_checkpoint_milestones_are_permanent():
    from tabicl.train._run import (
        _is_permanent_checkpoint,
        _parse_permanent_checkpoint_steps,
    )

    milestones = _parse_permanent_checkpoint_steps("500,1000,2000")
    assert milestones == {500, 1000, 2000}
    assert _is_permanent_checkpoint(500, 5000, milestones)
    assert _is_permanent_checkpoint(1000, 5000, milestones)
    assert _is_permanent_checkpoint(2000, 5000, milestones)
    assert _is_permanent_checkpoint(5000, 5000, milestones)
    assert not _is_permanent_checkpoint(1500, 5000, milestones)
    with pytest.raises(ValueError, match="duplicates"):
        _parse_permanent_checkpoint_steps("500,500")
    with pytest.raises(ValueError, match="positive"):
        _parse_permanent_checkpoint_steps("0,500")


def test_checkpoint_cleanup_preserves_exact_milestones(tmp_path):
    from tabicl.train._run import Trainer

    for step in [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]:
        (tmp_path / f"step-{step}.ckpt").touch()

    trainer = Trainer.__new__(Trainer)
    trainer.config = types.SimpleNamespace(
        checkpoint_dir=str(tmp_path),
        max_checkpoints=2,
        save_perm_every=5000,
    )
    trainer.permanent_checkpoint_steps = {500, 1000, 2000}
    trainer.manage_checkpoint()

    remaining = {
        int(path.stem.split("-")[1])
        for path in tmp_path.glob("step-*.ckpt")
    }
    assert remaining == {0, 500, 1000, 2000, 4000, 4500, 5000}


def test_holdout_generator_cli_imports_from_repo_root():
    subprocess.run(
        [sys.executable, "scripts/generate_survival_holdout.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )


def test_tiny_holdout_evaluation_end_to_end(tmp_path):
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    checkpoint = tmp_path / "step-0.ckpt"
    output = tmp_path / "results"
    tiny_survival_checkpoint(checkpoint)

    X = torch.randn(4, 8, 3)
    context_t = torch.tensor([1.0, 2.0, 3.0, 4.0]).repeat(4, 1)
    query_t = torch.tensor([1.5, 2.5, 3.5, 4.5]).repeat(4, 1)
    t_event = torch.cat([context_t, query_t], dim=1)
    delta = torch.ones_like(t_event)
    d = torch.full((4,), 3)
    seq_lens = torch.full((4,), 8)
    entry = save_holdout_slice(
        holdout / "00_id.pt",
        (X, t_event, delta, t_event, d, seq_lens, seq_lens),
        task_offset=0,
        group_offset=0,
    )
    manifest = {
        "schema_version": HOLDOUT_SCHEMA_VERSION,
        "holdout_id": "tiny",
        "context_rows": 4,
        "query_rows": 4,
        "task_count": 4,
        "slices": [{
            "filename": "00_id.pt",
            "name": "id",
            "suite": "id",
            "mechanism": "mix",
            "baseline": "mix",
            **entry,
        }],
    }
    manifest["holdout_hash"] = canonical_hash(manifest)
    (holdout / "manifest.json").write_text(json.dumps(manifest))

    subprocess.run([
        sys.executable,
        "scripts/evaluate_survival_holdout.py",
        "--holdout-dir", str(holdout),
        "--checkpoints", str(checkpoint),
        "--output-dir", str(output),
        "--device", "cpu",
        "--task-batch-size", "2",
    ], check=True)

    assert (output / "summary.json").is_file()
    assert (output / "per_task.jsonl").is_file()
    assert (output / "comparison.csv").is_file()
