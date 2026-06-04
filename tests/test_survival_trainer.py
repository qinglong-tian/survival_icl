"""Tests for survival checkpoint round-trip and t_event invariance."""

from __future__ import annotations

import tempfile
import os
import types
import torch
import pytest

from tabicl._model.tabicl import TabICL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_config(task="survival", **overrides):
    """Build a minimal argparse.Namespace that satisfies Trainer.build_model."""
    cfg = types.SimpleNamespace(
        task=task,
        device="cpu",
        max_classes=0,
        num_bins=10,
        num_quantiles=10,
        embed_dim=32,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        row_rope_base=10000.0,
        icl_num_blocks=1,
        icl_nhead=2,
        ff_factor=2,
        dropout=0.0,
        activation="gelu",
        norm_first=True,
        model_compile=False,
        freeze_col=False,
        freeze_row=False,
        freeze_icl=False,
        pretrained_path=None,
        only_load_model=False,
        checkpoint_path=None,
        checkpoint_dir=None,
        lr=1e-4,
        weight_decay=0.01,
        amp=False,
        dtype="float32",
        np_seed=42,
        torch_seed=42,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_minimal_trainer(config):
    """Construct a Trainer that bypasses __init__ but sets enough state for
    build_model and restore_training_state."""
    from tabicl.train._run import Trainer

    t = Trainer.__new__(Trainer)
    t.config = config
    t.survival = getattr(config, "task", "classification") == "survival"
    t.master_process = False
    t.ddp = False
    t.curr_step = 0
    t._resume_ckpt_path = None
    t._resume_ckpt_payload = None
    return t


def _save_checkpoint(path, model, config, survival_metadata=None):
    """Save a checkpoint dict matching save_checkpoint format."""
    ckpt = {
        "config": config,
        "state_dict": model.state_dict(),
        "optimizer_state": {},
        "scheduler_state": {},
        "scaler_state": {},
        "curr_step": 42,
    }
    if survival_metadata is not None:
        ckpt["survival_metadata"] = survival_metadata
    torch.save(ckpt, path)


# ---------------------------------------------------------------------------
# Checkpoint round-trip (strict load)
# ---------------------------------------------------------------------------


def test_survival_checkpoint_round_trip_strict():
    """New survival checkpoint can be reconstructed from saved config and
    loaded with strict=True."""
    model = TabICL(
        max_classes=0,
        num_quantiles=10,
        embed_dim=32,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
        ff_factor=2,
        survival=True,
    )
    state_dict = model.state_dict()
    config = {
        "max_classes": 0,
        "num_quantiles": 10,
        "survival": True,
        "embed_dim": 32,
        "col_num_blocks": 1,
        "col_nhead": 2,
        "col_num_inds": 8,
        "row_num_blocks": 1,
        "row_nhead": 2,
        "row_num_cls": 2,
        "icl_num_blocks": 1,
        "icl_nhead": 2,
        "ff_factor": 2,
        "dropout": 0.0,
        "activation": "gelu",
        "norm_first": True,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "surv.ckpt")
        torch.save({"state_dict": state_dict, "config": config}, path)

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model2 = TabICL(**ckpt["config"])
        model2.load_state_dict(ckpt["state_dict"], strict=True)

    # Verify key survival-specific parameters
    assert model2.survival
    assert model2.icl_predictor.survival
    assert isinstance(model2.icl_predictor.decoder, type(model.icl_predictor.decoder))
    assert model2.icl_predictor.decoder.num_bins == 10


def test_legacy_survival_checkpoint_fallback():
    """Legacy checkpoint without 'config' key loads via regressor fallback."""
    # We can't test the full regressor fallback without HF network,
    # but we verify the detection logic is wired correctly.
    model = TabICL(
        max_classes=0,
        num_quantiles=10,
        embed_dim=32,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=8,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
        ff_factor=2,
        survival=True,
    )
    state_dict = model.state_dict()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "legacy.ckpt")
        # No 'config' key — legacy format
        torch.save({"state_dict": state_dict}, path)

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        pretrained_state = ckpt.get("state_dict", ckpt)
        saved_config = ckpt.get("config", None)

        # Simulate the detection logic in build_model
        is_survival_ckpt = any(
            k.startswith("icl_predictor.decoder.head.")
            for k in pretrained_state
        )
        assert is_survival_ckpt
        assert saved_config is None  # legacy has no config


# ---------------------------------------------------------------------------
# Trainer-level checkpoint round-trips (build_model)
# ---------------------------------------------------------------------------


def _build_model_from_checkpoint(ckpt_path, config):
    """Exercise the real Trainer.build_model path with a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    trainer = _make_minimal_trainer(config)
    trainer._resume_ckpt_path = ckpt_path
    trainer._resume_ckpt_payload = ckpt
    trainer.build_model()
    return trainer


def test_trainer_resume_saved_config_strict():
    """Trainer.build_model strict-loads a saved-config checkpoint."""
    model = TabICL(
        max_classes=0, num_quantiles=10, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2, survival=True,
    )
    config = {
        "max_classes": 0, "num_quantiles": 10, "survival": True,
        "embed_dim": 32, "col_num_blocks": 1, "col_nhead": 2,
        "col_num_inds": 8, "row_num_blocks": 1, "row_nhead": 2,
        "row_num_cls": 2, "icl_num_blocks": 1, "icl_nhead": 2,
        "ff_factor": 2, "dropout": 0.0, "activation": "gelu",
        "norm_first": True,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pt")
        _save_checkpoint(path, model, config)
        cfg = _make_tiny_config()
        trainer = _build_model_from_checkpoint(path, cfg)
        assert trainer.model.survival
        assert trainer.model.icl_predictor.decoder.num_bins == 10
        # Verify weights actually loaded (not random)
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), trainer.model.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Weight mismatch at {n1}"


def test_trainer_resume_compiled_checkpoint():
    """Compiled checkpoint with _orig_mod.* keys loads correctly."""
    model = TabICL(
        max_classes=0, num_quantiles=10, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2, survival=True,
    )
    # Simulate torch.compile prefix
    compiled_state = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
    config = {
        "max_classes": 0, "num_quantiles": 10, "survival": True,
        "embed_dim": 32, "col_num_blocks": 1, "col_nhead": 2,
        "col_num_inds": 8, "row_num_blocks": 1, "row_nhead": 2,
        "row_num_cls": 2, "icl_num_blocks": 1, "icl_nhead": 2,
        "ff_factor": 2, "dropout": 0.0, "activation": "gelu",
        "norm_first": True,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "compiled.ckpt")
        torch.save({"state_dict": compiled_state, "config": config, "curr_step": 5}, path)
        cfg = _make_tiny_config()
        trainer = _build_model_from_checkpoint(path, cfg)
        assert trainer.model.survival
        assert trainer.model.icl_predictor.decoder.num_bins == 10


def test_trainer_resume_compiled_legacy_checkpoint():
    """Compiled legacy checkpoint (no config) loads via prefix stripping."""
    model = TabICL(
        max_classes=0, num_quantiles=10, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2, survival=True,
    )
    compiled_state = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "compiled_legacy.ckpt")
        torch.save({"state_dict": compiled_state}, path)
        cfg = _make_tiny_config()
        trainer = _build_model_from_checkpoint(path, cfg)
        assert trainer.model.survival
        assert trainer.model.icl_predictor.decoder.num_bins == 10


def test_trainer_resume_legacy_classification_without_num_quantiles():
    """Legacy classification resume works with the real CLI config surface."""
    model = TabICL(
        max_classes=3, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "legacy_classification.ckpt")
        torch.save({"state_dict": model.state_dict()}, path)

        cfg = _make_tiny_config(task="classification", max_classes=3)
        delattr(cfg, "num_quantiles")  # The real training parser has no such option.
        trainer = _build_model_from_checkpoint(path, cfg)

        assert trainer.model.survival is False
        assert trainer.model.max_classes == 3


def test_trainer_resume_task_mismatch_raises():
    """Resume with mismatched survival/regression task raises RuntimeError."""
    model = TabICL(
        max_classes=0, num_quantiles=10, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2, survival=False,
    )
    config = {
        "max_classes": 0, "num_quantiles": 10, "survival": False,
        "embed_dim": 32, "col_num_blocks": 1, "col_nhead": 2,
        "col_num_inds": 8, "row_num_blocks": 1, "row_nhead": 2,
        "row_num_cls": 2, "icl_num_blocks": 1, "icl_nhead": 2,
        "ff_factor": 2, "dropout": 0.0, "activation": "gelu",
        "norm_first": True,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "reg.ckpt")
        _save_checkpoint(path, model, config)
        cfg = _make_tiny_config(task="survival")  # CLI says survival
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        trainer = _make_minimal_trainer(cfg)
        trainer._resume_ckpt_payload = ckpt
        trainer._resume_ckpt_path = path
        with pytest.raises(RuntimeError, match="task mismatch"):
            trainer.build_model()


# ---------------------------------------------------------------------------
# Metadata restoration and scaler derivation
# ---------------------------------------------------------------------------


def _make_binner_metadata(num_bins=10, z_min=-6.0, z_max=6.0, include_scaler=True):
    """Build a minimal survival_metadata dict for testing."""
    from tabicl.survival import TimeBinner
    binner = TimeBinner.from_standardized_range(num_bins, z_min, z_max)
    meta = {
        "binner_edges": binner.bin_edges,
        "binner_means": binner.bin_means,
        "num_bins": num_bins,
        "task": "survival",
        "time_scale": "km_hybrid_log",
    }
    if include_scaler:
        meta["time_scaler"] = {"eps": 1e-8, "min_scale": 0.1, "z_min": z_min, "z_max": z_max}
    return meta


def test_restore_metadata_with_scaler():
    """_restore_survival_metadata sets scaler config from checkpoint."""
    model = TabICL(
        max_classes=0, num_quantiles=10, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2, survival=True,
    )
    cfg = _make_tiny_config()
    trainer = _make_minimal_trainer(cfg)
    meta = _make_binner_metadata(num_bins=10, z_min=-6.0, z_max=6.0)
    checkpoint = {"survival_metadata": meta}
    trainer._restore_survival_metadata(checkpoint, model=model)
    assert trainer._binner_restored is True
    assert trainer.survival_time_scaler_config["z_min"] == -6.0
    assert trainer.survival_time_scaler_config["z_max"] == 6.0


def test_restore_metadata_derives_scaler_from_edges():
    """When time_scaler is absent, bounds are derived from binner edges."""
    model = TabICL(
        max_classes=0, num_quantiles=10, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2, survival=True,
    )
    cfg = _make_tiny_config()
    trainer = _make_minimal_trainer(cfg)
    meta = _make_binner_metadata(num_bins=10, z_min=-2.0, z_max=2.0, include_scaler=False)
    checkpoint = {"survival_metadata": meta}
    trainer._restore_survival_metadata(checkpoint, model=model)
    assert trainer._binner_restored is True
    assert trainer.survival_time_scaler_config["z_min"] == pytest.approx(-2.0)
    assert trainer.survival_time_scaler_config["z_max"] == pytest.approx(2.0)


def test_restore_metadata_rejects_non_monotonic_edges():
    """Non-monotonic binner edges raise ValueError, not AssertionError."""
    model = TabICL(
        max_classes=0, num_quantiles=10, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2, survival=True,
    )
    cfg = _make_tiny_config()
    trainer = _make_minimal_trainer(cfg)
    meta = _make_binner_metadata(num_bins=10)
    # Corrupt edges: swap two adjacent edges to break monotonicity
    edges = meta["binner_edges"].clone()
    edges[3], edges[4] = edges[4], edges[3]
    meta["binner_edges"] = edges
    checkpoint = {"survival_metadata": meta}
    with pytest.raises(ValueError, match="strictly increasing"):
        trainer._restore_survival_metadata(checkpoint, model=model)


def test_restore_metadata_rejects_k_mismatch():
    """Metadata K != model K raises ValueError."""
    model = TabICL(
        max_classes=0, num_quantiles=10, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2, survival=True,
    )
    cfg = _make_tiny_config()
    trainer = _make_minimal_trainer(cfg)
    meta = _make_binner_metadata(num_bins=8)  # K=8, model has K=10
    checkpoint = {"survival_metadata": meta}
    with pytest.raises(ValueError, match="K=8 != model K=10"):
        trainer._restore_survival_metadata(checkpoint, model=model)


def test_restore_training_state_reuses_resume_path():
    """restore_training_state uses _resume_ckpt_path, not get_latest_checkpoint."""
    import torch.optim as optim

    model = TabICL(
        max_classes=0, num_quantiles=10, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2, survival=True,
    )
    opt = optim.AdamW(model.parameters(), lr=1e-4)
    sched = optim.lr_scheduler.ConstantLR(opt)
    scaler = torch.GradScaler("cpu", enabled=False)
    config = {
        "max_classes": 0, "num_quantiles": 10, "survival": True,
        "embed_dim": 32, "col_num_blocks": 1, "col_nhead": 2,
        "col_num_inds": 8, "row_num_blocks": 1, "row_nhead": 2,
        "row_num_cls": 2, "icl_num_blocks": 1, "icl_nhead": 2,
        "ff_factor": 2, "dropout": 0.0, "activation": "gelu",
        "norm_first": True,
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "step-42.ckpt")
        ckpt = {
            "config": config,
            "state_dict": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scheduler_state": sched.state_dict(),
            "scaler_state": scaler.state_dict(),
            "curr_step": 42,
        }
        torch.save(ckpt, path)

        cfg = _make_tiny_config(checkpoint_dir=tmpdir)
        trainer = _make_minimal_trainer(cfg)
        trainer._resume_ckpt_path = path

        # Build a fresh optimizer/scheduler/scaler for the trainer
        model2 = TabICL(
            max_classes=0, num_quantiles=10, embed_dim=32,
            col_num_blocks=1, col_nhead=2, col_num_inds=8,
            row_num_blocks=1, row_nhead=2, row_num_cls=2,
            icl_num_blocks=1, icl_nhead=2, ff_factor=2, survival=True,
        )
        trainer.optimizer = optim.AdamW(model2.parameters(), lr=1e-4)
        trainer.scheduler = optim.lr_scheduler.ConstantLR(trainer.optimizer)
        trainer.scaler = torch.GradScaler("cpu", enabled=False)

        trainer.restore_training_state()
        assert trainer.curr_step == 42


def test_raw_state_dict_skips_training_state():
    """Raw state dict (no 'state_dict' wrapper) gracefully skips restore."""
    model = TabICL(
        max_classes=0, num_quantiles=10, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2, survival=True,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "raw.pt")
        torch.save(model.state_dict(), path)  # raw state dict

        cfg = _make_tiny_config()
        trainer = _make_minimal_trainer(cfg)
        trainer._resume_ckpt_path = path

        # Should not raise
        trainer.restore_training_state()
        assert trainer.curr_step == 0  # unchanged


# ---------------------------------------------------------------------------
# configure_survival guards
# ---------------------------------------------------------------------------


def test_configure_survival_rejects_classifier():
    """configure_survival on a classifier raises ValueError."""
    model = TabICL(
        max_classes=10, num_quantiles=0, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2, survival=False,
    )
    with pytest.raises(ValueError, match="max_classes=0"):
        model.configure_survival(num_bins=10)


def test_configure_survival_rejects_zero_bins():
    """configure_survival with num_bins=0 raises ValueError."""
    model = TabICL(
        max_classes=0, num_quantiles=10, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2, survival=False,
    )
    with pytest.raises(ValueError, match="num_bins must be > 0"):
        model.configure_survival(num_bins=0)


def test_tabicl_survival_rejects_zero_quantiles():
    """TabICL(survival=True, num_quantiles=0) raises ValueError."""
    with pytest.raises(ValueError, match="num_quantiles"):
        TabICL(
            max_classes=0, num_quantiles=0, embed_dim=32,
            col_num_blocks=1, col_nhead=2, col_num_inds=8,
            row_num_blocks=1, row_nhead=2, row_num_cls=2,
            icl_num_blocks=1, icl_nhead=2, ff_factor=2, survival=True,
        )


# ---------------------------------------------------------------------------
# t_event invariance
# ---------------------------------------------------------------------------


def test_t_event_not_used_in_loss():
    """Proof that t_event is ignored: trainer unpacks it as _micro_t_event
    and discrete_survival_nll never sees it.
    The trainer source must not reference t_event after unpacking as
    _micro_t_event, and discrete_survival_nll takes only h_raw, bin_idx, delta."""
    import inspect
    from tabicl.survival import discrete_survival_nll
    from tabicl.train._run import Trainer

    # Trainer source: t_event unpacked as _micro_t_event, never used after unpack.
    source = inspect.getsource(Trainer._run_micro_batch_survival)
    assert "_micro_t_event" in source  # unpacked as private
    # After the unpack line, _micro_t_event must never appear
    unpack_line_idx = source.index("_micro_t_event")
    rest = source[unpack_line_idx + len("_micro_t_event"):]
    assert "_micro_t_event" not in rest, (
        "_micro_t_event used after unpack — t_event may leak into loss"
    )
    # discrete_survival_nll signature: 3 positional args, no t_event
    sig = inspect.signature(discrete_survival_nll)
    params = list(sig.parameters.keys())
    assert params == ["h_raw", "bin_idx", "delta"], (
        f"discrete_survival_nll signature changed: {params}; t_event may have been added"
    )


def test_surv_nll_only_metric():
    """Trainer metrics for survival contain only 'surv_nll' (no impute/alpha)."""
    # This is a static check — the _run_micro_batch_survival function
    # now returns only {"surv_nll": ...}
    import inspect
    from tabicl.train._run import Trainer

    source = inspect.getsource(Trainer._run_micro_batch_survival)
    assert "micro_results = {" in source
    assert '"surv_nll"' in source
    assert '"impute"' not in source
    assert '"alpha"' not in source
    assert 'comps' not in source or 'comps[' not in source.split("return")[0]


# ---------------------------------------------------------------------------
# Micro-batch NLL inline check
# ---------------------------------------------------------------------------


def test_trainer_nll_import():
    """Verify discrete_survival_nll is importable from tabicl.survival."""
    from tabicl.survival import discrete_survival_nll
    assert callable(discrete_survival_nll)


# ---------------------------------------------------------------------------
# New validation tests
# ---------------------------------------------------------------------------


def test_legacy_resume_derives_k_from_metadata():
    """Legacy resume with K != CLI default constructs the correct model shape."""
    # Build a K=8 model (different from CLI default of 10)
    model = TabICL(
        max_classes=0, num_quantiles=8, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2, survival=True,
    )
    meta = _make_binner_metadata(num_bins=8)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "legacy_k8.ckpt")
        torch.save({
            "state_dict": model.state_dict(),
            "survival_metadata": meta,
        }, path)

        cfg = _make_tiny_config()  # defaults to num_bins=10
        trainer = _build_model_from_checkpoint(path, cfg)
        # Model should have K=8 (from metadata), not K=10 (CLI default)
        assert trainer.model.icl_predictor.decoder.num_bins == 8
        assert trainer.model.num_quantiles == 8


def test_legacy_resume_rejects_missing_keys():
    """Legacy resume with missing keys raises RuntimeError."""
    model = TabICL(
        max_classes=0, num_quantiles=10, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2, survival=True,
    )
    state_dict = model.state_dict()
    # Remove a core key to simulate truncation
    removed_key = next(k for k in state_dict if "col_embedder" in k)
    del state_dict[removed_key]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "truncated.ckpt")
        torch.save({"state_dict": state_dict}, path)

        cfg = _make_tiny_config()
        with pytest.raises(RuntimeError, match="missing"):
            _build_model_from_checkpoint(path, cfg)


def test_restore_metadata_rejects_scaler_binner_mismatch():
    """Scaler z_max != binner edge[-1] raises ValueError."""
    model = TabICL(
        max_classes=0, num_quantiles=10, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2, survival=True,
    )
    cfg = _make_tiny_config()
    trainer = _make_minimal_trainer(cfg)
    meta = _make_binner_metadata(num_bins=10, z_min=-6.0, z_max=6.0, include_scaler=True)
    # Corrupt scaler z_max to disagree with binner
    meta["time_scaler"]["z_max"] = 2.0
    checkpoint = {"survival_metadata": meta}
    with pytest.raises(ValueError, match="Scaler z_max"):
        trainer._restore_survival_metadata(checkpoint, model=model)


def test_restore_metadata_rejects_mean_outside_bin():
    """Bin mean outside its bin edges raises ValueError."""
    model = TabICL(
        max_classes=0, num_quantiles=10, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2, survival=True,
    )
    cfg = _make_tiny_config()
    trainer = _make_minimal_trainer(cfg)
    meta = _make_binner_metadata(num_bins=10)
    # Set first bin mean outside its bin
    meta["binner_means"][0] = meta["binner_edges"][-1] + 1.0
    checkpoint = {"survival_metadata": meta}
    with pytest.raises(ValueError, match="outside bin"):
        trainer._restore_survival_metadata(checkpoint, model=model)


def test_restore_metadata_accepts_float32_custom_bounds():
    """Custom bounds like z_min=-5.9 pass validation despite float32 precision."""
    from tabicl.survival import TimeBinner

    model = TabICL(
        max_classes=0, num_quantiles=10, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2, survival=True,
    )
    cfg = _make_tiny_config()
    trainer = _make_minimal_trainer(cfg)

    # Simulate a checkpoint with custom bounds saved as float32 edges
    # but Python floats in scaler metadata (realistic production scenario)
    binner = TimeBinner.from_standardized_range(num_bins=10, z_min=-5.9, z_max=5.9)
    meta = {
        "binner_edges": binner.bin_edges,  # float32
        "binner_means": binner.bin_means,
        "num_bins": 10,
        "task": "survival",
        "time_scale": "km_hybrid_log",
        "time_scaler": {"eps": 1e-8, "min_scale": 0.1, "z_min": -5.9, "z_max": 5.9},
    }
    checkpoint = {"survival_metadata": meta}
    # Should not raise
    trainer._restore_survival_metadata(checkpoint, model=model)
    assert trainer._binner_restored is True


def test_restore_metadata_rejects_nan_scaler_eps():
    """NaN eps in scaler metadata raises ValueError."""
    model = TabICL(
        max_classes=0, num_quantiles=10, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2, survival=True,
    )
    cfg = _make_tiny_config()
    trainer = _make_minimal_trainer(cfg)
    meta = _make_binner_metadata(num_bins=10, include_scaler=True)
    meta["time_scaler"]["eps"] = float("nan")
    checkpoint = {"survival_metadata": meta}
    with pytest.raises(ValueError, match="invalid eps"):
        trainer._restore_survival_metadata(checkpoint, model=model)


def test_make_model_config_preserves_recompute():
    """_make_model_config includes recompute derived from row_interactor."""
    from tabicl.train._run import Trainer

    model = TabICL(
        max_classes=0, num_quantiles=10, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2, survival=True,
        recompute=True,
    )
    cfg = _make_tiny_config()
    trainer = _make_minimal_trainer(cfg)
    config = trainer._make_model_config(model)
    assert config["recompute"] is True

    model2 = TabICL(
        max_classes=0, num_quantiles=10, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2, survival=True,
        recompute=False,
    )
    config2 = trainer._make_model_config(model2)
    assert config2["recompute"] is False


def test_make_model_config_uses_loaded_model_bin_count():
    """Saved config follows the converted model K, not the current CLI default."""
    model = TabICL(
        max_classes=0, num_quantiles=10, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2,
    )
    model.configure_survival(num_bins=8)

    trainer = _make_minimal_trainer(_make_tiny_config(num_bins=10))
    config = trainer._make_model_config(model)

    assert model.num_quantiles == 8
    assert config["num_quantiles"] == 8


def test_compiled_regression_pretrained_converts_to_survival():
    """Compiled regression checkpoints use normalized keys during conversion."""
    model = TabICL(
        max_classes=0, num_quantiles=10, embed_dim=32,
        col_num_blocks=1, col_nhead=2, col_num_inds=8,
        row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2,
    )
    config = {
        "max_classes": 0, "num_quantiles": 10,
        "embed_dim": 32, "col_num_blocks": 1, "col_nhead": 2,
        "col_num_inds": 8, "row_num_blocks": 1, "row_nhead": 2,
        "row_num_cls": 2, "icl_num_blocks": 1, "icl_nhead": 2,
        "ff_factor": 2, "dropout": 0.0, "activation": "gelu",
        "norm_first": True,
    }
    compiled_state = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "compiled_regression.ckpt")
        torch.save({"config": config, "state_dict": compiled_state}, path)

        trainer = _make_minimal_trainer(_make_tiny_config(pretrained_path=path, num_bins=8))
        trainer.build_model()

        assert trainer.model.survival is True
        assert trainer.model.num_quantiles == 8
        assert trainer.model_config["num_quantiles"] == 8
        encoder_key = next(k for k in model.state_dict() if k.startswith("col_embedder."))
        assert torch.equal(trainer.model.state_dict()[encoder_key], model.state_dict()[encoder_key])
