"""Tests for survival model inference paths (uncached, rep-cache, KV-cache)."""

from __future__ import annotations

import torch
import pytest

from tabicl._model.tabicl import TabICL
from tabicl._model.learning import ICLearning
from tabicl.survival import TimeBinner, discrete_survival_nll


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_survival_model():
    """Build a tiny TabICL model with survival=True for fast testing."""
    return TabICL(
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


@pytest.fixture
def tiny_regression_model():
    return TabICL(
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
        survival=False,
    )


@pytest.fixture
def dummy_batch():
    B, T, H = 2, 8, 5
    X = torch.randn(B, T, H)
    y_train = torch.rand(B, T // 2) * 10  # (B, 4) raw times
    delta_train = torch.randint(0, 2, (B, T // 2)).float()
    return X, y_train, delta_train


# ---------------------------------------------------------------------------
# Training with delta_train
# ---------------------------------------------------------------------------


def test_training_accepts_delta_train(tiny_survival_model, dummy_batch):
    X, y_train, delta_train = dummy_batch
    tiny_survival_model.train()
    h_raw = tiny_survival_model(X, y_train, delta_train=delta_train)
    assert h_raw.shape == (2, X.shape[1] // 2, 10)  # (B, test_size, K)
    assert torch.isfinite(h_raw).all()


def test_training_missing_delta_raises(tiny_survival_model, dummy_batch):
    X, y_train, _ = dummy_batch
    tiny_survival_model.train()
    with pytest.raises(AssertionError, match="delta_train"):
        tiny_survival_model(X, y_train)


# ---------------------------------------------------------------------------
# Eval with delta_train
# ---------------------------------------------------------------------------


def test_eval_accepts_delta_train(tiny_survival_model, dummy_batch):
    X, y_train, delta_train = dummy_batch
    tiny_survival_model.eval()
    h_raw = tiny_survival_model(X, y_train, delta_train=delta_train)
    assert h_raw.shape == (2, X.shape[1] // 2, 10)
    assert torch.isfinite(h_raw).all()


def test_eval_missing_delta_raises(tiny_survival_model, dummy_batch):
    X, y_train, _ = dummy_batch
    tiny_survival_model.eval()
    with pytest.raises(AssertionError, match="delta_train"):
        tiny_survival_model(X, y_train)


# ---------------------------------------------------------------------------
# predict_stats raises for survival
# ---------------------------------------------------------------------------


def test_predict_stats_raises_for_survival(tiny_survival_model, dummy_batch):
    X, y_train, delta_train = dummy_batch
    tiny_survival_model.eval()
    with pytest.raises(RuntimeError, match="not supported for survival"):
        tiny_survival_model.predict_stats(X, y_train)


# ---------------------------------------------------------------------------
# configure_survival
# ---------------------------------------------------------------------------


def test_configure_survival_from_regression(tiny_regression_model):
    model = tiny_regression_model
    assert not model.survival
    assert not model.icl_predictor.survival

    model.configure_survival(num_bins=10)
    assert model.survival
    assert model.icl_predictor.survival
    # Verify decoder has survival head structure (Linear→GELU→Linear inside head.Sequential)
    dec = model.icl_predictor.decoder
    assert hasattr(dec, "head") and isinstance(dec.head, torch.nn.Sequential)
    assert len(dec.head) == 3
    assert dec.num_bins == 10
    assert model.icl_predictor.inference_mgr.out_dim == 10
    assert model.num_quantiles == 10
    assert model.max_classes == 0

    # Should accept delta_train after conversion
    X = torch.randn(2, 8, 5)
    y_train = torch.rand(2, 4) * 10
    delta_train = torch.ones(2, 4)
    model.train()
    h_raw = model(X, y_train, delta_train=delta_train)
    assert h_raw.shape == (2, 4, 10)


# ---------------------------------------------------------------------------
# ICLearning direct construction
# ---------------------------------------------------------------------------


def test_iclearning_survival_builds_head_directly():
    icl = ICLearning(
        max_classes=0,
        out_dim=20,
        d_model=64,
        num_blocks=1,
        nhead=2,
        dim_feedforward=128,
        survival=True,
    )
    assert icl.survival
    assert icl.y_encoder.weight.shape[0] == 64
    assert icl.y_encoder.weight.shape[1] == 2  # (t, delta) channels
    # Head should be DiscreteTimeSurvivalHead, not sequential
    from tabicl.survival._head import DiscreteTimeSurvivalHead
    assert isinstance(icl.decoder, DiscreteTimeSurvivalHead)
    assert icl.decoder.num_bins == 20


def test_iclearning_non_survival_unchanged():
    icl = ICLearning(
        max_classes=0,
        out_dim=20,
        d_model=64,
        num_blocks=1,
        nhead=2,
        dim_feedforward=128,
        survival=False,
    )
    assert not icl.survival
    # Should be standard sequential decoder
    assert isinstance(icl.decoder, torch.nn.Sequential)
    assert icl.y_encoder.weight.shape[1] == 1  # single channel regression


# ---------------------------------------------------------------------------
# Cache agreement
# ---------------------------------------------------------------------------


def test_uncached_repr_cache_agree(tiny_survival_model, dummy_batch):
    """Uncached and representation-cache paths produce identical logits."""
    X, y_train, delta_train = dummy_batch
    tiny_survival_model.eval()

    # Only one dataset per batch for rep-cache simplicity
    X1 = X[:1]
    y1 = y_train[:1]
    d1 = delta_train[:1]

    # Uncached
    h_uncached = tiny_survival_model(X1, y1, delta_train=d1)

    # Rep-cache
    with torch.no_grad():
        # Build representations
        rep = tiny_survival_model.row_interactor(
            tiny_survival_model.col_embedder(X1, y_train=y1),
        )
        # Bake in y_train
        rep = tiny_survival_model.icl_predictor.prepare_repr_cache(
            rep, y1, delta_train=d1,
        )
        h_cached = tiny_survival_model.icl_predictor.forward_with_repr_cache(
            rep, train_size=y1.shape[1],
        )

    assert torch.allclose(h_uncached, h_cached, atol=1e-4)


def test_prepare_repr_cache_survival_requires_delta(tiny_survival_model, dummy_batch):
    X, y_train, delta_train = dummy_batch
    tiny_survival_model.eval()

    X1 = X[:1]
    y1 = y_train[:1]
    rep = tiny_survival_model.row_interactor(
        tiny_survival_model.col_embedder(X1, y_train=y1),
    )
    with pytest.raises(AssertionError, match="delta_train"):
        tiny_survival_model.icl_predictor.prepare_repr_cache(rep, y1)


# ---------------------------------------------------------------------------
# scale_survival_context
# ---------------------------------------------------------------------------


def test_scale_survival_context_returns_correct_shapes():
    from tabicl.survival import scale_survival_context
    t_ctx = torch.tensor([[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]])
    delta_ctx = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    z, d, scalers = scale_survival_context(t_ctx, delta_ctx)
    assert z.shape == t_ctx.shape
    assert d.shape == delta_ctx.shape
    assert len(scalers) == 2
    assert torch.isfinite(z).all()
    assert ((z >= -6.0) & (z <= 6.0)).all()


# ---------------------------------------------------------------------------
# Public forward_with_cache: repr and KV modes
# ---------------------------------------------------------------------------


def _forward_with_cache_agreement(model, X, y_train, delta_train, cache_mode):
    """Verify store-then-use produces identical logits to uncached."""
    model.eval()
    B = X.shape[0]
    train_size = y_train.shape[1]
    X_train = X[:, :train_size]
    X_test = X[:, train_size:]

    # Uncached
    with torch.no_grad():
        h_uncached = model(X, y_train, delta_train=delta_train)

    # Store cache
    model.clear_cache()
    with torch.no_grad():
        model.forward_with_cache(
            X_train=X_train, y_train=y_train, X_test=X_test,
            store_cache=True, use_cache=False, cache_mode=cache_mode,
            delta_train=delta_train,
        )

    # Use cache
    with torch.no_grad():
        h_cached = model.forward_with_cache(
            X_test=X_test, store_cache=False, use_cache=True,
        )

    assert torch.allclose(h_uncached, h_cached, atol=1e-4), (
        f"Uncached and {cache_mode}-cache outputs differ: "
        f"max diff = {(h_uncached - h_cached).abs().max().item()}"
    )


def test_repr_cache_survival_agreement(tiny_survival_model, dummy_batch):
    """Public forward_with_cache with cache_mode='repr' agrees with uncached."""
    X, y_train, delta_train = dummy_batch
    _forward_with_cache_agreement(
        tiny_survival_model, X, y_train, delta_train, cache_mode="repr",
    )


def test_kv_cache_survival_agreement(tiny_survival_model, dummy_batch):
    """Public forward_with_cache with cache_mode='kv' agrees with uncached."""
    X, y_train, delta_train = dummy_batch
    _forward_with_cache_agreement(
        tiny_survival_model, X, y_train, delta_train, cache_mode="kv",
    )


def test_external_cache_survival(tiny_survival_model, dummy_batch):
    """Passing an external TabICLCache works for survival."""
    from tabicl._model.kv_cache import TabICLCache

    X, y_train, delta_train = dummy_batch
    model = tiny_survival_model
    model.eval()
    B = X.shape[0]
    train_size = y_train.shape[1]
    X_train = X[:, :train_size]
    X_test = X[:, train_size:]

    # Store to external cache
    ext_cache = TabICLCache(train_shape=X_train.shape, num_classes=0, survival=True)
    with torch.no_grad():
        model.forward_with_cache(
            X_train=X_train, y_train=y_train, X_test=X_test,
            store_cache=True, use_cache=False, cache_mode="kv",
            delta_train=delta_train,
        )
    # Grab internal cache
    internal_cache = model._cache

    # Use via external cache parameter
    model.clear_cache()
    with torch.no_grad():
        h_ext = model.forward_with_cache(
            X_test=X_test, cache=internal_cache,
        )

    # Uncached reference
    with torch.no_grad():
        h_ref = model(X, y_train, delta_train=delta_train)

    assert torch.allclose(h_ref, h_ext, atol=1e-4)


def test_cache_task_mismatch_raises(tiny_survival_model, tiny_regression_model, dummy_batch):
    """Using a survival cache with a regression model (or vice versa) raises."""
    from tabicl._model.kv_cache import TabICLCache

    X, y_train, delta_train = dummy_batch
    train_size = y_train.shape[1]
    X_train = X[:, :train_size]
    X_test = X[:, train_size:]

    # Store survival cache
    tiny_survival_model.eval()
    with torch.no_grad():
        tiny_survival_model.forward_with_cache(
            X_train=X_train, y_train=y_train, X_test=X_test,
            store_cache=True, use_cache=False, cache_mode="kv",
            delta_train=delta_train,
        )
    surv_cache = tiny_survival_model._cache

    # Using survival cache with regression model should raise
    tiny_regression_model.eval()
    tiny_regression_model._cache = surv_cache
    with pytest.raises(ValueError, match="task mismatch"):
        tiny_regression_model.forward_with_cache(
            X_test=X_test, store_cache=False, use_cache=True,
        )


# ---------------------------------------------------------------------------
# scale_survival_context argument validation
# ---------------------------------------------------------------------------


def test_scale_survival_context_rejects_nan_eps():
    from tabicl.survival import scale_survival_context
    t_ctx = torch.tensor([[1.0, 2.0, 4.0]])
    d_ctx = torch.tensor([[1.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="eps"):
        scale_survival_context(t_ctx, d_ctx, eps=float("nan"))


def test_scale_survival_context_rejects_reversed_bounds():
    from tabicl.survival import scale_survival_context
    t_ctx = torch.tensor([[1.0, 2.0, 4.0]])
    d_ctx = torch.tensor([[1.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="z_min.*z_max"):
        scale_survival_context(t_ctx, d_ctx, z_min=6.0, z_max=-6.0)
