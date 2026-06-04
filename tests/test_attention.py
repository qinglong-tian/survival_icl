"""Tests for attention backend dtype selection."""

from __future__ import annotations

import torch
from types import SimpleNamespace

from tabicl._model import attention


def test_float32_attention_is_not_eligible_for_fa3(monkeypatch):
    """Float32 attention must remain on the float32-capable SDPA path."""
    monkeypatch.setattr(attention, "HAS_FLASH_ATTN3", True)
    monkeypatch.setattr(attention, "_use_flash_attn3", True)

    q = SimpleNamespace(is_cuda=True, dtype=torch.float32)
    assert not attention._can_use_flash_attn3(q, attn_mask=None, dropout_p=0.0)


def test_bfloat16_attention_is_eligible_for_fa3(monkeypatch):
    monkeypatch.setattr(attention, "HAS_FLASH_ATTN3", True)
    monkeypatch.setattr(attention, "_use_flash_attn3", True)

    q = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16)
    assert attention._can_use_flash_attn3(q, attn_mask=None, dropout_p=0.0)


def test_float32_attention_output_remains_finite():
    q = torch.randn(2, 2, 8, 4, dtype=torch.float32)
    out = attention.sdpa_with_flattened_batch(q, q, q)

    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
