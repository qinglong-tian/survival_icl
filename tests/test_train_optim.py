"""Tests for pretraining optimizer schedules."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tabicl.train._optim import get_scheduler


def _config(*, max_steps: int, scheduler_total_steps: int | None):
    return SimpleNamespace(
        scheduler="cosine_warmup",
        max_steps=max_steps,
        scheduler_total_steps=scheduler_total_steps,
        warmup_proportion=0.02,
        warmup_steps=0,
        cosine_num_cycles=1,
        cosine_amplitude_decay=1.0,
        cosine_lr_end=0.0,
        poly_decay_lr_end=1e-7,
        poly_decay_power=1.0,
    )


def _step(optimizer, scheduler, count: int):
    for _ in range(count):
        optimizer.step()
        scheduler.step()


def test_chunked_scheduler_matches_uninterrupted_schedule():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scheduler = get_scheduler(_config(max_steps=100, scheduler_total_steps=1000), optimizer)
    _step(optimizer, scheduler, 100)

    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()

    uninterrupted_lr = scheduler.get_last_lr()[0]
    _step(optimizer, scheduler, 1)
    uninterrupted_next_lr = scheduler.get_last_lr()[0]

    resumed_parameter = torch.nn.Parameter(torch.tensor(0.0))
    resumed_optimizer = torch.optim.AdamW([resumed_parameter], lr=1e-4)
    resumed_scheduler = get_scheduler(
        _config(max_steps=200, scheduler_total_steps=1000),
        resumed_optimizer,
    )
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)

    assert resumed_scheduler.get_last_lr()[0] == pytest.approx(uninterrupted_lr)
    _step(resumed_optimizer, resumed_scheduler, 1)
    assert resumed_scheduler.get_last_lr()[0] == pytest.approx(uninterrupted_next_lr)


def test_scheduler_horizon_cannot_end_before_training():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)

    with pytest.raises(ValueError, match="must be greater than or equal"):
        get_scheduler(_config(max_steps=100, scheduler_total_steps=50), optimizer)
