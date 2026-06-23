from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_real_quality_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "compare_real_imputation_quality.py"
    script_dir = str(path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location("compare_real_imputation_quality", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


real_quality = _load_real_quality_module()


def test_artificial_censor_times_are_strictly_before_events():
    rng = np.random.default_rng(123)
    events = np.array([10.0, 20.0, 30.0], dtype=float)
    natural_censors = np.array([3.0, 7.0, 12.0, 18.0], dtype=float)

    censors = real_quality.artificial_censor_times(
        events,
        natural_censors,
        rng,
        strategy="empirical",
        fraction_low=0.2,
        fraction_high=0.8,
    )

    assert censors.shape == events.shape
    assert np.all(censors > 0.0)
    assert np.all(censors < events)


def test_choose_holdout_event_indices_leaves_context_events():
    rng = np.random.default_rng(123)
    event = np.r_[np.ones(20), np.zeros(10)]

    holdout = real_quality.choose_holdout_event_indices(
        event,
        rng,
        holdout_fraction=0.5,
        max_holdout_events=8,
        min_holdout_events=4,
        min_context_events=10,
    )

    assert holdout.size == 8
    assert np.all(event[holdout] == 1.0)
    assert int(event.sum()) - holdout.size >= 10


def test_real_imputation_quality_smoke_with_local_cache(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    n = 32
    time = np.linspace(2.0, 33.0, n)
    event = np.ones(n, dtype=int)
    event[-8:] = 0
    frame = pd.DataFrame(
        {
            "time": time,
            "event": event,
            "x": np.linspace(-1.0, 1.0, n),
            "group": np.where(np.arange(n) % 2 == 0, "a", "b"),
        }
    )
    frame.to_csv(processed / "veteran.csv", index=False)

    config = real_quality.RealImputationQualityConfig(
        datasets=("veteran",),
        data_dir=tmp_path,
        n_trials=1,
        holdout_fraction=0.25,
        min_holdout_events=4,
        min_context_events=10,
        grid_size=32,
        n_imputation_samples=8,
        parametric_fit_families=(),
        skip_tabicl=True,
    )
    results = real_quality.run_real_imputation_quality_comparison(config)
    summary = real_quality.summarize_real_quality(results)
    ranks = real_quality.summarize_real_ranks(results)

    assert set(results["method"]) == {"kaplan_meier"}
    assert set(results["mode"]) == {"conditional", "unconditional"}
    assert np.all(results["status"] == "ok")
    assert np.all(np.isfinite(results["median_log_mae"]))
    assert np.all(np.isfinite(results["sample_crps_normalized"]))
    assert not summary.empty
    assert not ranks.empty
