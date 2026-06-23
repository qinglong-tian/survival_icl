from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("lifelines")


def _load_benchmark_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_comparison.py"
    spec = importlib.util.spec_from_file_location("benchmark_comparison", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark_module()


def test_baseline_functions_are_well_behaved():
    times = np.linspace(0.05, 10.0, 80)
    hazards = np.linspace(0.01, 3.0, 40)
    for family_key, family in benchmark.FAMILIES.items():
        params = family.true_params
        cumulative = benchmark.cumulative_hazard0(times, family_key, params)
        survival = benchmark.baseline_survival(times, family_key, params)
        log_hazard = benchmark.log_hazard0(times, family_key, params)
        inverted = benchmark.inverse_cumulative_hazard0(hazards, family_key, params)
        round_trip = benchmark.cumulative_hazard0(inverted, family_key, params)

        assert np.all(np.isfinite(cumulative))
        assert np.all(np.diff(cumulative) >= -1e-10)
        assert np.all((survival >= 0.0) & (survival <= 1.0))
        assert np.all(np.diff(survival) <= 1e-10)
        assert np.all(np.isfinite(log_hazard))
        assert np.allclose(round_trip, hazards, rtol=1e-5, atol=1e-5)


def test_parametric_ph_fit_produces_valid_survival_curves():
    config = benchmark.BenchmarkConfig(
        n_samples=180,
        n_features=3,
        n_context=120,
        grid_size=20,
        skip_tabicl=True,
    )
    data = benchmark.generate_ph_data("weibull", config, seed=123)
    model = benchmark.fit_parametric_ph(
        "weibull",
        data.X[: config.n_context],
        data.t_obs[: config.n_context],
        data.delta[: config.n_context],
        "weibull",
    )
    grid, _ = benchmark.evaluation_grid(
        data.t_obs[: config.n_context],
        data.delta[: config.n_context],
        config.grid_size,
    )
    curves = model.survival(data.X[config.n_context : config.n_context + 5], grid)

    assert model.specification == "correct"
    assert curves.shape == (5, config.grid_size)
    assert np.all(np.isfinite(curves))
    assert np.all((curves >= 0.0) & (curves <= 1.0))
    assert np.all(np.diff(curves, axis=1) <= 1e-8)


def test_small_benchmark_smoke_without_tabicl():
    config = benchmark.BenchmarkConfig(
        n_samples=96,
        n_features=3,
        n_context=64,
        n_trials=1,
        grid_size=12,
        skip_tabicl=True,
    )
    results, calibration, examples = benchmark.run_benchmark(config)

    assert not results.empty
    assert not calibration.empty
    assert set(results["data_family"]) == {family.label for family in benchmark.FAMILIES.values()}
    assert {"c_index", "brier", "ibs", "calibration_error", "oracle_ise", "oracle_iae"}.issubset(results.columns)
    assert np.all(np.isfinite(results[["c_index", "brier", "ibs", "calibration_error", "oracle_ise", "oracle_iae"]]))
    assert set(examples) == {family.label for family in benchmark.FAMILIES.values()}

    summaries = benchmark.summarize_results(results)
    gaps = benchmark.parametric_calibration_gaps(results)
    assert {"metrics", "brier", "calibration"}.issubset(summaries)
    assert not gaps.empty
    assert "oracle_ise_gap" in gaps.columns

    fig = benchmark.plot_calibration_by_family(calibration)
    assert fig.axes
