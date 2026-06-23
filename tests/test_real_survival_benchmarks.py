from __future__ import annotations

import pytest

from tabicl.survival._real_datasets import (
    DEFAULT_REAL_SURVIVAL_DATA_DIR,
    REAL_SURVIVAL_DATASETS,
    dataset_names,
    load_real_survival_benchmark,
)


def test_real_survival_registry_is_small_right_censored_and_model_sized():
    assert dataset_names() == tuple(REAL_SURVIVAL_DATASETS)
    assert len(REAL_SURVIVAL_DATASETS) >= 6
    for name, spec in REAL_SURVIVAL_DATASETS.items():
        assert spec.name == name
        assert spec.csv_url.startswith("https://vincentarelbundock.github.io/Rdatasets/csv/")
        assert spec.doc_url.startswith("https://vincentarelbundock.github.io/Rdatasets/doc/")
        assert spec.time_col
        assert spec.event_col
        assert 1 <= len(spec.feature_cols) <= 100
        assert "interval" not in spec.event_description.lower()
        assert "left" not in spec.event_description.lower()


def test_downloaded_real_survival_benchmarks_load_if_present():
    if not (DEFAULT_REAL_SURVIVAL_DATA_DIR / "manifest.json").is_file():
        pytest.skip("real survival benchmark cache has not been downloaded")

    for name in dataset_names():
        data = load_real_survival_benchmark(name)
        assert len(data.X) == len(data.time) == len(data.event)
        assert 0 < len(data.X) < 30000
        assert 1 <= data.X.shape[1] <= 100
        assert (data.time > 0).all()
        assert set(data.event.tolist()).issubset({0.0, 1.0})
        assert not data.frame.isna().any().any()
