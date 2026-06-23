"""Real right-censored survival benchmark dataset helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable


DEFAULT_REAL_SURVIVAL_DATA_DIR = Path("survival_data") / "real_benchmarks"


@dataclass(frozen=True)
class RealSurvivalDatasetSpec:
    """Metadata and transformation rules for one real benchmark dataset."""

    name: str
    package: str
    item: str
    title: str
    csv_url: str
    doc_url: str
    time_col: str
    event_col: str
    feature_cols: tuple[str, ...]
    event_description: str
    categorical_cols: tuple[str, ...] = ()
    notes: str = ""
    filter_col: str | None = None
    filter_value: int | str | None = None


@dataclass
class RealSurvivalDataset:
    """Loaded real survival benchmark in normalized right-censored format."""

    name: str
    X: object
    time: object
    event: object
    frame: object
    metadata: dict


def _rdatasets_csv(package: str, item: str) -> str:
    return f"https://vincentarelbundock.github.io/Rdatasets/csv/{package}/{item}.csv"


def _rdatasets_doc(package: str, item: str) -> str:
    return f"https://vincentarelbundock.github.io/Rdatasets/doc/{package}/{item}.html"


REAL_SURVIVAL_DATASETS: dict[str, RealSurvivalDatasetSpec] = {
    "veteran": RealSurvivalDatasetSpec(
        name="veteran",
        package="survival",
        item="veteran",
        title="Veterans' Administration lung cancer trial",
        csv_url=_rdatasets_csv("survival", "veteran"),
        doc_url=_rdatasets_doc("survival", "veteran"),
        time_col="time",
        event_col="status",
        feature_cols=("trt", "celltype", "karno", "diagtime", "age", "prior"),
        event_description="status is the observed death indicator; 0/1 right-censoring coding.",
        categorical_cols=("trt", "celltype", "prior"),
    ),
    "pbc": RealSurvivalDatasetSpec(
        name="pbc",
        package="survival",
        item="pbc",
        title="Mayo Clinic primary biliary cholangitis trial",
        csv_url=_rdatasets_csv("survival", "pbc"),
        doc_url=_rdatasets_doc("survival", "pbc"),
        time_col="time",
        event_col="status",
        feature_cols=(
            "trt", "age", "sex", "ascites", "hepato", "spiders", "edema",
            "bili", "chol", "albumin", "copper", "alk.phos", "ast", "trig",
            "platelet", "protime", "stage",
        ),
        event_description="event is death: status == 2; status 0/1 are treated as censored for the death endpoint.",
        categorical_cols=("trt", "sex", "ascites", "hepato", "spiders", "edema", "stage"),
    ),
    "gbsg": RealSurvivalDatasetSpec(
        name="gbsg",
        package="survival",
        item="gbsg",
        title="German Breast Cancer Study Group recurrence-free survival",
        csv_url=_rdatasets_csv("survival", "gbsg"),
        doc_url=_rdatasets_doc("survival", "gbsg"),
        time_col="rfstime",
        event_col="status",
        feature_cols=("age", "meno", "size", "grade", "nodes", "pgr", "er", "hormon"),
        event_description="status is recurrence or death; 0 means alive without recurrence at last follow-up.",
        categorical_cols=("meno", "grade", "hormon"),
    ),
    "colon_death": RealSurvivalDatasetSpec(
        name="colon_death",
        package="survival",
        item="colon",
        title="Stage B/C colon cancer death endpoint",
        csv_url=_rdatasets_csv("survival", "colon"),
        doc_url=_rdatasets_doc("survival", "colon"),
        time_col="time",
        event_col="status",
        feature_cols=(
            "rx", "sex", "age", "obstruct", "perfor", "adhere", "nodes",
            "differ", "extent", "surg", "node4",
        ),
        event_description="status is the event indicator after filtering etype == 2 (death records).",
        categorical_cols=("rx", "sex", "obstruct", "perfor", "adhere", "differ", "extent", "surg", "node4"),
        notes="The source has one recurrence row and one death row per subject; this benchmark keeps death rows only.",
        filter_col="etype",
        filter_value=2,
    ),
    "rossi": RealSurvivalDatasetSpec(
        name="rossi",
        package="carData",
        item="Rossi",
        title="Rossi criminal recidivism data",
        csv_url=_rdatasets_csv("carData", "Rossi"),
        doc_url=_rdatasets_doc("carData", "Rossi"),
        time_col="week",
        event_col="arrest",
        feature_cols=("fin", "age", "race", "wexp", "mar", "paro", "prio", "educ"),
        event_description="arrest is 1 if first arrest was observed; censored observations are at 52 weeks.",
        categorical_cols=("fin", "race", "wexp", "mar", "paro"),
        notes="Uses static baseline covariates; week-by-week employment indicators are time-varying and omitted.",
    ),
    "bmt_death": RealSurvivalDatasetSpec(
        name="bmt_death",
        package="KMsurv",
        item="bmt",
        title="Bone marrow transplant death endpoint",
        csv_url=_rdatasets_csv("KMsurv", "bmt"),
        doc_url=_rdatasets_doc("KMsurv", "bmt"),
        time_col="t1",
        event_col="d1",
        feature_cols=(
            "group", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10",
        ),
        event_description="d1 is 1 for death and 0 for alive/on-study censoring.",
        categorical_cols=("group", "z3", "z4", "z5", "z6", "z8", "z9", "z10"),
    ),
    "burn_infection": RealSurvivalDatasetSpec(
        name="burn_infection",
        package="KMsurv",
        item="burn",
        title="Burn data staphylococcal infection endpoint",
        csv_url=_rdatasets_csv("KMsurv", "burn"),
        doc_url=_rdatasets_doc("KMsurv", "burn"),
        time_col="T3",
        event_col="D3",
        feature_cols=(
            "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9", "Z10", "Z11",
        ),
        event_description="D3 is 1 for staphylococcal infection and 0 for no infection/on-study censoring.",
        categorical_cols=("Z1", "Z2", "Z3", "Z5", "Z6", "Z7", "Z8", "Z9", "Z10", "Z11"),
    ),
    "kidney": RealSurvivalDatasetSpec(
        name="kidney",
        package="KMsurv",
        item="kidney",
        title="Kidney catheter infection data",
        csv_url=_rdatasets_csv("KMsurv", "kidney"),
        doc_url=_rdatasets_doc("KMsurv", "kidney"),
        time_col="time",
        event_col="delta",
        feature_cols=("type",),
        event_description="delta is 1 for infection and 0 for right-censoring.",
        categorical_cols=("type",),
    ),
}


def _event_values(spec: RealSurvivalDatasetSpec, event):
    if spec.name == "pbc":
        return (event == 2).astype("int8")
    return event.astype("int8")


def _processed_path(data_dir: str | Path, name: str) -> Path:
    return Path(data_dir) / "processed" / f"{name}.csv"


def dataset_names() -> tuple[str, ...]:
    """Return available real survival benchmark names."""
    return tuple(REAL_SURVIVAL_DATASETS)


def load_real_survival_benchmark(
    name: str,
    *,
    data_dir: str | Path = DEFAULT_REAL_SURVIVAL_DATA_DIR,
) -> RealSurvivalDataset:
    """Load a processed real right-censored survival benchmark.

    Parameters
    ----------
    name : str
        Dataset name from :func:`dataset_names`.
    data_dir : str or Path, default="survival_data/real_benchmarks"
        Directory produced by ``scripts/download_real_survival_benchmarks.py``.

    Returns
    -------
    RealSurvivalDataset
        ``frame`` has columns ``time``, ``event``, and the feature columns.
        ``X`` is the feature frame; ``time`` and ``event`` are NumPy arrays.
    """
    if name not in REAL_SURVIVAL_DATASETS:
        raise KeyError(f"Unknown survival benchmark {name!r}. Available: {dataset_names()}")

    import pandas as pd

    path = _processed_path(data_dir, name)
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. Run "
            "`python scripts/download_real_survival_benchmarks.py` first."
        )
    frame = pd.read_csv(path)
    metadata_path = Path(data_dir) / "manifest.json"
    manifest = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    manifest_item = next(
        (
            item for item in manifest.get("datasets", [])
            if item.get("name") == name
        ),
        {},
    )
    metadata = {**asdict(REAL_SURVIVAL_DATASETS[name]), **manifest_item}
    X = frame.drop(columns=["time", "event"])
    return RealSurvivalDataset(
        name=name,
        X=X,
        time=frame["time"].to_numpy(dtype="float32"),
        event=frame["event"].to_numpy(dtype="float32"),
        frame=frame,
        metadata=metadata,
    )


def load_all_real_survival_benchmarks(
    *,
    data_dir: str | Path = DEFAULT_REAL_SURVIVAL_DATA_DIR,
) -> dict[str, RealSurvivalDataset]:
    """Load all processed real survival benchmarks from ``data_dir``."""
    return {
        name: load_real_survival_benchmark(name, data_dir=data_dir)
        for name in dataset_names()
    }


def download_real_survival_benchmarks(
    *,
    data_dir: str | Path = DEFAULT_REAL_SURVIVAL_DATA_DIR,
    reader: Callable | None = None,
) -> dict:
    """Download and normalize the registered real survival benchmarks.

    The output directory contains ``raw/*.csv``, ``processed/*.csv``, and a
    ``manifest.json`` with provenance, row counts, feature counts, event rates,
    and complete-case filtering statistics.
    """
    import pandas as pd

    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    read_csv = pd.read_csv if reader is None else reader

    manifest = {
        "schema_version": 1,
        "description": "Small real right-censored survival benchmarks normalized to time/event/features.",
        "datasets": [],
    }
    for spec in REAL_SURVIVAL_DATASETS.values():
        raw = read_csv(spec.csv_url)
        raw_row_count = len(raw)
        raw_path = raw_dir / f"{spec.name}.csv"
        raw.to_csv(raw_path, index=False)

        if spec.filter_col is not None:
            raw = raw[raw[spec.filter_col] == spec.filter_value].copy()

        missing = [
            col for col in (spec.time_col, spec.event_col, *spec.feature_cols)
            if col not in raw.columns
        ]
        if missing:
            raise ValueError(f"{spec.name} is missing expected columns: {missing}")

        selected = raw[[spec.time_col, spec.event_col, *spec.feature_cols]].copy()
        selected = selected.rename(columns={spec.time_col: "time", spec.event_col: "event"})
        selected["event"] = _event_values(spec, selected["event"])
        selected["time"] = selected["time"].astype("float64")
        selected = selected[selected["time"] > 0].copy()
        before_complete_case = len(selected)
        selected = selected.dropna(axis=0, how="any").reset_index(drop=True)
        if not set(selected["event"].unique()).issubset({0, 1}):
            raise ValueError(f"{spec.name} has non-binary event values after normalization.")

        processed_path = processed_dir / f"{spec.name}.csv"
        selected.to_csv(processed_path, index=False)
        event_rate = float(selected["event"].mean()) if len(selected) else float("nan")
        manifest["datasets"].append({
            **asdict(spec),
            "raw_path": str(raw_path),
            "processed_path": str(processed_path),
            "n_rows_raw": int(raw_row_count),
            "n_rows_after_endpoint_filter": int(before_complete_case),
            "n_rows": int(len(selected)),
            "n_features": int(len(selected.columns) - 2),
            "event_rate": event_rate,
            "dropped_incomplete_rows": int(before_complete_case - len(selected)),
            "format": "CSV with columns: time, event, then features",
        })

    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
