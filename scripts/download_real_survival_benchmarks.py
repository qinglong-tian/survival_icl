#!/usr/bin/env python
"""Download small real right-censored survival benchmark datasets."""

from __future__ import annotations

import argparse
import json

from tabicl.survival._real_datasets import (
    DEFAULT_REAL_SURVIVAL_DATA_DIR,
    download_real_survival_benchmarks,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_REAL_SURVIVAL_DATA_DIR),
        help="Output directory for raw CSVs, processed CSVs, and manifest.json.",
    )
    args = parser.parse_args()
    manifest = download_real_survival_benchmarks(data_dir=args.data_dir)
    print(json.dumps({
        item["name"]: {
            "rows": item["n_rows"],
            "features": item["n_features"],
            "event_rate": round(item["event_rate"], 4),
            "dropped_incomplete_rows": item["dropped_incomplete_rows"],
        }
        for item in manifest["datasets"]
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
