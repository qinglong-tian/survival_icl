# Real Survival Benchmarks

This project keeps small real right-censored survival datasets in
`survival_data/real_benchmarks/`. They are downloaded from public R dataset
mirrors and normalized to:

```text
time,event,<feature columns...>
```

Run:

```bash
cd tabicl-main
python scripts/download_real_survival_benchmarks.py
```

Load:

```python
from tabicl.survival import load_real_survival_benchmark

data = load_real_survival_benchmark("gbsg")
X, t, delta = data.X, data.time, data.event
```

Registered datasets:

| Name | R package/item | Endpoint | Notes |
| --- | --- | --- | --- |
| `veteran` | `survival::veteran` | death | Veterans' Administration lung cancer trial. |
| `pbc` | `survival::pbc` | death | `status == 2` is death; transplant is treated as censored for the death endpoint. |
| `gbsg` | `survival::gbsg` | recurrence or death | German Breast Cancer Study Group recurrence-free survival. |
| `colon_death` | `survival::colon` | death | Keeps `etype == 2` rows from the two-row-per-subject source. |
| `rossi` | `carData::Rossi` | first arrest | One-year right-censored recidivism data. |
| `bmt_death` | `KMsurv::bmt` | death | Bone marrow transplant data, death endpoint. |
| `burn_infection` | `KMsurv::burn` | staphylococcal infection | Burn study infection endpoint. |
| `kidney` | `KMsurv::kidney` | infection | Kidney catheter infection data. |

Rows with missing feature, time, or event values are dropped in the processed
CSV files so the outputs can be passed directly to the survival estimator.
