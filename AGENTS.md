# AGENTS.md

## Project Overview

**TabICL / TabICLv2** — a state-of-the-art tabular foundation model from Inria's SODA team (ICML 2025 + ICML 2026). The model performs in-context learning (ICL) on tabular data: given `(X_train, y_train, X_test)`, it predicts `y_test` in a single transformer forward pass without gradient updates. It is pre-trained on millions of synthetic datasets generated via structural causal models.

- **Python**: >= 3.10
- **PyPI**: `pip install tabicl`
- **Checkpoints**: hosted on Hugging Face Hub (`jingang/TabICL`)
- **License**: BSD 3-Clause (subpackage `forecast/` is Apache 2.0)

## Directory Structure

```
tabicl-main/
├── src/tabicl/          # Main package
│   ├── __init__.py              # Public API: TabICLClassifier, TabICLRegressor, InferenceConfig
│   ├── _model/                  # Core PyTorch nn.Module (TabICL, embeddings, attention, etc.)
│   ├── _sklearn/                # scikit-learn wrappers (Classifier, Regressor, preprocessing)
│   ├── _finetune/               # Full PyTorch fine-tuning (AdamW, early stopping, multi-GPU)
│   ├── _unsupervised/           # Imputation, outlier detection, synthetic data generation
│   ├── forecast/                # Time series forecasting (derived from TabPFN-TS)
│   ├── prior/                   # Synthetic data generation for pre-training (SCMs)
│   ├── train/                   # Distributed pre-training infrastructure (Stage 1→2→3)
│   └── shap/                    # SHAP / ShapIQ explainability
├── tests/                       # Pytest tests (sklearn checks, KV cache, input formats)
├── tutorials/                   # Sphinx Gallery tutorials (10 Python scripts)
├── docs/                        # Sphinx documentation
├── scripts/                     # Shell scripts for pre-training stages
└── pyproject.toml               # Build config (hatchling), dependencies
```

## Commands

Install in editable mode:
```bash
pip install -e .
```

Run tests:
```bash
hatch test
```

Type checking:
```bash
hatch run types:check
```

Run a single test file:
```bash
hatch run pytests tests/test_sklearn.py -v
```

## Architecture

The model has three sequential stages:

1. **ColEmbedding** (`_model/embedding.py`) — Column-wise transformer using Induced Set Attention Blocks (ISAB). Produces distribution-aware embeddings per feature column, optionally target-aware.
2. **RowInteraction** (`_model/interaction.py`) — Row-wise transformer with Rotary Position Encoding (RoPE). Captures feature interactions within each row, producing CLS token representations.
3. **ICLearning** (`_model/learning.py`) — Decoder-style transformer that conditions on `(X_train, y_train)` and produces predictions for `X_test`.

Key techniques:
- **Scalable Softmax (SSMax)** — variants like `qassmax-mlp-elementwise` for length generalization
- **Mixed-radix ensembling** — handles `> max_classes` via hierarchical classification
- **Quantile prediction** — regression outputs full distribution via `QuantileToDistribution`
- **KV caching** — `TabICLCache` stores training-data projections for faster repeated inference
- **CPU/disk offloading** — `InferenceManager` handles large datasets beyond GPU memory

## Code Conventions

- **Public API** lives in `src/tabicl/__init__.py`. Internal modules are prefixed with `_` (e.g., `_sklearn/`, `_model/`).
- **Lazy imports** via `__getattr__` for optional dependencies (`forecast`, `finetune`, etc.).
- **scikit-learn compatibility**: estimators inherit from `BaseEstimator`, `ClassifierMixin`, `RegressorMixin` and pass `parametrize_with_checks`.
- **Docstrings**: NumPy-style with full parameter descriptions.
- **Type hints**: used throughout, checked with `mypy`.
- **No comments in code** unless strictly necessary.
- **Imports**: `from __future__ import annotations` at top of all modules.
- **Version**: dynamically sourced from `src/tabicl/__about__.py`.

## Environment Setup

Core deps: `torch>=2.2`, `scikit-learn>=1.3.0`, `numpy`, `scipy`, `einops>=0.7`, `psutil`, `tqdm>=4.64.0`, `huggingface-hub`.

Optional extras:
| Extra | Deps | Purpose |
|-------|------|---------|
| `forecast` | pandas, gluonts, statsmodels, matplotlib | Time series forecasting |
| `shap` | shap>=0.42, shapiq>=1.0, matplotlib, numba | SHAP explainability |
| `pretrain` | xgboost, wandb | Pre-training (v1) |
| `finetune` | wandb | Single-dataset fine-tuning |
| `test` | pandas | Test suite |
| `all` | all of the above | Everything |

On Intel Macs, install PyTorch via conda first: `conda install pytorch -c pytorch`

## Testing

- **Framework**: pytest with `hatch test` (uses `hatch-test` env with `features = ["test"]`)
- **CI**: GitHub Actions on push/PR to main, nightly cron, manual dispatch. Matrix: 3 OS × 5 Python versions.
- **Coverage**: Codecov upload on CI; `tool.coverage` in pyproject.toml with branch coverage.
- **Test files**: `tests/test_sklearn.py` (sklearn compatibility + KV cache), `tests/test_numpy_inputs.py` (NaN/dtype handling), `tests/test_string_input.py` (DataFrame string handling).

## Research Extensions: Survival Prior Data Generation

This fork extends TabICL with synthetic **survival** (time-to-event) data generation
built on top of the SCM-based prior. The goal is to produce `(X, t, delta)` datasets
for pre-training a tabular survival foundation model.

### Quick Test

```bash
python survival_prior.py --model_type ph --num_batches 1 --batch_size 4 \
    --prior_type mlp_scm --baseline_types weibull --baseline_mode weibull
```

### Key Extension Files

| File | Purpose |
|------|---------|
| `survival_prior.py` | `SurvivalPriorDataset`, `SaveSurvivalPriorDataset`, `LoadSurvivalPriorDataset`, CLI |
| `src/tabicl/prior/_survival.py` | PH/AFT samplers, baselines, `SurvivalSCMPrior`, censoring logic |
| `inspect_censoring.py` | Diagnostic script: event rate distribution across censor scales |
| `scripts/train_survival_curriculum.sh` | Canonical 3-stage on-the-fly pretraining curriculum |

### Data Format (`get_batch()` return tuple)

| Tensor | Shape | Description |
|--------|-------|-------------|
| `X` | `(batch, seq, max_features)` | Features (sparse when saved, dense in-memory) |
| `t` | `(batch, seq)` | Observed time = **min**(event_time, censoring_time) |
| `delta` | `(batch, seq)` | Event indicator: **1** = event observed, **0** = censored |
| `d` | `(batch,)` | Number of active features per dataset |
| `seq_lens` | `(batch,)` | Samples per dataset (always `max_seq_len` for Stage 1) |
| `train_sizes` | `(batch,)` | Always equals `seq_len` — no train/test split in survival data |

### Survival Pipeline

```
SCM (MLP, no trees) → continuous y
    │ standard_scaling (Reg2Cls)
    ▼
SurvivalSCMPrior(model_type="ph" or "aft")
    ├── PH:  t = inverse_CDF(u, β*y, baseline_params)
    │        c = inverse_CDF(u_c, 0, baseline_params) × censor_scale
    └── AFT: t = T₀ · exp(-β*y)
             c = T₀_c × censor_scale
    │
    ▼
t_obs = min(t, c)     ← observed time
delta = (t < c).float()  ← 1=event happened before censoring
```

### Current Defaults (post-fix)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `prior_type` | `mlp_scm` | TreeSCM disabled (xgboost too slow at large N) |
| `n_jobs` | `1` | Changed from `-1` to avoid fork() memory explosion |
| `min_event_rate` | `0.40` | Down from 0.50 (fewer rejections at low censor scales) |
| `max_event_rate` | `1.0` | Effectively no upper bound (≤1.0 always) |
| `min_train_size` | `1.0` | No train/test split; full dataset |
| `max_train_size` | `1.0` | No train/test split; full dataset |
| `backend` | `threading` | Changed from `loky` (fork unsafe on macOS) |

### Critical Gotchas

- **NEVER use `n_jobs > 1` with `loky` backend on macOS** — `fork()` after PyTorch is loaded causes:
  1. Deadlocked child processes (background threads' mutexes locked forever)
  2. Corrupted memory allocator → unbounded RSS growth
  3. MacOS `fork` from multi-threaded process → undefined behavior per Apple docs
  The fix: `backend="threading"` in `_survival.py:734` and `_dataset.py:680`.

- **`t` is `min(event_time, censoring_time)`, NOT max** — `delta=1` means `t` is the event time.

- **Censor scale controls event rate**: `cs=1.0` → ~50% events, `cs=5.0` → ~80-90%.
  At `cs=1.0` (curriculum minimum), event rates cluster at ~0.50. The old bound
  of `0.50` rejected half of datasets. Lowered to `0.40` to fix this.

- **`train_sizes` is always `seq_len`** — the train/test split from upstream TabICL
  was repurposed to produce undivided datasets for survival. The tensors in `.pt`
  files still carry the field, but its value is always the full sequence length.

- **TreeSCM is disabled** — `prior_type` defaults changed from `mix_scm` to `mlp_scm`
  in `survival_prior.py` and the archived `scripts/_upstream/survival_curriculum.sh`. TreeSCM uses xgboost
  internally (slow at N > 10K) and the rejection loop made it 10-100× slower than MLP.

- **Generation loop is bounded** — `while True` replaced with `for _ in range(5000)`
  in `_survival.py:610`. A `RuntimeError` is raised if 5000 attempts fail.

- **Gompertz is PH-only** — AFT baseline pool omits gompertz. If listed in
  `baseline_types` with `--model_type aft`, it's silently ignored.

- **`num_classes` is forced to 0** — `Reg2Cls` skips discretization; targets stay continuous.

### Baseline Distributions

**PH (4):** Weibull (k~U[0.5,3]), Gompertz (γ~LogU[0.01,0.5]), LogLogistic (β~U[0.5,3]), LogNormal (μ~U[-2,2],σ=1)
**AFT (3):** Weibull, LogLogistic, LogNormal (same params, no Gompertz)

### Clipping / Scaling

1. `u.clamp(ε, 1-ε)` with ε=1e-6 (configurable via `--u_eps`)
2. Internal: arg ≤ 36.0 in `inverse_cdf`; p clamped in `ndtri`
3. Raw times use a numerical safety max (`--max_time`, default 1e30), not a modeling horizon.
4. Model-facing times are per-task standardized log-times fit on context observed times only, then clipped to [-6, 6].

### Training Script

The canonical on-the-fly pretraining curriculum:
```bash
bash scripts/train_survival_curriculum.sh
```
Trains one mixed PH/AFT survival foundation model in 3 stages
(small fixed-length → medium variable-length → large variable-length).
All data is generated on-the-fly; no disk pre-generation is required.
Stages 1 and 2 train the full model. Stage 3 freezes the column and row
encoders while refining the ICL predictor and survival head. The optimization
schedule follows the original TabICL recipe: `1e-4` cosine with warmup,
`2e-5 → 5e-6` polynomial decay, then constant `2e-6`.

Optional environment overrides:
```bash
RUN_STAGES=1,2,3                     # which stages to run
CURRICULUM_ID=author_adapted_v1      # checkpoint namespace
STAGE1_STEPS=100000                  # override per-stage step count
STAGE1_SCHEDULER_STEPS=100000        # fixed LR horizon for chunked Stage 1
SURVIVAL_QUERY_PINBALL_WEIGHT=0.0    # optional oracle-query pinball weight
PRIOR_NUM_WORKERS=1                  # prefetch workers per DDP rank
PRIOR_N_JOBS=1                       # within-batch generation threads per worker
NPROC_PER_NODE=1                     # GPUs
PYTHON_MODULE=python/3.11            # match the Python used to create the HPC venv
WANDB_MODE=offline                   # online/offline/disabled
```

Alliance Fir/Nibi two-H100 Stage 1 test and formal launcher:
```bash
sbatch scripts/train_survival_stage1_nibi.sh
sbatch --time=00:10:00 --export=ALL,PREFLIGHT_ONLY=1 scripts/train_survival_stage1_nibi.sh
sbatch --time=08:00:00 --export=ALL,RUN_MODE=formal scripts/train_survival_stage1_nibi.sh
```
The safe default is an isolated 50-step test. It preserves the formal
100,000-step scheduler horizon, so all 50 steps remain inside the 2,000-step
warmup and are not a convergence test. Formal mode runs completed
500-step chunks, preserves the 100,000-step cosine schedule, and resubmits
itself only after verifying the expected checkpoint. Stage 1 currently uses
true float32 training with AMP disabled; float32 attention does not silently
downcast through FlashAttention-3. The Fir/Nibi launcher uses one background prior
worker with three within-batch generation threads per rank, targeting its eight
allocated CPU cores without multiplying full-batch prefetch memory.
The launcher performs a real CUDA allocation before starting `torchrun`; an
NVML-only device count is insufficient because Fir can list GPUs even when an
incompatible PyTorch CUDA runtime cannot initialize them. It also runs a
two-rank NCCL all-reduce preflight. GPU commands run inside an explicit
single-task `srun` step so Fir grants the task access to both allocated GPUs.
If driver-level `cuInit` still returns `CUDA_ERROR_NO_DEVICE`, rebuilding the
Python environment cannot fix it; send the job ID and preflight log to Alliance
support.

The previous disk-based generation script is archived at
`scripts/_upstream/survival_curriculum.sh`.
