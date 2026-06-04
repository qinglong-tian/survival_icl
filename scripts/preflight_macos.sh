#!/bin/bash
# macOS MPS Preflight — 3-stage survival pretraining curriculum
# Validates end-to-end pipeline before moving to HPC/CUDA.
#
# Usage:
#   bash scripts/preflight_macos.sh
#
# Environment overrides:
#   REPO_DIR     — path to tabicl-main (default: auto-detected)
#   CKPT_DIR     — checkpoint directory (default: ./checkpoints/preflight)
#   WANDB_DIR    — wandb directory (default: ./wandb)
#   WANDB_MODE   — wandb mode (default: offline)
#
# Stages (small steps for fast validation):
#   Stage 1 (20 steps): fixed-length 128, freeze encoders, HF Hub init
#   Stage 2 (20 steps): variable-length 128–512, freeze encoders, load Stage 1
#   Stage 3 (10 steps): variable-length 256–1024, unfreeze all, load Stage 2
#
# Gates:
#   1. Imports OK (tabicl.survival, SurvivalPriorDataset)
#   2. HF Hub checkpoint downloads
#   3. All 3 stages complete without NaN/inf/error
#   4. Checkpoints exist at each stage boundary
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-}"
if [[ -z "$REPO_DIR" ]]; then
    for CANDIDATE in "$PWD" "$SCRIPT_DIR/.." "$SCRIPT_DIR"; do
        [[ -z "${CANDIDATE:-}" ]] && continue
        if [[ -f "$CANDIDATE/survival_prior.py" ]]; then
            REPO_DIR="$CANDIDATE"
            break
        fi
    done
fi
if [[ -z "$REPO_DIR" || ! -f "$REPO_DIR/survival_prior.py" ]]; then
    echo "ERROR: Could not locate tabicl-main repo root (expected survival_prior.py)." >&2
    echo "Set REPO_DIR=/path/to/tabicil-main and retry." >&2
    exit 2
fi

CKPT_DIR="${CKPT_DIR:-$REPO_DIR/checkpoints/preflight}"
WANDB_DIR="${WANDB_DIR:-$REPO_DIR/wandb}"
WANDB_MODE="${WANDB_MODE:-offline}"

DEVICE="cpu"
AMP="False"
DTYPE="float32"

mkdir -p "$CKPT_DIR" "$WANDB_DIR"

echo "============================================"
echo "macOS Preflight — Survival Pretraining (CPU)"
echo "Repo:      $REPO_DIR"
echo "Checkpts:  $CKPT_DIR"
echo "Device:    $DEVICE"
echo "WandB:     $WANDB_MODE"
echo "============================================"

cd "$REPO_DIR"

# ---- Verify environment ---------------------------------------------------

echo ""
echo "--- PyTorch + MPS check ---"
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'MPS built:     {torch.backends.mps.is_built()}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('Running on CPU (MPS has buffer limits for this model size).')
"

echo ""
echo "--- Import check ---"
python -c "
from tabicl.survival import TimeBinner, DiscreteTimeSurvivalHead, HybridSurvivalLoss
from survival_prior import SurvivalPriorDataset
print('All imports OK')
"

echo ""
echo "--- Pre-downloading TabICL regressor checkpoint ---"
python -c "
from tabicl._sklearn.regressor import TabICLRegressor
r = TabICLRegressor(n_estimators=1, model_path=None, allow_auto_download=True, device='cpu')
r._resolve_device()
r._load_model()
print('HF Hub checkpoint cached successfully.')
"

# ---- Shared flags (arrays — preserves quoting in paths with spaces) ---------

MODEL_FLAGS=(
    --embed_dim 128
    --col_num_blocks 3 --col_nhead 4 --col_num_inds 128
    --row_num_blocks 3 --row_nhead 8 --row_num_cls 4 --row_rope_base 100000
    --icl_num_blocks 12 --icl_nhead 4
    --ff_factor 2 --norm_first True
)

SURVIVAL_FLAGS=(
    --survival_model_type mix
    --baseline_types weibull,gompertz,loglogistic,lognormal
    --baseline_mode mix
    --beta_sampling log_uniform --min_beta 0.25 --max_beta 2.0
    --baseline_param_prior broad
    --time_scale_sampling log_uniform --min_time_scale 0.2 --max_time_scale 5.0
    --censoring_strategy target_event_rate
    --min_event_rate 0.40 --max_event_rate 0.90
    --prior_type mlp_scm
    --prior_device cpu
    --min_train_size 1.0 --max_train_size 1.0
)

TRAIN_FLAGS=(
    --task survival
    --device "$DEVICE"
    --dtype "$DTYPE"
    --amp "$AMP"
    --np_seed 42 --torch_seed 42
    --batch_size 128
    --scheduler cosine_warmup --warmup_proportion 0.02
    --gradient_clipping 1.0
    --num_bins 50
    --min_features 2 --max_features 100
)

WANDB_FLAGS=(
    --wandb_log True
    --wandb_project TabICL-Survival-Preflight
    --wandb_dir "$WANDB_DIR"
    --wandb_mode "$WANDB_MODE"
)

_run_python() {
    python src/tabicl/train/_run.py "$@"
}

# ---- Stage 1: small fixed-length, freeze encoders, HF Hub init ------------

echo ""
echo "============================================"
echo "  Stage 1 — fixed length 128, freeze encoders"
echo "  Init from: HF Hub"
echo "============================================"

mkdir -p "$CKPT_DIR/stage1"

_run_python \
    "${TRAIN_FLAGS[@]}" \
    "${MODEL_FLAGS[@]}" \
    "${SURVIVAL_FLAGS[@]}" \
    "${WANDB_FLAGS[@]}" \
    --wandb_name "Preflight-Stage1" \
    --max_steps 20 \
    --lr 1e-4 \
    --batch_size_per_gp 4 --micro_batch_size 4 \
    --max_seq_len 128 \
    --freeze_col True --freeze_row True \
    --checkpoint_dir "$CKPT_DIR/stage1" \
    --save_temp_every 20 --save_perm_every 20

STAGE1_CKPT="$CKPT_DIR/stage1/step-20.ckpt"
if [[ ! -f "$STAGE1_CKPT" ]]; then
    echo "ERROR: Stage 1 checkpoint not found: $STAGE1_CKPT" >&2
    exit 1
fi
echo "Stage 1 complete: $(ls -lh "$STAGE1_CKPT" | awk '{print $5}')"

# ---- Stage 2: medium variable-length, freeze encoders, load Stage 1 --------

echo ""
echo "============================================"
echo "  Stage 2 — variable length 128–512"
echo "  Init from: $STAGE1_CKPT"
echo "============================================"

mkdir -p "$CKPT_DIR/stage2"

_run_python \
    "${TRAIN_FLAGS[@]}" \
    "${MODEL_FLAGS[@]}" \
    "${SURVIVAL_FLAGS[@]}" \
    "${WANDB_FLAGS[@]}" \
    --wandb_name "Preflight-Stage2" \
    --max_steps 20 \
    --lr 1e-5 \
    --batch_size_per_gp 2 --micro_batch_size 2 \
    --min_seq_len 128 --max_seq_len 512 \
    --log_seq_len True --seq_len_per_gp True \
    --freeze_col True --freeze_row True \
    --pretrained_path "$STAGE1_CKPT" \
    --checkpoint_dir "$CKPT_DIR/stage2" \
    --save_temp_every 20 --save_perm_every 20

STAGE2_CKPT="$CKPT_DIR/stage2/step-20.ckpt"
if [[ ! -f "$STAGE2_CKPT" ]]; then
    echo "ERROR: Stage 2 checkpoint not found: $STAGE2_CKPT" >&2
    exit 1
fi
echo "Stage 2 complete: $(ls -lh "$STAGE2_CKPT" | awk '{print $5}')"

# ---- Stage 3: large variable-length, unfreeze all, load Stage 2 ------------

echo ""
echo "============================================"
echo "  Stage 3 — variable length 256–1024, unfreeze all"
echo "  Init from: $STAGE2_CKPT"
echo "============================================"

mkdir -p "$CKPT_DIR/stage3"

_run_python \
    "${TRAIN_FLAGS[@]}" \
    "${MODEL_FLAGS[@]}" \
    "${SURVIVAL_FLAGS[@]}" \
    "${WANDB_FLAGS[@]}" \
    --wandb_name "Preflight-Stage3" \
    --max_steps 10 \
    --lr 1e-5 \
    --batch_size_per_gp 1 --micro_batch_size 1 \
    --min_seq_len 256 --max_seq_len 1024 \
    --log_seq_len True --seq_len_per_gp True \
    --replay_small True \
    --pretrained_path "$STAGE2_CKPT" \
    --checkpoint_dir "$CKPT_DIR/stage3" \
    --save_temp_every 10 --save_perm_every 10

STAGE3_CKPT="$CKPT_DIR/stage3/step-10.ckpt"
if [[ ! -f "$STAGE3_CKPT" ]]; then
    echo "ERROR: Stage 3 checkpoint not found: $STAGE3_CKPT" >&2
    exit 1
fi
echo "Stage 3 complete: $(ls -lh "$STAGE3_CKPT" | awk '{print $5}')"

# ---- Gate checks -----------------------------------------------------------

echo ""
echo "============================================"
echo "Gate checks"
echo "============================================"

FAILED=0

for stage in 1 2 3; do
    dir="$CKPT_DIR/stage$stage"
    if [[ -d "$dir" ]]; then
        n_ckpt=$(find "$dir" -maxdepth 1 -name 'step-*.ckpt' | wc -l | tr -d ' ')
        echo "  Stage $stage: $n_ckpt checkpoint(s) in $dir"
        if (( n_ckpt == 0 )); then
            echo "  [WARN] No checkpoints for Stage $stage"
        fi
        for ckpt in "$dir"/step-*.ckpt; do
            [[ -f "$ckpt" ]] || continue
            size=$(ls -lh "$ckpt" | awk '{print $5}')
            echo "           $(basename "$ckpt") ($size)"
        done
    else
        echo "  [FAIL] Stage $stage directory missing: $dir"
        FAILED=1
    fi
done

echo ""
echo "============================================"
echo "Final model checkpoint:"
echo "  $STAGE3_CKPT"
echo "============================================"

if (( FAILED == 0 )); then
    echo ""
    echo "=== Preflight complete — pipeline is healthy for HPC ==="
    echo ""
    echo "Next steps:"
    echo "  1. Copy checkpoints/preflight/stage3/step-10.ckpt to HPC"
    echo "  2. Kick off full curriculum:"
    echo "     bash scripts/train_survival_curriculum.sh"
    echo "  3. Or resume from this checkpoint:"
    echo "     RESUME_CKPT=$STAGE3_CKPT sbatch scripts/resume_smoke_run.sh"
else
    echo ""
    echo "=== Preflight FAILED — review output above ==="
    exit 1
fi
