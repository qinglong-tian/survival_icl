#!/bin/bash
#SBATCH --account=def-qltian
#SBATCH --job-name=surv-s1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --signal=TERM@120
#SBATCH --mail-user=qltian2021@gmail.com
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

# ==========================================================================
# Stage 1 PH Survival Pretraining — Auto-Resuming Chunked HPC Job
#
# Config:
#   STAGE1_TARGET_STEPS=100000   (override via env/sbatch)
#   STAGE1_CHUNK_STEPS=5000
#   SURVIVAL_CHECKPOINT_DIR=/scratch/$USER/survival-stage1
#
# Each job trains STAGE1_CHUNK_STEPS then resubmits itself.
# Last chunk auto-stops at STAGE1_TARGET_STEPS.
# ==========================================================================

set -euo pipefail

# ---- tunables ----------------------------------------------------------
STAGE1_TARGET_STEPS="${STAGE1_TARGET_STEPS:-100000}"
STAGE1_CHUNK_STEPS="${STAGE1_CHUNK_STEPS:-5000}"
SURVIVAL_CHECKPOINT_DIR="${SURVIVAL_CHECKPOINT_DIR:-/scratch/${USER}/survival-stage1}"
export SURVIVAL_CHECKPOINT_DIR  # so resubmitted jobs inherit it
export STAGE1_TARGET_STEPS
export STAGE1_CHUNK_STEPS

CKPT_DIR="${SURVIVAL_CHECKPOINT_DIR}/checkpoints"
OUTFILE="logs/surv-s1-${SLURM_JOB_ID}.out"
ERRFILE="logs/surv-s1-${SLURM_JOB_ID}.err"

echo "============================================"
echo "Stage 1 PH Survival — Auto Chunk"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "GPUs:         ${SLURM_GPUS_ON_NODE:-?}"
echo "CPU/task:     ${SLURM_CPUS_PER_TASK:-?}"
echo "Target steps: ${STAGE1_TARGET_STEPS}"
echo "Chunk steps:  ${STAGE1_CHUNK_STEPS}"
echo "Checkpoints:  ${CKPT_DIR}"
echo "============================================"

# ---- environment -------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_HANDLING=1
export OMP_NUM_THREADS=$((${SLURM_CPUS_PER_TASK:-24} / ${SLURM_GPUS_ON_NODE:-2}))
export MKL_NUM_THREADS=${OMP_NUM_THREADS}

# ---- modules + venv ----------------------------------------------------
module --force purge
module load StdEnv/2023 python/3.10.13
source ~/venvs/icl/bin/activate

# ---- project setup -----------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-}"
if [[ -z "$REPO_DIR" ]]; then
    for CANDIDATE in "${SLURM_SUBMIT_DIR:-}" "$PWD" "$SCRIPT_DIR" "$SCRIPT_DIR/.."; do
        [[ -z "${CANDIDATE:-}" ]] && continue
        if [[ -f "$CANDIDATE/survival_prior.py" ]]; then
            REPO_DIR="$CANDIDATE"
            break
        fi
    done
fi

if [[ -z "$REPO_DIR" || ! -f "$REPO_DIR/survival_prior.py" ]]; then
    echo "ERROR: Could not locate survival-pretrain repo root." >&2
    exit 2
fi

cd "$REPO_DIR"
echo "Repo dir:     ${REPO_DIR}"

pip install -e . --quiet 2>&1 | tail -2

# ---- determine current step --------------------------------------------
mkdir -p "${CKPT_DIR}"

CURRENT_STEP=0
if ls "${CKPT_DIR}"/step-*.ckpt &>/dev/null; then
    # Parse highest step from checkpoint filenames.
    # Extract step number first, then numeric-sort (portable, no GNU sed/sort -V).
    CURRENT_STEP=$(ls "${CKPT_DIR}"/step-*.ckpt 2>/dev/null \
        | while IFS= read -r f; do
            b=$(basename "$f")
            s="${b#step-}"; s="${s%%.ckpt}"; s="${s%%-*}"
            echo "$s"
        done \
        | sort -n \
        | tail -1)
    LATEST_FILE=$(ls "${CKPT_DIR}"/step-*.ckpt 2>/dev/null \
        | while IFS= read -r f; do
            b=$(basename "$f")
            s="${b#step-}"; s="${s%%.ckpt}"; s="${s%%-*}"
            echo "$s $f"
        done \
        | sort -n \
        | tail -1 \
        | cut -d' ' -f2-)
    if [[ -n "$CURRENT_STEP" && "$CURRENT_STEP" =~ ^[0-9]+$ ]]; then
        echo "Latest ckpt:  $(basename "$LATEST_FILE")  (step ${CURRENT_STEP})"
    else
        echo "ERROR: Could not parse step from checkpoints" >&2
        exit 2
    fi
else
    echo "No checkpoints found — starting from scratch."
fi

NEXT_MAX=$((CURRENT_STEP + STAGE1_CHUNK_STEPS))
if (( NEXT_MAX > STAGE1_TARGET_STEPS )); then
    NEXT_MAX=$STAGE1_TARGET_STEPS
fi

if (( CURRENT_STEP >= STAGE1_TARGET_STEPS )); then
    echo "Already at target (${CURRENT_STEP} >= ${STAGE1_TARGET_STEPS}). Nothing to do."
    exit 0
fi

echo "Current step: ${CURRENT_STEP}"
echo "Next max:     ${NEXT_MAX}"

# ---- pre-download TabICL checkpoint ------------------------------------
echo ""
echo "--- Pre-downloading TabICL regressor checkpoint ---"
python -c "
from tabicl._sklearn.regressor import TabICLRegressor
r = TabICLRegressor(n_estimators=1, model_path=None, allow_auto_download=True, device='cpu')
r._resolve_device()
r._load_model()
print('Checkpoint cached')
"

# ---- training ----------------------------------------------------------
NPROC="${SLURM_GPUS_ON_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo ""
echo "============================================"
echo "Training: ${NPROC} GPUs, steps ${CURRENT_STEP} → ${NEXT_MAX}"
echo "Checkpoints: ${CKPT_DIR}"
echo "============================================"
echo ""

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    src/tabicl/train/_run.py \
    --task survival \
    --device cuda \
    --dtype float32 \
    --amp True \
    --np_seed 42 \
    --torch_seed 42 \
    --max_steps "${NEXT_MAX}" \
    --batch_size 512 \
    --micro_batch_size 4 \
    --lr 1e-4 \
    --scheduler cosine_warmup \
    --warmup_proportion 0.02 \
    --gradient_clipping 1.0 \
    --prior_type mlp_scm \
    --prior_device cpu \
    --batch_size_per_gp 4 \
    --min_features 2 \
    --max_features 100 \
    --max_seq_len 1024 \
    --min_train_size 1.0 \
    --max_train_size 1.0 \
    --survival_model_type ph \
    --survival_beta 1.0 \
    --baseline_types weibull,gompertz,loglogistic,lognormal \
    --baseline_mode mix \
    --min_censor_scale 1.0 \
    --max_censor_scale 5.0 \
    --min_event_rate 0.40 \
    --max_event_rate 1.0 \
    --num_bins 50 \
    --alpha_start 3.0 \
    --alpha_floor 0.05 \
    --embed_dim 128 \
    --col_num_blocks 3 \
    --col_nhead 4 \
    --col_num_inds 128 \
    --row_num_blocks 3 \
    --row_nhead 8 \
    --row_num_cls 4 \
    --row_rope_base 100000 \
    --icl_num_blocks 12 \
    --icl_nhead 4 \
    --ff_factor 2 \
    --norm_first True \
    --freeze_col True \
    --freeze_row True \
    --checkpoint_dir "${CKPT_DIR}" \
    --save_temp_every 50 \
    --save_perm_every 5000 \
    --wandb_log False

EXIT_CODE=$?

echo ""
echo "--- Training exited with code ${EXIT_CODE} ---"

# ---- verify checkpoint after training ----------------------------------
if [[ $EXIT_CODE -ne 0 ]]; then
    echo "ERROR: Training failed (exit ${EXIT_CODE}). Will NOT resubmit." >&2
    echo "Check ${ERRFILE} for details." >&2
    exit $EXIT_CODE
fi

# Confirm we have a checkpoint at or past NEXT_MAX
FOUND_STEP=0
if ls "${CKPT_DIR}"/step-*.ckpt &>/dev/null; then
    FOUND_STEP=$(ls "${CKPT_DIR}"/step-*.ckpt 2>/dev/null \
        | while IFS= read -r f; do
            b=$(basename "$f")
            s="${b#step-}"; s="${s%%.ckpt}"; s="${s%%-*}"
            echo "$s"
        done \
        | sort -n \
        | tail -1)
fi

echo "Highest checkpoint after run: step-${FOUND_STEP}"

if (( FOUND_STEP < CURRENT_STEP + STAGE1_CHUNK_STEPS / 2 )); then
    echo "WARNING: Checkpoint progressed only ${FOUND_STEP} out of expected ${NEXT_MAX}."
    echo "Will NOT resubmit — check logs."
    exit 1
fi

# ---- auto-resubmit if not done -----------------------------------------
if (( FOUND_STEP >= STAGE1_TARGET_STEPS )); then
    echo ""
    echo "============================================"
    echo "STAGE 1 COMPLETE — reached step ${FOUND_STEP} >= ${STAGE1_TARGET_STEPS}"
    echo "Final checkpoint: ${CKPT_DIR}/step-${FOUND_STEP}.ckpt"
    echo "============================================"
    exit 0
fi

# Resubmit self
THIS_SCRIPT="${BASH_SOURCE[0]}"
NEXT_JOB_ID=$(sbatch --parsable "$THIS_SCRIPT")
echo ""
echo "============================================"
echo "Chunk complete (step ${FOUND_STEP}/${STAGE1_TARGET_STEPS})."
echo "Resubmitted:  ${NEXT_JOB_ID}"
echo "============================================"
