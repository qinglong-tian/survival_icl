#!/bin/bash
#SBATCH --account=def-qltian
#SBATCH --job-name=surv-s1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-user=qltian2021@gmail.com
#SBATCH --mail-type=FAIL,TIME_LIMIT

# Nibi two-H100 launcher for the canonical survival Stage 1 curriculum.
#
# Safe default: an isolated 50-step test with no resubmission.
#   sbatch scripts/train_survival_stage1_nibi.sh
#
# Formal pilot: 5,000 steps in completed 500-step chunks. Each successful
# chunk resubmits this script and resumes model, optimizer, and scheduler state.
# The scheduler retains its 100,000-step horizon, and checkpoints
# 0/500/1000/2000/5000 are preserved for evaluation.
#   sbatch --time=08:00:00 --export=ALL,RUN_MODE=formal \
#       scripts/train_survival_stage1_nibi.sh
#
# Test mode uses a compressed 5,000-step scheduler horizon so short stability
# runs reach the full Stage 1 learning rate. Set STAGE1_SCHEDULER_STEPS=100000
# to reproduce the exact beginning of the formal schedule.

set -euo pipefail

RUN_MODE="${RUN_MODE:-test}"
JOB_ID="${SLURM_JOB_ID:-local}"
STAGE1_CHUNK_TIME="${STAGE1_CHUNK_TIME:-08:00:00}"
WANDB_MODE="${WANDB_MODE:-offline}"
SURVIVAL_QUERY_PINBALL_WEIGHT="${SURVIVAL_QUERY_PINBALL_WEIGHT:-0.0}"
SURVIVAL_QUERY_PINBALL_QUANTILES="${SURVIVAL_QUERY_PINBALL_QUANTILES:-0.1,0.25,0.5,0.75,0.9}"
PRIOR_NUM_WORKERS="${PRIOR_NUM_WORKERS:-1}"
PRIOR_N_JOBS="${PRIOR_N_JOBS:-3}"
BASE_NP_SEED="${BASE_NP_SEED:-42}"
BASE_TORCH_SEED="${BASE_TORCH_SEED:-42}"
VENV_PATH="${VENV_PATH:-${HOME}/venvs/icl/bin/activate}"

case "$RUN_MODE" in
    test)
        STAGE1_TARGET_STEPS="${STAGE1_TARGET_STEPS:-50}"
        STAGE1_CHUNK_STEPS="${STAGE1_CHUNK_STEPS:-50}"
        STAGE1_SCHEDULER_STEPS="${STAGE1_SCHEDULER_STEPS:-5000}"
        CURRICULUM_ID="${CURRICULUM_ID:-nibi_stage1_test_${JOB_ID}}"
        SURVIVAL_CHECKPOINT_DIR="${SURVIVAL_CHECKPOINT_DIR:-/scratch/${USER}/survival-icl-tests/${JOB_ID}}"
        AUTO_RESUBMIT=0
        ;;
    formal)
        STAGE1_TARGET_STEPS="${STAGE1_TARGET_STEPS:-5000}"
        STAGE1_CHUNK_STEPS="${STAGE1_CHUNK_STEPS:-500}"
        STAGE1_SCHEDULER_STEPS="${STAGE1_SCHEDULER_STEPS:-100000}"
        CURRICULUM_ID="${CURRICULUM_ID:-author_adapted_v1}"
        SURVIVAL_CHECKPOINT_DIR="${SURVIVAL_CHECKPOINT_DIR:-/scratch/${USER}/survival-icl}"
        AUTO_RESUBMIT=1
        ;;
    *)
        echo "ERROR: RUN_MODE must be 'test' or 'formal' (got '${RUN_MODE}')." >&2
        exit 2
        ;;
esac

for name in STAGE1_TARGET_STEPS STAGE1_CHUNK_STEPS STAGE1_SCHEDULER_STEPS; do
    value="${!name}"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "ERROR: ${name} must be a positive integer divisible by 50 (got '${value}')." >&2
        exit 2
    fi
    value=$((10#${value}))
    printf -v "$name" "%d" "$value"
    if (( value <= 0 || value % 50 != 0 )); then
        echo "ERROR: ${name} must be a positive integer divisible by 50 (got '${value}')." >&2
        exit 2
    fi
done

for name in BASE_NP_SEED BASE_TORCH_SEED; do
    value="${!name}"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "ERROR: ${name} must be a non-negative integer (got '${value}')." >&2
        exit 2
    fi
    printf -v "$name" "%d" "$((10#${value}))"
done

if [[ ! "$PRIOR_NUM_WORKERS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: PRIOR_NUM_WORKERS must be a non-negative integer (got '${PRIOR_NUM_WORKERS}')." >&2
    exit 2
fi
PRIOR_NUM_WORKERS=$((10#${PRIOR_NUM_WORKERS}))
if [[ ! "$PRIOR_N_JOBS" =~ ^[0-9]+$ ]] || (( 10#${PRIOR_N_JOBS} < 1 )); then
    echo "ERROR: PRIOR_N_JOBS must be a positive integer (got '${PRIOR_N_JOBS}')." >&2
    exit 2
fi
PRIOR_N_JOBS=$((10#${PRIOR_N_JOBS}))

if (( STAGE1_CHUNK_STEPS > STAGE1_TARGET_STEPS )); then
    echo "ERROR: STAGE1_CHUNK_STEPS cannot exceed STAGE1_TARGET_STEPS." >&2
    exit 2
fi
if (( STAGE1_SCHEDULER_STEPS < STAGE1_TARGET_STEPS )); then
    echo "ERROR: STAGE1_SCHEDULER_STEPS cannot be less than STAGE1_TARGET_STEPS." >&2
    exit 2
fi

GPU_COUNT="${SLURM_GPUS_ON_NODE:-2}"
GPU_COUNT="${GPU_COUNT##*:}"
GPU_COUNT="${GPU_COUNT%%(*}"
if [[ ! "$GPU_COUNT" =~ ^[0-9]+$ ]] || (( GPU_COUNT != 2 )); then
    echo "ERROR: This launcher requires exactly 2 allocated GPUs (got '${SLURM_GPUS_ON_NODE:-unset}')." >&2
    exit 2
fi
CPU_COUNT="${SLURM_CPUS_PER_TASK:-8}"

export CURRICULUM_ID
export RUN_MODE
export STAGE1_CHUNK_STEPS
export STAGE1_CHUNK_TIME
export STAGE1_SCHEDULER_STEPS
export STAGE1_TARGET_STEPS
export SURVIVAL_CHECKPOINT_DIR
export SURVIVAL_QUERY_PINBALL_QUANTILES
export SURVIVAL_QUERY_PINBALL_WEIGHT
export PRIOR_NUM_WORKERS
export PRIOR_N_JOBS
export BASE_NP_SEED
export BASE_TORCH_SEED
export VENV_PATH
export WANDB_MODE

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Each GPU has one training rank plus n_jobs active threads per prior worker.
if (( PRIOR_NUM_WORKERS > 0 )); then
    CPU_PROCESS_COUNT=$((GPU_COUNT * (1 + PRIOR_NUM_WORKERS * PRIOR_N_JOBS)))
else
    CPU_PROCESS_COUNT=$((GPU_COUNT * PRIOR_N_JOBS))
fi
if (( CPU_PROCESS_COUNT > CPU_COUNT )); then
    echo "WARNING: Estimated active CPU threads (${CPU_PROCESS_COUNT}) exceed allocated cores (${CPU_COUNT})." >&2
fi
export OMP_NUM_THREADS=$((CPU_COUNT / CPU_PROCESS_COUNT))
if (( OMP_NUM_THREADS < 1 )); then
    export OMP_NUM_THREADS=1
fi
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export TORCH_NCCL_ASYNC_HANDLING=1

module --force purge
module load StdEnv/2023 python/3.10.13

if [[ ! -f "$VENV_PATH" ]]; then
    echo "ERROR: Python environment activation script not found: ${VENV_PATH}" >&2
    exit 2
fi
source "$VENV_PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIS_SCRIPT="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_DIR="${REPO_DIR:-}"
if [[ -z "$REPO_DIR" ]]; then
    for candidate in "${SLURM_SUBMIT_DIR:-}" "$PWD" "$SCRIPT_DIR/.."; do
        [[ -z "${candidate:-}" ]] && continue
        if [[ -f "$candidate/survival_prior.py" ]]; then
            REPO_DIR="$candidate"
            break
        fi
    done
fi
if [[ -z "$REPO_DIR" || ! -f "$REPO_DIR/survival_prior.py" ]]; then
    echo "ERROR: Could not locate tabicl-main. Set REPO_DIR and resubmit." >&2
    exit 2
fi
export REPO_DIR

cd "$REPO_DIR"
pip install -e . --quiet 2>&1 | tail -2

WANDB_DIR="${WANDB_DIR:-${SURVIVAL_CHECKPOINT_DIR}/wandb}"
STAGE1_DIR="${SURVIVAL_CHECKPOINT_DIR}/survival_mix_${CURRICULUM_ID}_stage1"
mkdir -p "$STAGE1_DIR" "$WANDB_DIR"
export WANDB_DIR

CURRENT_STEP=0
for checkpoint in "$STAGE1_DIR"/step-*.ckpt; do
    [[ -f "$checkpoint" ]] || continue
    filename="${checkpoint##*/}"
    if [[ "$filename" =~ ^step-([0-9]+).*\.ckpt$ ]] && (( BASH_REMATCH[1] > CURRENT_STEP )); then
        CURRENT_STEP="${BASH_REMATCH[1]}"
    fi
done

if (( CURRENT_STEP > STAGE1_TARGET_STEPS )); then
    echo "ERROR: Latest checkpoint step ${CURRENT_STEP} exceeds target ${STAGE1_TARGET_STEPS}." >&2
    exit 2
fi
if (( CURRENT_STEP == STAGE1_TARGET_STEPS )); then
    echo "Stage 1 already complete at step ${CURRENT_STEP}: ${STAGE1_DIR}"
    exit 0
fi

NP_SEED=$((BASE_NP_SEED + CURRENT_STEP))
TORCH_SEED=$((BASE_TORCH_SEED + CURRENT_STEP))
export NP_SEED
export TORCH_SEED

NEXT_STEP=$((CURRENT_STEP + STAGE1_CHUNK_STEPS))
if (( NEXT_STEP > STAGE1_TARGET_STEPS )); then
    NEXT_STEP="$STAGE1_TARGET_STEPS"
fi

echo "============================================"
echo "Nibi Survival Stage 1"
echo "Mode:             ${RUN_MODE}"
echo "Job ID:           ${JOB_ID}"
echo "Node:             $(hostname)"
echo "GPUs:             ${GPU_COUNT}"
echo "CPU cores:        ${CPU_COUNT}"
echo "Prior workers:    ${PRIOR_NUM_WORKERS} per rank"
echo "Generation jobs:  ${PRIOR_N_JOBS} per worker"
echo "CPU thread demand:${CPU_PROCESS_COUNT} estimated"
echo "OMP threads:      ${OMP_NUM_THREADS} per process"
echo "Repository:       ${REPO_DIR}"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Current step:     ${CURRENT_STEP}"
echo "Next step:        ${NEXT_STEP}"
echo "Scheduler horizon:${STAGE1_SCHEDULER_STEPS}"
echo "Chunk seeds:      numpy=${NP_SEED}, torch=${TORCH_SEED}"
echo "Checkpoint dir:   ${STAGE1_DIR}"
echo "WandB mode:       ${WANDB_MODE}"
if [[ "$RUN_MODE" == "test" ]]; then
    echo "LR interpretation: test warmup lasts $((STAGE1_SCHEDULER_STEPS * 2 / 100)) steps."
fi
echo "============================================"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -c "
import hashlib
from pathlib import Path

import torch
from tabicl._model.attention import HAS_FLASH_ATTN3

paths = [Path('survival_prior.py')]
paths += sorted(Path('src').rglob('*.py'))
paths += sorted(Path('scripts').rglob('*.sh'))
digest = hashlib.sha256()
for path in paths:
    digest.update(str(path).encode())
    digest.update(path.read_bytes())

print(f'PyTorch {torch.__version__}; CUDA {torch.version.cuda}; GPUs {torch.cuda.device_count()}')
print(f'FlashAttention-3 installed: {HAS_FLASH_ATTN3}; float32 training uses PyTorch SDPA')
print(f'Source fingerprint: {digest.hexdigest()[:16]}')
"

CHECKPOINT_DIR="$SURVIVAL_CHECKPOINT_DIR" \
NPROC_PER_NODE="$GPU_COUNT" \
RUN_STAGES=1 \
STAGE1_SCHEDULER_STEPS="$STAGE1_SCHEDULER_STEPS" \
STAGE1_STEPS="$NEXT_STEP" \
bash scripts/train_survival_curriculum.sh

EXPECTED_CHECKPOINT="${STAGE1_DIR}/step-${NEXT_STEP}.ckpt"
if [[ ! -f "$EXPECTED_CHECKPOINT" ]]; then
    echo "ERROR: Expected completed-chunk checkpoint not found: ${EXPECTED_CHECKPOINT}" >&2
    exit 1
fi
python scripts/inspect_training_checkpoint.py "$EXPECTED_CHECKPOINT"

echo "Completed Stage 1 chunk: ${CURRENT_STEP} -> ${NEXT_STEP}"

if (( NEXT_STEP >= STAGE1_TARGET_STEPS )); then
    echo "Stage 1 complete: ${EXPECTED_CHECKPOINT}"
    exit 0
fi

if (( ! AUTO_RESUBMIT )); then
    echo "Test mode complete. No follow-up job submitted."
    exit 0
fi

NEXT_JOB_ID=$(sbatch --parsable \
    --time="$STAGE1_CHUNK_TIME" \
    --export=ALL,RUN_MODE=formal \
    "$THIS_SCRIPT")
echo "Submitted next Stage 1 chunk as job ${NEXT_JOB_ID}."
