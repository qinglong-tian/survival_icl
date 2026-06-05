#!/bin/bash
#SBATCH --account=aip-qltian
#SBATCH --job-name=surv-s1-vulcan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-user=qltian2021@gmail.com
#SBATCH --mail-type=FAIL,TIME_LIMIT

# Vulcan single-node L40S launcher for the canonical survival Stage 1 curriculum.
#
# Safe default: an isolated 50-step four-GPU test with no resubmission.
#   sbatch scripts/train_survival_stage1_vulcan.sh
#
# Formal pilot: 5,000 steps in completed 500-step chunks. Each successful
# chunk resubmits this script and resumes model, optimizer, and scheduler state.
# The scheduler retains its 100,000-step horizon, and checkpoints
# 0/500/1000/2000/5000 are preserved for evaluation.
#   sbatch --time=12:00:00 --export=ALL,RUN_MODE=formal \
#       scripts/train_survival_stage1_vulcan.sh
#
# Vulcan uses four 48 GB L40S GPUs by default. Stage 1 micro-batches default to
# two tasks per GPU to leave memory headroom. Override STAGE1_MICRO_BATCH_SIZE=1
# after a test if needed. A one-GPU smoke test can be submitted with:
#   sbatch --gres=gpu:1 --cpus-per-task=8 scripts/train_survival_stage1_vulcan.sh

set -euo pipefail

RUN_MODE="${RUN_MODE:-test}"
JOB_ID="${SLURM_JOB_ID:-local}"
STAGE1_CHUNK_TIME="${STAGE1_CHUNK_TIME:-12:00:00}"
STAGE1_MICRO_BATCH_SIZE="${STAGE1_MICRO_BATCH_SIZE:-2}"
WANDB_MODE="${WANDB_MODE:-offline}"
SURVIVAL_QUERY_PINBALL_WEIGHT="${SURVIVAL_QUERY_PINBALL_WEIGHT:-0.0}"
SURVIVAL_QUERY_PINBALL_QUANTILES="${SURVIVAL_QUERY_PINBALL_QUANTILES:-0.1,0.25,0.5,0.75,0.9}"
PRIOR_NUM_WORKERS="${PRIOR_NUM_WORKERS:-1}"
PRIOR_N_JOBS="${PRIOR_N_JOBS:-3}"
BASE_NP_SEED="${BASE_NP_SEED:-42}"
BASE_TORCH_SEED="${BASE_TORCH_SEED:-42}"
SOFTWARE_STACK="${SOFTWARE_STACK:-StdEnv/2023}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.10.13}"
VENV_PATH="${VENV_PATH:-${HOME}/venvs/icl-vulcan/bin/activate}"

case "$RUN_MODE" in
    test)
        STAGE1_TARGET_STEPS="${STAGE1_TARGET_STEPS:-50}"
        STAGE1_CHUNK_STEPS="${STAGE1_CHUNK_STEPS:-50}"
        STAGE1_SCHEDULER_STEPS="${STAGE1_SCHEDULER_STEPS:-5000}"
        CURRICULUM_ID="${CURRICULUM_ID:-vulcan_stage1_test_${JOB_ID}}"
        SURVIVAL_CHECKPOINT_DIR="${SURVIVAL_CHECKPOINT_DIR:-/scratch/${USER}/survival-icl-tests/${JOB_ID}}"
        AUTO_RESUBMIT=0
        ;;
    formal)
        STAGE1_TARGET_STEPS="${STAGE1_TARGET_STEPS:-5000}"
        STAGE1_CHUNK_STEPS="${STAGE1_CHUNK_STEPS:-500}"
        STAGE1_SCHEDULER_STEPS="${STAGE1_SCHEDULER_STEPS:-100000}"
        CURRICULUM_ID="${CURRICULUM_ID:-vulcan_stage1_pilot_5k_v1}"
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

if [[ ! "$STAGE1_MICRO_BATCH_SIZE" =~ ^[0-9]+$ ]]; then
    echo "ERROR: STAGE1_MICRO_BATCH_SIZE must be a positive integer (got '${STAGE1_MICRO_BATCH_SIZE}')." >&2
    exit 2
fi
STAGE1_MICRO_BATCH_SIZE=$((10#${STAGE1_MICRO_BATCH_SIZE}))
if (( STAGE1_MICRO_BATCH_SIZE <= 0 || 4 % STAGE1_MICRO_BATCH_SIZE != 0 )); then
    echo "ERROR: STAGE1_MICRO_BATCH_SIZE must be a positive divisor of 4 (got '${STAGE1_MICRO_BATCH_SIZE}')." >&2
    exit 2
fi

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

GPU_COUNT="${SLURM_GPUS_ON_NODE:-4}"
GPU_COUNT="${GPU_COUNT##*:}"
GPU_COUNT="${GPU_COUNT%%(*}"
if [[ ! "$GPU_COUNT" =~ ^[0-9]+$ ]] || (( GPU_COUNT < 1 || GPU_COUNT > 4 )); then
    echo "ERROR: This launcher requires between 1 and 4 allocated GPUs (got '${SLURM_GPUS_ON_NODE:-unset}')." >&2
    exit 2
fi
CPU_COUNT="${SLURM_CPUS_PER_TASK:-32}"

export CURRICULUM_ID
export RUN_MODE
export STAGE1_CHUNK_STEPS
export STAGE1_CHUNK_TIME
export STAGE1_MICRO_BATCH_SIZE
export STAGE1_SCHEDULER_STEPS
export STAGE1_TARGET_STEPS
export SURVIVAL_CHECKPOINT_DIR
export SURVIVAL_QUERY_PINBALL_QUANTILES
export SURVIVAL_QUERY_PINBALL_WEIGHT
export PRIOR_NUM_WORKERS
export PRIOR_N_JOBS
export BASE_NP_SEED
export BASE_TORCH_SEED
export SOFTWARE_STACK
export PYTHON_MODULE
export VENV_PATH
export WANDB_MODE

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
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
export HF_HOME="${HF_HOME:-${SURVIVAL_CHECKPOINT_DIR}/huggingface}"

module --force purge
module load "$SOFTWARE_STACK" "$PYTHON_MODULE"

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
python -m pip install -e . --no-deps --no-build-isolation --quiet 2>&1 | tail -2

WANDB_DIR="${WANDB_DIR:-${SURVIVAL_CHECKPOINT_DIR}/wandb}"
STAGE1_DIR="${SURVIVAL_CHECKPOINT_DIR}/survival_mix_${CURRICULUM_ID}_stage1"
mkdir -p "$STAGE1_DIR" "$WANDB_DIR" "$HF_HOME"
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

if (( CURRENT_STEP == 0 )); then
    python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='jingang/TabICL', filename='tabicl-regressor-v2-20260212.ckpt', local_files_only=True))" \
        || {
            echo "ERROR: Pretrained TabICL regressor is not cached under HF_HOME=${HF_HOME}." >&2
            echo "       Download it on a Vulcan login node before resubmitting." >&2
            exit 2
        }
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
echo "Vulcan Survival Stage 1"
echo "Mode:             ${RUN_MODE}"
echo "Job ID:           ${JOB_ID}"
echo "Node:             $(hostname)"
echo "GPUs:             ${GPU_COUNT}"
echo "CPU cores:        ${CPU_COUNT}"
echo "Prior workers:    ${PRIOR_NUM_WORKERS} per rank"
echo "Generation jobs:  ${PRIOR_N_JOBS} per worker"
echo "CPU thread demand:${CPU_PROCESS_COUNT} estimated"
echo "OMP threads:      ${OMP_NUM_THREADS} per process"
echo "Micro-batch:      ${STAGE1_MICRO_BATCH_SIZE} per GPU"
echo "Repository:       ${REPO_DIR}"
echo "Git commit:       $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Current step:     ${CURRENT_STEP}"
echo "Next step:        ${NEXT_STEP}"
echo "Scheduler horizon:${STAGE1_SCHEDULER_STEPS}"
echo "Chunk seeds:      numpy=${NP_SEED}, torch=${TORCH_SEED}"
echo "Checkpoint dir:   ${STAGE1_DIR}"
echo "HF cache:         ${HF_HOME}"
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
print(f'FlashAttention-3 installed: {HAS_FLASH_ATTN3}; L40S float32 training uses PyTorch SDPA')
print(f'Source fingerprint: {digest.hexdigest()[:16]}')
"

CHECKPOINT_DIR="$SURVIVAL_CHECKPOINT_DIR" \
NPROC_PER_NODE="$GPU_COUNT" \
RUN_STAGES=1 \
STAGE1_MICRO_BATCH_SIZE="$STAGE1_MICRO_BATCH_SIZE" \
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
    --gres="gpu:${GPU_COUNT}" \
    --cpus-per-task="$CPU_COUNT" \
    --time="$STAGE1_CHUNK_TIME" \
    --export=ALL,RUN_MODE=formal \
    "$THIS_SCRIPT")
echo "Submitted next Stage 1 chunk as job ${NEXT_JOB_ID}."
