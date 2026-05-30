#!/bin/bash
#SBATCH --account=def-qltian
#SBATCH --job-name=survresume
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --signal=TERM@120
#SBATCH --mail-user=qltian2021@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

# ==========================================================================
# 2-GPU H100 resume smoke test (~30 min) — checkpoint-load validation.
#
# Usage:
#   RESUME_CKPT=/scratch/$USER/survival-medium-<JOBID>/checkpoints/step-300.ckpt \
#     sbatch scripts/resume_smoke_run.sh
#
# Gates:
#   1. Log says "Loading checkpoint from ..."
#   2. Log says "Resuming training at step 300"
#   3. step-350.ckpt exists
#   4. surv_nll + impute remain finite after resume
#   5. No optimizer/scheduler load-state errors
# ==========================================================================

set -euo pipefail

# ---- require RESUME_CKPT -----------------------------------------------
: "${RESUME_CKPT:?Set RESUME_CKPT=/path/to/step-300.ckpt}"

if [[ ! -f "$RESUME_CKPT" ]]; then
    echo "ERROR: RESUME_CKPT file not found: ${RESUME_CKPT}" >&2
    exit 2
fi

echo "============================================"
echo "Survival Pretraining Resume Smoke Test"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        $(hostname)"
echo "GPUs:        ${SLURM_GPUS_ON_NODE:-?}"
echo "CPU/task:    ${SLURM_CPUS_PER_TASK:-?}"
echo "Resume from: ${RESUME_CKPT}"
echo "============================================"

# ---- environment -------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_HANDLING=1
export OMP_NUM_THREADS=$(($SLURM_CPUS_PER_TASK / ${SLURM_GPUS_ON_NODE:-2}))
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
    echo "Expected survival_prior.py at the repo root." >&2
    exit 2
fi

cd "$REPO_DIR"
echo "Repo dir:    ${REPO_DIR}"

# Install project in editable mode
pip install -e . --quiet 2>&1 | tail -2

# ---- directories -------------------------------------------------------
RUN_DIR="/scratch/${USER}/survival-resume-${SLURM_JOB_ID}"
CKPT_DIR="${RUN_DIR}/checkpoints"
mkdir -p "$CKPT_DIR"

# ---- verify GPU + imports ----------------------------------------------
echo ""
echo "--- GPU check ---"
python -c "
import torch
print(f'PyTorch {torch.__version__}  CUDA {torch.version.cuda}')
print(f'Visible GPUs:   {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    mem_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
    print(f'  GPU {i}: {props.name} ({mem_gb:.1f} GB)')
"

echo ""
echo "--- Import check ---"
python -c "
from tabicl.survival import TimeBinner, DiscreteTimeSurvivalHead, HybridSurvivalLoss
print('All imports OK')
"

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
echo "Training: ${NPROC} GPUs, resume at step 300 → 350"
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
    --max_steps 350 \
    --batch_size 512 \
    --micro_batch_size 8 \
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
    --checkpoint_path "${RESUME_CKPT}" \
    --save_perm_every 50 \
    --wandb_log False

# ---- verify checkpoints ------------------------------------------------
echo ""
echo "============================================"
echo "Checkpoints:"
ls -lh "${CKPT_DIR}/" 2>/dev/null || echo "(no checkpoints)"
echo "============================================"

echo ""
echo "=== Resume smoke test complete ==="
echo "Output:     logs/survresume-${SLURM_JOB_ID}.out"
echo "Errors:     logs/survresume-${SLURM_JOB_ID}.err"
echo "Checkpoints: ${CKPT_DIR}/"
echo ""
echo "Verify:"
echo "  ls -lh ${CKPT_DIR}/"
echo "  grep -E 'Loading checkpoint|Resuming training' logs/survresume-${SLURM_JOB_ID}.out"
echo "  grep -E 'surv_nll|impute' logs/survresume-${SLURM_JOB_ID}.out | tail -10"
echo "  grep -iE 'traceback|RuntimeError|load_state|nan|inf' logs/survresume-${SLURM_JOB_ID}.err"
