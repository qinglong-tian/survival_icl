#!/bin/bash
#SBATCH --account=aip-qltian
#SBATCH --job-name=surv-s1-eval-vulcan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-user=qltian2021@gmail.com
#SBATCH --mail-type=FAIL,TIME_LIMIT

set -euo pipefail

SOFTWARE_STACK="${SOFTWARE_STACK:-StdEnv/2023}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.10.13}"
VENV_PATH="${VENV_PATH:-${HOME}/venvs/icl-vulcan/bin/activate}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
HOLDOUT_DIR="${HOLDOUT_DIR:?Set HOLDOUT_DIR to the immutable stage1_v1 holdout}"
STAGE1_DIR="${STAGE1_DIR:?Set STAGE1_DIR to the Stage 1 checkpoint directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${STAGE1_DIR}/evaluation-stage1-v1-${SLURM_JOB_ID:-local}}"
EVAL_STEPS="${EVAL_STEPS:-0,500,1000,2000,5000}"
TASK_BATCH_SIZE="${TASK_BATCH_SIZE:-2}"

if [[ ! "$TASK_BATCH_SIZE" =~ ^[0-9]+$ ]] || (( 10#${TASK_BATCH_SIZE} < 1 )); then
    echo "ERROR: TASK_BATCH_SIZE must be a positive integer (got '${TASK_BATCH_SIZE}')." >&2
    exit 2
fi
TASK_BATCH_SIZE=$((10#${TASK_BATCH_SIZE}))

module --force purge
module load "$SOFTWARE_STACK" "$PYTHON_MODULE"

if [[ ! -f "$VENV_PATH" ]]; then
    echo "ERROR: Python environment activation script not found: ${VENV_PATH}" >&2
    exit 2
fi
source "$VENV_PATH"

cd "$REPO_DIR"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -c "
from pathlib import Path

import numpy
import scipy
import tabicl
import torch

repo = Path('${REPO_DIR}').resolve()
loaded = Path(tabicl.__file__).resolve()
expected = (repo / 'src' / 'tabicl').resolve()
if expected not in loaded.parents:
    raise RuntimeError(f'Loaded tabicl from {loaded}, expected source under {expected}')
print(f'Python environment OK: {Path(torch.__file__).resolve().parent.parent}')
print(f'TabICL source: {loaded}')
"

echo "============================================"
echo "Vulcan Survival Stage 1 Evaluation"
echo "Node:             $(hostname)"
echo "Repository:       ${REPO_DIR}"
echo "Holdout:          ${HOLDOUT_DIR}"
echo "Checkpoint dir:   ${STAGE1_DIR}"
echo "Output dir:       ${OUTPUT_DIR}"
echo "Task batch size:  ${TASK_BATCH_SIZE}"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

checkpoints=()
IFS=',' read -ra steps <<< "$EVAL_STEPS"
for step in "${steps[@]}"; do
    [[ "$step" =~ ^[0-9]+$ ]] || { echo "Invalid checkpoint step: $step" >&2; exit 1; }
    checkpoint="${STAGE1_DIR}/step-${step}.ckpt"
    [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
    checkpoints+=("$checkpoint")
done

python scripts/evaluate_survival_holdout.py \
    --holdout-dir "$HOLDOUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --task-batch-size "$TASK_BATCH_SIZE" \
    --checkpoints "${checkpoints[@]}"
