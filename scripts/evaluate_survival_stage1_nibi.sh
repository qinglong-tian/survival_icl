#!/bin/bash
#SBATCH --account=def-qltian
#SBATCH --job-name=surv-s1-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

VENV_PATH="${VENV_PATH:-${HOME}/venvs/icl/bin/activate}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
HOLDOUT_DIR="${HOLDOUT_DIR:?Set HOLDOUT_DIR to the immutable stage1_v1 holdout}"
STAGE1_DIR="${STAGE1_DIR:?Set STAGE1_DIR to the Stage 1 checkpoint directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${STAGE1_DIR}/evaluation-stage1-v1-${SLURM_JOB_ID:-local}}"
EVAL_STEPS="${EVAL_STEPS:-0,500,1000,2000,5000}"

module --force purge
module load StdEnv/2023 python/3.10.13
source "$VENV_PATH"
cd "$REPO_DIR"
pip install -e . --quiet

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
    --checkpoints "${checkpoints[@]}"
