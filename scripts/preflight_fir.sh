#!/bin/bash
#SBATCH --account=def-qltian
#SBATCH --job-name=surv-fir-check
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-user=qltian2021@gmail.com
#SBATCH --mail-type=FAIL,TIME_LIMIT

# Validate the isolated Fir environment, direct CUDA execution, and two-rank NCCL.
#
#   sbatch scripts/preflight_fir.sh

set -euo pipefail

PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.5}"
FIR_VENV_DIR="${FIR_VENV_DIR:-/project/6079932/${USER}/venvs/survival-icl-fir-py311}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
EXPECTED_GPUS=2

module --force purge
module load StdEnv/2023 "$PYTHON_MODULE"

if [[ ! -f "$FIR_VENV_DIR/.fir-env-ready" ]]; then
    echo "ERROR: Fir environment is not ready: ${FIR_VENV_DIR}" >&2
    echo "       Run: sbatch scripts/setup_fir_env.sh" >&2
    exit 2
fi
source "$FIR_VENV_DIR/bin/activate"
cd "$REPO_DIR"

echo "============================================"
echo "Fir GPU Preflight"
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo "Repository:   ${REPO_DIR}"
echo "Git commit:   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Python:       $(command -v python)"
echo "Environment:  ${FIR_VENV_DIR}"
echo "============================================"

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
python scripts/check_fir_runtime.py --expected-gpus "$EXPECTED_GPUS"
torchrun --standalone --nproc_per_node="$EXPECTED_GPUS" \
    scripts/check_fir_runtime.py --expected-gpus "$EXPECTED_GPUS" --distributed
python -m pip check

echo "Fir CUDA and NCCL preflight passed."
