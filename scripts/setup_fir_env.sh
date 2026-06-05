#!/bin/bash
#SBATCH --account=def-qltian
#SBATCH --job-name=surv-fir-env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-user=qltian2021@gmail.com
#SBATCH --mail-type=FAIL,TIME_LIMIT

# Build an isolated Fir environment from scratch on a CPU compute node.
#
#   sbatch scripts/setup_fir_env.sh
#
# Override FIR_VENV_DIR or REPO_DIR with --export=ALL,... when needed.

set -euo pipefail

PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.5}"
FIR_VENV_DIR="${FIR_VENV_DIR:-/project/6079932/${USER}/venvs/survival-icl-fir-py311}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
READY_MARKER="${FIR_VENV_DIR}/.fir-env-ready"

module --force purge
module load StdEnv/2023 "$PYTHON_MODULE"

if [[ ! -f "$REPO_DIR/survival_prior.py" || ! -f "$REPO_DIR/pyproject.toml" ]]; then
    echo "ERROR: REPO_DIR is not tabicl-main: ${REPO_DIR}" >&2
    exit 2
fi
if [[ -e "$FIR_VENV_DIR" ]]; then
    echo "ERROR: Refusing to modify existing Fir environment: ${FIR_VENV_DIR}" >&2
    echo "       Choose a new FIR_VENV_DIR or move the existing directory first." >&2
    exit 2
fi

mkdir -p "$(dirname "$FIR_VENV_DIR")"
python -m venv "$FIR_VENV_DIR"
source "$FIR_VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install "torch==2.10.0"
python -m pip install -e "${REPO_DIR}[pretrain,test]"
python -m pip install pytest
python -m pip check
python -m pip freeze > "$FIR_VENV_DIR/fir-requirements.lock"

cd "$REPO_DIR"
python -m pytest tests/test_train_optim.py -q
python - <<'PY'
import sys

import joblib
import numpy
import sklearn
import torch
import transformers
import wandb

print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
print(f"Transformers: {transformers.__version__}")
print(f"WandB: {wandb.__version__}")
print(f"NumPy: {numpy.__version__}; scikit-learn: {sklearn.__version__}; joblib: {joblib.__version__}")
PY

{
    echo "python_module=${PYTHON_MODULE}"
    echo "python=$(command -v python)"
    echo "repo=${REPO_DIR}"
    echo "commit=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "created=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$READY_MARKER"

echo "Fir environment ready: ${FIR_VENV_DIR}"
echo "Next: sbatch scripts/preflight_fir.sh"
