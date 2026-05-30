#!/bin/bash
# Setup Python environment on Nibi for survival pretraining.
#
# Usage (interactive or in a job script):
#   source scripts/setup_env.sh
#
# This creates a virtualenv on $SLURM_TMPDIR for faster IO during jobs,
# or in $HOME/.venvs if run from a login node (not inside a job).

set -euo pipefail

# ---- module loading ----
module load StdEnv/2023 python/3.11

# ---- venv location ----
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    VENV_DIR="${SLURM_TMPDIR}/surv-env"
else
    VENV_DIR="${HOME}/.venvs/survival-pretrain"
    echo "[setup] Not inside a SLURM job; using persistent venv at ${VENV_DIR}"
fi

# ---- create/activate venv ----
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[setup] Creating virtualenv at ${VENV_DIR}..."
    virtualenv --no-download "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

# ---- install PyTorch (H100 requires >= 2.5.1) ----
echo "[setup] Installing PyTorch..."
pip install --no-index torch

# ---- install project + deps ----
echo "[setup] Installing project dependencies..."
pip install --no-index transformers

# Install tabicl in editable mode from the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
pip install -e "${PROJECT_DIR}"

# ---- verify ----
echo "[setup] Verifying installation..."
python -c "
import torch
import tabicl
from tabicl.survival import TimeBinner, DiscreteTimeSurvivalHead, HybridSurvivalLoss
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print('TabICL + survival imports: OK')
print('Environment ready.')
"

echo "[setup] Done. Virtualenv active at ${VENV_DIR}"
echo "[setup] Run: source ${VENV_DIR}/bin/activate  (to re-activate later)"
