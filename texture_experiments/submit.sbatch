#!/bin/bash
#SBATCH --account=naiss2026-4-902-gpu
#SBATCH --partition=gpu
#SBATCH --job-name=texture-six
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=texture-%j.out
#SBATCH --error=texture-%j.err

set -euo pipefail
# Submit from the repository root.
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"

# Match submit_experiments.sbatch: activate the existing cluster environment.
if [[ -z "${CONDA_BASE:-}" ]]; then
    if [[ -n "${CONDA_EXE:-}" ]]; then
        CONDA_BASE="$("$CONDA_EXE" info --base)"
    elif command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
    else
        echo 'Set CONDA_BASE to the existing conda installation directory before sbatch.' >&2
        exit 1
    fi
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate avalanche

python -u -c 'import sys; print("Python:", sys.executable, flush=True); import numpy; import torch; print("NumPy:", numpy.__version__, "PyTorch:", torch.__version__, flush=True)'

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python -u texture_experiments/run_all.py "$@"
 
