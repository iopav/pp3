#!/usr/bin/env bash
# Arrhenius GPU template. Override resource options with sbatch flags if needed.
#SBATCH --account=naiss2026-4-1562-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --job-name=pp3
#SBATCH --output=pp3-%j.out
#SBATCH --error=pp3-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the pp3 project directory}"

# CONDA_BASE may be exported explicitly before sbatch. No installation in this job.
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

# Respect Slurm device allocation. Do not pass host GPU numbers manually.
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose a GPU allocation}"
python -u run_experiments.py --output result --workers 2 --threads 2 "$@"
