#!/bin/bash
#SBATCH --account=naiss2026-4-902-gpu
#SBATCH --job-name=texture-five
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=texture-%j.out
#SBATCH --error=texture-%j.err

set -euo pipefail
# Submit from the repository root. Use an already activated Python environment.
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python -u texture_experiments/run_all.py --data-root /home/yizhil/pp3/data/texture-mini "$@"
