#!/usr/bin/env bash
# Edit these resource requests for your cluster.
#SBATCH --job-name=spsn-tp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
# Activate your existing Python/PyTorch environment here.
: "${PP3_DATA_ROOT:?Set PP3_DATA_ROOT to the dataset parent directory}"
python run_experiments.py --data-root "$PP3_DATA_ROOT" --workers 2 --threads 2 --output outputs/slurm
