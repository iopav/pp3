#!/bin/bash
#SBATCH --job-name=texLegacy
#SBATCH --account=naiss2026-4-902-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-29%6
#SBATCH --output=outputs/slurm-%x-%A_%a.out
#SBATCH --error=outputs/slurm-%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${BDNN_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export BDNN_GROWTH_PROTOCOL=legacy_test
cd "$PROJECT_DIR"
bash gate_zo_bdnn/compare_texture_methods_gpu.sbatch
