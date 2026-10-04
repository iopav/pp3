#!/bin/bash
#SBATCH --job-name=texG5k2
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
export BDNN_GROWTH_PROTOCOL=validation_checkpoint
export BDNN_OUTPUT_NAME=texture_gatezo_preselect5_keep2_l0003
export BDNN_CANDIDATE_POOL_SIZE=100
export BDNN_GROWTH_STEP=5
export BDNN_MIN_KEEP=2
export BDNN_ZO_SPARSITY_LAMBDA=0.0003
export BDNN_ADAPTIVE_TOLERANCE=0.001
export BDNN_COMPUTE_BATCH_SIZE=256

cd "$PROJECT_DIR"
bash gate_zo_bdnn/compare_texture_methods_gpu.sbatch
