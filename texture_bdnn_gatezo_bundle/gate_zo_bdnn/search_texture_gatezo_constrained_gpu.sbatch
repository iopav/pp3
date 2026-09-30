#!/bin/bash
#SBATCH --job-name=texGconst
#SBATCH --account=naiss2026-4-902-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-24%6
#SBATCH --output=outputs/slurm-%x-%A_%a.out
#SBATCH --error=outputs/slurm-%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${BDNN_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONTAINER_IMAGE="${BDNN_CONTAINER_IMAGE:-$PROJECT_DIR/bdnn_arm64.sif}"
SEARCH_ROOT=texture_gatezo_constrained_search

CONFIG_NAMES=(eps0000 eps0001 eps0003 eps0005 eps0010)
TOLERANCES=(0 0.0001 0.0003 0.0005 0.001)

CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
SEED=$((22 + SEED_INDEX))
CONFIG_NAME="${CONFIG_NAMES[$CONFIG_INDEX]}"
TOLERANCE="${TOLERANCES[$CONFIG_INDEX]}"

cd "$PROJECT_DIR"
mkdir -p outputs

echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID seed=$SEED config=$CONFIG_NAME"
echo "gate_rule=full_block_loss_constraint epsilon=$TOLERANCE gate_estimates_per_round=1 no_repeats=true"
nvidia-smi || true

apptainer exec --nv \
  --bind "$PROJECT_DIR:/work" \
  --pwd /work \
  --env PYTHONPATH=/work \
  "$CONTAINER_IMAGE" \
  python -u -m gate_zo_bdnn.run_texture_comparison_task \
    --selection-mode gatezo \
    --growth-protocol validation_checkpoint \
    --seed "$SEED" \
    --data-root /work/data \
    --output-root "/work/outputs/$SEARCH_ROOT/$CONFIG_NAME" \
    --candidate-pool-size 100 \
    --growth-step 5 \
    --max-hidden 200 \
    --min-keep 1 \
    --zo-sparsity-lambda 0 \
    --zo-loss-tolerance "$TOLERANCE" \
    --adaptive-tolerance 0.001 \
    --zo-sample-size 0 \
    --calibration-ratio 0.5 \
    --validation-ratio 0.2 \
    --ridge 0 \
    --growth-min-delta 0.0001 \
    --failure-patience 10 \
    --device cuda \
    --compute-batch-size 256
