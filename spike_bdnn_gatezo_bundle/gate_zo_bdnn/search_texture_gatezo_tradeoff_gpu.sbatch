#!/bin/bash
#SBATCH --job-name=texGtrade
#SBATCH --account=naiss2026-4-902-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-39%6
#SBATCH --output=outputs/slurm-%x-%A_%a.out
#SBATCH --error=outputs/slurm-%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${BDNN_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONTAINER_IMAGE="${BDNN_CONTAINER_IMAGE:-$PROJECT_DIR/bdnn_arm64.sif}"
SEARCH_ROOT=texture_gatezo_tradeoff_search

# Eight deliberately narrow configurations around the observed accuracy/size
# boundary.  Every task performs one gate estimate only; there are no repeats.
CONFIG_NAMES=(
  k2_l0000
  k2_l00005
  k2_l0001
  k2_l0002
  k2_l0003
  k3_l00005
  k3_l0001
  k3_l0002
)
MIN_KEEPS=(2 2 2 2 2 3 3 3)
LAMBDAS=(0 0.00005 0.0001 0.0002 0.0003 0.00005 0.0001 0.0002)

CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID % 8))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 8))
SEED=$((22 + SEED_INDEX))
CONFIG_NAME="${CONFIG_NAMES[$CONFIG_INDEX]}"
MIN_KEEP="${MIN_KEEPS[$CONFIG_INDEX]}"
SPARSITY_LAMBDA="${LAMBDAS[$CONFIG_INDEX]}"
OUTPUT_ROOT="/work/outputs/$SEARCH_ROOT/$CONFIG_NAME"

cd "$PROJECT_DIR"
mkdir -p outputs

echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID seed=$SEED config=$CONFIG_NAME"
echo "method=gatezo pool=100 preselect=5 min_keep=$MIN_KEEP lambda=$SPARSITY_LAMBDA gate_estimates_per_round=1 no_repeats=true"
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
    --output-root "$OUTPUT_ROOT" \
    --candidate-pool-size 100 \
    --growth-step 5 \
    --max-hidden 200 \
    --min-keep "$MIN_KEEP" \
    --zo-sparsity-lambda "$SPARSITY_LAMBDA" \
    --adaptive-tolerance 0.001 \
    --zo-sample-size 0 \
    --calibration-ratio 0.5 \
    --validation-ratio 0.2 \
    --ridge 0 \
    --growth-min-delta 0.0001 \
    --failure-patience 10 \
    --device cuda \
    --compute-batch-size 256
