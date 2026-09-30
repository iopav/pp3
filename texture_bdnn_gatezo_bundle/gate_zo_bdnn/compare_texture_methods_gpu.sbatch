#!/bin/bash
#SBATCH --job-name=texGate3
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
DATA_ROOT="$PROJECT_DIR/data"
OUTPUT_NAME="${BDNN_OUTPUT_NAME:-texture_gatezo_protocols}"
OUTPUT_ROOT="$PROJECT_DIR/outputs/$OUTPUT_NAME"
CONTAINER_IMAGE="${BDNN_CONTAINER_IMAGE:-$PROJECT_DIR/bdnn_arm64.sif}"
CANDIDATE_POOL_SIZE="${BDNN_CANDIDATE_POOL_SIZE:-100}"
GROWTH_STEP="${BDNN_GROWTH_STEP:-3}"
MIN_KEEP="${BDNN_MIN_KEEP:-1}"
ZO_SPARSITY_LAMBDA="${BDNN_ZO_SPARSITY_LAMBDA:-0.0001}"
ADAPTIVE_TOLERANCE="${BDNN_ADAPTIVE_TOLERANCE:-0.001}"
COMPUTE_BATCH_SIZE="${BDNN_COMPUTE_BATCH_SIZE:-128}"

METHODS=(fixed_nozo adaptive_nozo gatezo)
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
METHOD="${METHODS[$METHOD_INDEX]}"
SEED=$((22 + SEED_INDEX))

cd "$PROJECT_DIR"
mkdir -p outputs "$OUTPUT_ROOT"

PROTOCOL="${BDNN_GROWTH_PROTOCOL:-legacy_test}"

echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID protocol=$PROTOCOL method=$METHOD seed=$SEED"
echo "data_root=$DATA_ROOT"
echo "output_root=$OUTPUT_ROOT"
nvidia-smi || true

apptainer exec --nv \
  --bind "$PROJECT_DIR:/work" \
  --pwd /work \
  --env PYTHONPATH=/work \
  "$CONTAINER_IMAGE" \
  python -u -m gate_zo_bdnn.run_texture_comparison_task \
    --selection-mode "$METHOD" \
    --growth-protocol "$PROTOCOL" \
    --seed "$SEED" \
    --data-root /work/data \
    --output-root "/work/outputs/$OUTPUT_NAME" \
    --candidate-pool-size "$CANDIDATE_POOL_SIZE" \
    --growth-step "$GROWTH_STEP" \
    --max-hidden 200 \
    --min-keep "$MIN_KEEP" \
    --zo-sparsity-lambda "$ZO_SPARSITY_LAMBDA" \
    --adaptive-tolerance "$ADAPTIVE_TOLERANCE" \
    --zo-sample-size 0 \
    --calibration-ratio 0.5 \
    --validation-ratio 0.2 \
    --ridge 0 \
    --growth-min-delta 0.0001 \
    --failure-patience 10 \
    --device cuda \
    --compute-batch-size "$COMPUTE_BATCH_SIZE"
