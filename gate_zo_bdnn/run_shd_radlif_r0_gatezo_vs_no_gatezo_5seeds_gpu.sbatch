#!/bin/bash
#SBATCH --job-name=shd-r0-gzcmp
#SBATCH --account=naiss2026-4-902-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --array=0-9%8
#SBATCH --output=outputs/slurm-%x-%A_%a.out
#SBATCH --error=outputs/slurm-%x-%A_%a.err

set -euo pipefail

PROJECT_DIR=/nobackup/proj/disk/naiss2026-4-902/personal/zhenbangdai/Zhenbang/BDNNversion1
CONFIG_FILE="$PROJECT_DIR/gate_zo_bdnn/configs_shd_radlif_r0_gatezo_vs_no_gatezo_5seeds.json"
cd "$PROJECT_DIR"
mkdir -p outputs python_deps
test -f "$CONFIG_FILE"
test -f "$PROJECT_DIR/data/SHD/shd_train.h5"
test -f "$PROJECT_DIR/data/SHD/shd_test.h5"
test -f "$PROJECT_DIR/bdnn_arm64.sif"

apptainer exec --nv \
  --bind "$PROJECT_DIR:/work" --pwd /work \
  --env PYTHONPATH=/work/python_deps:/work \
  --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  --env OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" \
  "$PROJECT_DIR/bdnn_arm64.sif" \
  python -u gate_zo_bdnn/run_shd_adlif_gatezo_task.py \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --output-root /work/outputs/shd_radlif_r0_gatezo_vs_no_gatezo_5seeds \
    --data-dir /work/data/SHD \
    --seeds 57 58 59 60 61 \
    --config-file /work/gate_zo_bdnn/configs_shd_radlif_r0_gatezo_vs_no_gatezo_5seeds.json \
    --expected-configs 2
