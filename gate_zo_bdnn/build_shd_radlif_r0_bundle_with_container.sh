#!/bin/bash
set -euo pipefail

# Run from the BDNNversion1 project root, or pass the project root as argument 1.
PROJECT_DIR="${1:-$PWD}"
OUTPUT="${2:-$PROJECT_DIR/shd_radlif_r0_gatezo_vs_no_gatezo_with_container.tar}"
cd "$PROJECT_DIR"

FILES=(
  bdnn_arm64.sif
  gate_zo_bdnn/__init__.py
  gate_zo_bdnn/configs_shd_radlif_r0_gatezo_vs_no_gatezo_5seeds.json
  gate_zo_bdnn/run_shd_radlif_r0_gatezo_vs_no_gatezo_5seeds_gpu.sbatch
  gate_zo_bdnn/summarize_shd_radlif_r0_gatezo_vs_no_gatezo.py
  gate_zo_bdnn/README_shd_radlif_r0_gatezo_vs_no_gatezo.md
  gate_zo_bdnn/SHD_R0_GATEZO_VS_NO_GATEZO_RUNBOOK_CN.md
  gate_zo_bdnn/build_shd_radlif_r0_bundle_with_container.sh
  gate_zo_bdnn/run_shd_adlif_gatezo_task.py
  gate_zo_bdnn/shd_structural_zo_search_space.py
  gate_zo_bdnn/train_shd_radlif_residual_proposal.py
  gate_zo_bdnn/shd_radlif_residual_proposal_core.py
  gate_zo_bdnn/shd_radlif_advanced_gatezo_core.py
  gate_zo_bdnn/shd_radlif_delay_gatezo_core.py
  gate_zo_bdnn/shd_adlif_gatezo_core.py
  gate_zo_bdnn/train_shd_individual_pool_structural_zo.py
  gate_zo_bdnn/shd_gatezo_support.py
  gate_zo_bdnn/train_shd_bdnn_rsnn_blocks.py
  gate_zo_bdnn/train_shd_bdnn_rsnn_pure.py
  gate_zo_bdnn/tests/test_shd_radlif_r0_gatezo_vs_no_gatezo_package.py
)

for path in "${FILES[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required bundle file: $PROJECT_DIR/$path" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT" ]]; then
  echo "refusing to overwrite existing archive: $OUTPUT" >&2
  exit 3
fi

tar -cf "$OUTPUT" "${FILES[@]}"
echo "created $OUTPUT"
ls -lh "$OUTPUT"
sha256sum "$OUTPUT"
echo "SHD HDF5 data, python_deps, and experiment outputs are not included."
