# Structural gate-ZO BDNN

This folder adds the new coordinate structural gate-ZO method without changing
`bdnn1.py`, `bdnn2.py`, `zo_refinement.py`, or the original entry points.

For every growth round it:

1. lets the original BDNN inequality screen propose `nB` neurons;
2. temporarily adds the complete proposed recurrent block;
3. evaluates one hard ablation for every new neuron;
4. ranks neurons by `loss_without_i - loss_full`;
5. evaluates exact ranked prefixes from `gate_zo_min_keep` through `nB`;
6. minimizes `calibration_cross_entropy + lambda * kept_neurons`;
7. commits the selected block and lets the original BDNN training-accuracy
   window decide when growth stops;
8. restores the historical best checkpoint at the end.

Two checkpoint protocols are available. `legacy_test` exactly reproduces the
original test-selected best checkpoint and therefore records
`test_used_for_selection=true`. `validation_checkpoint` uses a held-out part of
training for checkpoint selection, refits the readout on all training samples,
and evaluates test only once.

The comparison classes use `cuda_backend.py` for both BDNN candidate simulation
and hidden-state forward passes.  The original CPU equations and NumPy random
candidate generation are retained.  Server jobs explicitly pass `--device
cuda`; `--compute-batch-size` controls GPU memory use.

Run from the repository root:

```powershell
$env:BDNN_SEED = "22"
python -m gate_zo_bdnn.main_exp1_gatezo
python -m gate_zo_bdnn.main_exp2_gatezo
```

Outputs are isolated under:

```text
outputs/gate_zo_bdnn/current/exp1
outputs/gate_zo_bdnn/current/exp2
```

To reproduce a seed sweep without overwriting runs:

```powershell
$env:BDNN_SEEDS = "22,23,24,25,26"
python -m gate_zo_bdnn.run_seed_sweep_gatezo
```

Every seed is written to `outputs/gate_zo_bdnn/seed_<seed>` and the paired
summary is written to `outputs/gate_zo_bdnn/seed_sweep_summary.csv`.

The main controls are in `config_gatezo.py`.  With the original `nB=3`, the
default `gate_zo_min_keep=1` lets each round retain one, two, or three neurons.
For a fair no-gate baseline, keep the same seed and all other settings but use
the untouched original programs.

## Matched three-method server comparison

`compare_texture_methods_gpu.sbatch` runs ten paired seeds (22--31) for:

- `fixed_nozo`: retain all three BDNN-screened neurons;
- `adaptive_nozo`: choose the smallest original-order prefix within the
  calibration-loss tolerance, without coordinate ablations;
- `gatezo`: use one hard-ablation query per proposed neuron and select the
  L0-penalized importance prefix.

Submit the original-code reproduction from the project root:

```bash
sed -i 's/\r$//' gate_zo_bdnn/*.sbatch
sbatch gate_zo_bdnn/compare_texture_methods_legacy_gpu.sbatch
```

Submit the leakage-free validation-checkpoint group separately:

```bash
sbatch gate_zo_bdnn/compare_texture_methods_validation_gpu.sbatch
```

The array has 30 tasks and runs at most six simultaneously.  To begin with a
three-seed pilot, override the array at submission:

```bash
sbatch --array=0-8%6 gate_zo_bdnn/compare_texture_methods_legacy_gpu.sbatch
sbatch --array=0-8%6 gate_zo_bdnn/compare_texture_methods_validation_gpu.sbatch
```

Summarize all completed paired seeds:

```bash
python3 gate_zo_bdnn/summarize_texture_comparison.py \
  --root outputs/texture_gatezo_protocols/legacy_test

python3 gate_zo_bdnn/summarize_texture_comparison.py \
  --root outputs/texture_gatezo_protocols/validation_checkpoint
```

## Five-neuron preselection experiment

`compare_texture_preselect5_validation_gpu.sbatch` keeps the candidate pool at
100, preselects five BDNN neurons per growth round, lets gate-ZO retain 1--5,
and uses `lambda=0.001`.  It uses the leakage-free validation-checkpoint
protocol and writes to a new output root:

```bash
sbatch gate_zo_bdnn/compare_texture_preselect5_validation_gpu.sbatch

python3 gate_zo_bdnn/summarize_texture_comparison.py \
  --root outputs/texture_gatezo_preselect5/validation_checkpoint
```

The single-estimate accuracy-recovery configuration uses five preselected
neurons, keeps at least two, applies `lambda=0.0003`, and does not repeat the
gate estimate:

```bash
sbatch gate_zo_bdnn/compare_texture_preselect5_keep2_l0003_validation_gpu.sbatch

python3 gate_zo_bdnn/summarize_texture_comparison.py \
  --root outputs/texture_gatezo_preselect5_keep2_l0003/validation_checkpoint
```

## One-estimate accuracy/size trade-off search

The targeted search keeps the pool at 100 and preselection at five.  It tests
eight low-penalty `min_keep`/lambda settings on seeds 22--26.  Each growth
round uses exactly one gate estimate (no repeats), and configuration selection
uses validation accuracy and hidden size rather than test accuracy:

```bash
sbatch gate_zo_bdnn/search_texture_gatezo_tradeoff_gpu.sbatch

python3 gate_zo_bdnn/summarize_texture_gatezo_tradeoff.py
```

The default rule accepts configurations whose mean validation accuracy is at
most 0.2 percentage points below `adaptive_nozo`, then recommends the one with
the smallest mean hidden size.  The adaptive baseline is read from the earlier
matched preselect-five experiment.
