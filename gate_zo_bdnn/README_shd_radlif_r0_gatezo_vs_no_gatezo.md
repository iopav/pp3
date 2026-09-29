# R0 frozen-RadLIF: GateZO versus no GateZO

完整中文运行、配置和命令说明见：

```text
gate_zo_bdnn/SHD_R0_GATEZO_VS_NO_GATEZO_RUNBOOK_CN.md
```

This package contains one matched two-method, five-seed SHD experiment.

- `R0_fixed_prior_gatezo`: the previously reported R0 setup with GateZO.
- `R0_fixed_prior_no_gatezo_fixed16`: the identical setup with GateZO coordinate
  ablation/sparsity pruning disabled. It keeps every residual/diversity
  preselected candidate, up to the fixed growth step of 16.

The no-GateZO control deliberately uses `selection_mode=fixed16`, not
`adaptive_nozo`. The latter still searches prefix sizes using internal loss and
would not isolate removal of GateZO as cleanly.

Both methods retain random frozen RadLIF neurons, closed-form ridge, the outer
speaker-validation growth gate, channel jitter, time scaling, heterogeneous
time constants, four input branches and sparse delays. Neither uses BPTT. The
only config differences are `config_id` and `selection_mode`.

## Cluster use

Unzip the bundle in the project root, then verify that these external artifacts
already exist:

```text
data/SHD/shd_train.h5
data/SHD/shd_test.h5
bdnn_arm64.sif
python_deps/
```

Submit ten tasks (two methods x five seeds; at most eight concurrent):

```bash
sbatch gate_zo_bdnn/run_shd_radlif_r0_gatezo_vs_no_gatezo_5seeds_gpu.sbatch
```

Task mapping (configuration varies fastest):

```text
task 0/1: seed57 GateZO/no-GateZO
task 2/3: seed58 GateZO/no-GateZO
task 4/5: seed59 GateZO/no-GateZO
task 6/7: seed60 GateZO/no-GateZO
task 8/9: seed61 GateZO/no-GateZO
```

Summarize partial or complete outputs:

```bash
python3 gate_zo_bdnn/summarize_shd_radlif_r0_gatezo_vs_no_gatezo.py
```

Outputs are written beneath:

```text
outputs/shd_radlif_r0_gatezo_vs_no_gatezo_5seeds/
```

源码 ZIP 不包含 SHD 数据、容器、依赖目录和历史输出。若使用在集群端生成的
`shd_radlif_r0_gatezo_vs_no_gatezo_with_container.tar`，则根目录同时包含
`bdnn_arm64.sif`；SHD 数据仍需单独提供。
