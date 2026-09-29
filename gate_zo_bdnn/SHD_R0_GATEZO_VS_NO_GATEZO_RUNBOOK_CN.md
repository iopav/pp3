# SHD R0 RadLIF：GateZO 与无 GateZO 对照实验运行手册

## 1. 实验目的

本实验在完全相同的冻结 RadLIF、数据增强、随机候选、读出和停止规则下，比较：

1. `R0_fixed_prior_gatezo`：使用 GateZO 对预筛候选做坐标消融和稀疏裁剪；
2. `R0_fixed_prior_no_gatezo_fixed16`：不做 GateZO，每轮直接保留最多 16 个预筛候选。

两组实验均不使用 BPTT，不通过梯度训练隐藏层参数。隐藏神经元和连接随机生成后冻结，输出权重由闭式 ridge 求解。

无 GateZO 版本仍保留：

- 残差相关性与多样性预筛；
- speaker-disjoint validation 外层增长门；
- 每轮生成 512 个随机候选；
- 最多提交 16 个预筛候选；
- 随机冻结 RadLIF 动力学；
- 闭式 ridge readout。

因此，这个对照只回答“GateZO 坐标裁剪是否有贡献”，不是“有结构学习”和“完全随机一次生成”之间的对比。

## 2. 两组配置的唯一差异

| 配置 | `selection_mode` | 每轮预筛后的处理 |
|---|---|---|
| `R0_fixed_prior_gatezo` | `gatezo` | 对最多 16 个候选做单坐标硬消融，结合稀疏惩罚选择至少 8 个 |
| `R0_fixed_prior_no_gatezo_fixed16` | `fixed16` | 不做坐标消融，保留全部预筛候选，最多 16 个 |

除 `config_id` 和 `selection_mode` 外，两份 JSON 配置逐字段完全相同。包内测试会检查这一点。

这里没有使用 `adaptive_nozo`，因为它仍会根据内部 loss 搜索保留前缀大小，不能干净隔离 GateZO 的贡献。

## 3. 公共实验配置

### 3.1 数据与随机种子

```text
dataset: SHD
seeds: 57, 58, 59, 60, 61
input channels: 700
validation speakers: 2（每个 seed 确定性选择）
test set: 不参与拟合、结构选择或停止
```

### 3.2 数据增强

```text
channel jitter std: 20
jitter copies: 1
time-scale range: 0.90–1.10
```

每个结构训练样本保留原始版本，并产生一个增强副本。通道抖动和时间缩放只用于训练副本，不用于 speaker-validation 或 test。

### 3.3 时间离散和读出

```text
dt: 0.0005 s（0.5 ms）
max_time: 1.0 s
readout_bins: 100
readout_warmup_bins: 5
temporal_windows: 16
temporal_pooling: window_mean
ridge: 0.5
threshold: 1.0
```

隐藏轨迹在去除前 5 个 readout bins 后划分为 16 个时间窗口，窗口均值拼接后用于闭式 ridge 分类。

### 3.4 RadLIF 动力学

```text
tau_adapt: log-uniform 60–200 ms
adapt_a: uniform -0.5–0.5
adapt_b: uniform 0.05–0.50

heterogeneous_tau_fraction: 0.5
tau_syn fixed/default: 10 ms
heterogeneous tau_syn: log-uniform 5–25 ms
tau_mem fixed/default: 20 ms
heterogeneous tau_mem: log-uniform 15–60 ms
```

50% 候选使用异质 `tau_syn/tau_mem`，其余候选使用固定的 10/20 ms。所有参数生成后冻结。

### 3.5 四分支输入与稀疏延迟

```text
input_branches: 4
input_delay_ms: 0, 10, 25, 50, 100, 200
frequency_bandwidths: 32, 64, 128, 256
input_channel_group_size: 1
```

- 分支 1：覆盖全部 700 个通道，延迟为 0；
- 分支 2–4：各自随机选择连续局部频带；
- 局部分支从非零延迟 `10/25/50/100/200 ms` 中随机选择；
- 各分支权重随机生成后冻结。

### 3.6 候选、增长和停止

```text
candidate_pool_size: 512
growth_step: 16
input_scales: 2, 3, 4, 6
self_radii: 0.7, 0.9, 1.1, 1.3（随机正负号）
old_to_new_scale: 0.5
bias_scale: 0.20
candidate firing-rate range: 0.0005–0.30
diversity_strength: 0.7
growth_min_delta: 0.002
growth_failure_patience: 12
max_growth_rounds: 120
max_hidden: 0（不设置显式上限）
```

新候选可读取旧神经元脉冲，但不会反向改变旧神经元，因此已提交隐藏特征在后续增长中保持不变。

### 3.7 GateZO 专用参数

```text
zo_sample_size: 2048
zo_calibration_ratio: 0.20
zo_min_keep: 8
zo_sparsity_lambda: 0.000025
```

这些字段也保留在无 GateZO JSON 中，以保证配置匹配；`selection_mode=fixed16` 时不会执行 GateZO 坐标消融。

### 3.8 提议模式

```text
proposal_mode: prior
```

`prior` 是残差引导提议的显式关闭开关。因此本实验使用固定随机先验，不使用 OOF residual-adaptive proposal。JSON 中的 `guided_proposal_*` 字段在该模式下不会引导候选分布。

## 4. 包内文件与功能

| 文件 | 功能 |
|---|---|
| `configs_shd_radlif_r0_gatezo_vs_no_gatezo_5seeds.json` | 两组匹配实验配置 |
| `run_shd_radlif_r0_gatezo_vs_no_gatezo_5seeds_gpu.sbatch` | 提交 10 个 Slurm GPU 数组任务 |
| `summarize_shd_radlif_r0_gatezo_vs_no_gatezo.py` | 输出逐 seed、均值、标准差和配对差值 |
| `run_shd_adlif_gatezo_task.py` | 把数组编号映射为配置和 seed，并组装训练命令 |
| `train_shd_radlif_residual_proposal.py` | 冻结 RadLIF 训练入口 |
| `shd_radlif_residual_proposal_core.py` | 固定/残差提议后端；本实验使用 `prior` |
| `shd_radlif_advanced_gatezo_core.py` | 异质 tau、四分支和局部频带实现 |
| `shd_radlif_delay_gatezo_core.py` | 延迟 RadLIF 基础状态和动力学 |
| `shd_adlif_gatezo_core.py` | adLIF/RadLIF 适应电流基础实现 |
| `train_shd_individual_pool_structural_zo.py` | 增长、GateZO/fixed16、checkpoint 和最终评估 |
| `shd_gatezo_support.py` | 候选预筛、ridge、采样和特征支持函数 |
| `train_shd_bdnn_rsnn_blocks.py` | SHD speaker split、增强与 DataLoader |
| `train_shd_bdnn_rsnn_pure.py` | SHD 事件读取、分箱和基础 LIF 支持 |
| `shd_structural_zo_search_space.py` | JSON 配置读取与校验 |
| `bdnn_arm64.sif` | Arrhenius ARM64/GH200 运行容器；仅完整集群 TAR 包含 |
| `build_shd_radlif_r0_bundle_with_container.sh` | 在集群端把源码与实际 ARM64 SIF 打成完整 TAR |

调用链：

```text
Slurm数组
  → run_shd_adlif_gatezo_task.py
  → train_shd_radlif_residual_proposal.py
  → train_shd_individual_pool_structural_zo.py
  → RadLIF dynamics backend
  → final_structural_zo.pt + summary.json
```

## 5. 包的两种形式

### 5.1 源码 ZIP

源码 ZIP 包含全部项目内部 Python 依赖，但不包含：

```text
bdnn_arm64.sif
data/SHD/shd_train.h5
data/SHD/shd_test.h5
python_deps中的外部安装内容
历史outputs
```

### 5.2 含容器的完整 TAR

完整 TAR 应在集群项目根目录生成，并在根层级包含：

```text
bdnn_arm64.sif
gate_zo_bdnn/...
```

`.sif` 本身已经压缩，因此推荐使用未再次压缩的 `.tar`，避免耗费大量时间而几乎不减小体积。SHD 数据因体积和许可问题仍单独存放。

在集群项目根目录生成完整 TAR：

```bash
cd /nobackup/proj/disk/naiss2026-4-902/personal/zhenbangdai/Zhenbang/BDNNversion1
bash gate_zo_bdnn/build_shd_radlif_r0_bundle_with_container.sh
```

该命令的功能是：

- 检查真实的 `bdnn_arm64.sif` 和全部源码是否存在；
- 创建 `shd_radlif_r0_gatezo_vs_no_gatezo_with_container.tar`；
- 不再次压缩已经压缩的 SIF；
- 打印归档大小和 SHA256；
- 如果同名 TAR 已存在则拒绝覆盖。

指定项目目录和输出路径：

```bash
bash gate_zo_bdnn/build_shd_radlif_r0_bundle_with_container.sh \
  /nobackup/proj/disk/naiss2026-4-902/personal/zhenbangdai/Zhenbang/BDNNversion1 \
  /nobackup/proj/disk/naiss2026-4-902/personal/zhenbangdai/Zhenbang/r0_comparison_with_container.tar
```

## 6. 解包和运行前检查

### 6.1 进入项目目录

```bash
cd /nobackup/proj/disk/naiss2026-4-902/personal/zhenbangdai/Zhenbang/BDNNversion1
```

### 6.2 解压源码 ZIP

```bash
unzip shd_radlif_r0_gatezo_vs_no_gatezo_5seeds_bundle.zip
```

### 6.3 解开含容器的完整 TAR

```bash
tar -xf shd_radlif_r0_gatezo_vs_no_gatezo_with_container.tar
```

不要同时把同一套旧文件和新文件并行写入项目目录。解包前应确认目标路径正确。

### 6.4 检查必要文件

```bash
test -f bdnn_arm64.sif &&
test -f data/SHD/shd_train.h5 &&
test -f data/SHD/shd_test.h5 &&
test -f gate_zo_bdnn/configs_shd_radlif_r0_gatezo_vs_no_gatezo_5seeds.json &&
test -f gate_zo_bdnn/run_shd_radlif_r0_gatezo_vs_no_gatezo_5seeds_gpu.sbatch &&
echo "all required artifacts found"
```

检查大小：

```bash
ls -lh bdnn_arm64.sif data/SHD/shd_train.h5 data/SHD/shd_test.h5
```

检查 JSON：

```bash
python3 -m json.tool \
  gate_zo_bdnn/configs_shd_radlif_r0_gatezo_vs_no_gatezo_5seeds.json >/dev/null &&
echo "config JSON OK"
```

## 7. 提交实验

```bash
sbatch gate_zo_bdnn/run_shd_radlif_r0_gatezo_vs_no_gatezo_5seeds_gpu.sbatch
```

Slurm 返回示例：

```text
Submitted batch job 1234567
```

资源配置：

```text
partition: gpu
GPU: 1/task
CPU: 8/task
memory: 128 GB/task
time limit: 72 hours/task
array: 0-9%8
```

`%8` 表示最多同时运行 8 个任务，不是每个任务使用 8 张 GPU。每个任务只申请 1 张 GPU。

## 8. 数组任务映射

配置编号变化最快：

| Task ID | Seed | 配置 |
|---:|---:|---|
| 0 | 57 | GateZO |
| 1 | 57 | no-GateZO fixed16 |
| 2 | 58 | GateZO |
| 3 | 58 | no-GateZO fixed16 |
| 4 | 59 | GateZO |
| 5 | 59 | no-GateZO fixed16 |
| 6 | 60 | GateZO |
| 7 | 60 | no-GateZO fixed16 |
| 8 | 61 | GateZO |
| 9 | 61 | no-GateZO fixed16 |

只提交某一个任务进行冒烟测试：

```bash
sbatch --array=0 gate_zo_bdnn/run_shd_radlif_r0_gatezo_vs_no_gatezo_5seeds_gpu.sbatch
```

只提交 seed57 的两个方法：

```bash
sbatch --array=0-1%2 gate_zo_bdnn/run_shd_radlif_r0_gatezo_vs_no_gatezo_5seeds_gpu.sbatch
```

## 9. 查看队列和资源状态

查看自己的任务：

```bash
squeue --me
```

查看指定作业：

```bash
squeue -j JOBID
```

查看数组子任务状态、耗时和退出码：

```bash
sacct -j JOBID -X -n -P \
  --format=JobID,State,Elapsed,ElapsedRaw,ExitCode
```

常见状态：

| 状态 | 含义 |
|---|---|
| `R` | 正在运行 |
| `PD (Resources)` | 等待可用 GPU/内存等资源 |
| `PD (AssocGrpGRESRunMinutes)` | 账户当前 GPU 运行分钟配额限制 |
| `PD (PartitionTimeLimit)` | 请求时限超过分区允许值，应调整 `#SBATCH --time` |
| `COMPLETED` | 正常结束 |
| `FAILED` | 程序或环境报错 |
| `TIMEOUT` | 达到时限；可重新提交并从 latest checkpoint 恢复 |

## 10. 查看日志

日志命名：

```text
outputs/slurm-shd-r0-gzcmp-JOBID_TASK.out
outputs/slurm-shd-r0-gzcmp-JOBID_TASK.err
```

实时查看 task 0：

```bash
tail -f outputs/slurm-shd-r0-gzcmp-JOBID_0.out
```

查看最近输出：

```bash
tail -n 100 outputs/slurm-shd-r0-gzcmp-JOBID_0.out
```

查看错误：

```bash
tail -n 100 outputs/slurm-shd-r0-gzcmp-JOBID_0.err
```

如果不确定日志路径：

```bash
scontrol show job JOBID_TASK | tr ' ' '\n' | grep -E '^(JobName|StdOut|StdErr|WorkDir)='
```

## 11. 取消任务

取消整个数组：

```bash
scancel JOBID
```

取消一个数组子任务：

```bash
scancel JOBID_TASK
```

## 12. Checkpoint 与恢复

任务分发程序始终传入 `--resume`。训练过程中会保存：

```text
latest_structural_zo.pt
```

如果任务因时限或节点故障中断，重新提交相同 task ID，会从相同输出目录中的 latest checkpoint 恢复，而不是从第 0 轮重新开始。

不要把另一个配置或 seed 的 checkpoint 复制到该目录。恢复时程序会检查结构分区等状态是否匹配。

## 13. 输出目录与文件

```text
outputs/shd_radlif_r0_gatezo_vs_no_gatezo_5seeds/
├── R0_fixed_prior_gatezo/
│   ├── seed57/
│   ├── seed58/
│   ├── seed59/
│   ├── seed60/
│   └── seed61/
└── R0_fixed_prior_no_gatezo_fixed16/
    ├── seed57/
    ├── seed58/
    ├── seed59/
    ├── seed60/
    └── seed61/
```

每个 seed 目录主要包含：

| 文件 | 含义 |
|---|---|
| `search_config.json` | 该任务使用的完整配置和 seed |
| `latest_structural_zo.pt` | 可恢复的增长中间 checkpoint |
| `final_structural_zo.pt` | 最终冻结状态、ridge 参数和历史 |
| `summary.json` | 最终训练、验证、测试准确率和结构统计 |

`summary.json` 的关键字段：

```text
selection_mode
hidden_size
structure_training
structure_validation
full_train_refit
test
accepted_rounds
attempted_rounds
mean_kept_per_accepted_round
total_preselected_on_accepted_rounds
total_pruned_on_accepted_rounds
zo_steps_per_growth
zo_coordinate_queries_per_growth
held_out_speakers
checkpoint
test_used_for_fitting_or_selection
```

无 GateZO 结果应显示：

```text
selection_mode = fixed16
zo_steps_per_growth = 0
zo_coordinate_queries_per_growth = 0
```

## 14. 汇总结果

任务未全部结束时也可以运行，缺失项会显示为 `incomplete`：

```bash
python3 gate_zo_bdnn/summarize_shd_radlif_r0_gatezo_vs_no_gatezo.py
```

汇总输出包括：

- 每个 seed 的 GateZO 测试准确率和 hidden 数量；
- 每个 seed 的无 GateZO 测试准确率和 hidden 数量；
- 配对差值 `GateZO − noGateZO`；
- 两种方法的五 seed 均值和样本标准差；
- GateZO 的配对胜场数；
- `paired_summary.json`。

汇总文件：

```text
outputs/shd_radlif_r0_gatezo_vs_no_gatezo_5seeds/paired_summary.json
```

## 15. 单独读取一个结果

例如查看 seed57 GateZO：

```bash
python3 - <<'PY'
import json
path = "outputs/shd_radlif_r0_gatezo_vs_no_gatezo_5seeds/R0_fixed_prior_gatezo/seed57/summary.json"
row = json.load(open(path))
print("accuracy", f"{100*row['test']['accuracy']:.4f}%")
print("hidden", row["hidden_size"])
print("accepted/attempted", row["accepted_rounds"], row["attempted_rounds"])
print("selection_mode", row["selection_mode"])
PY
```

## 16. 命令功能速查

| 命令 | 功能 |
|---|---|
| `sbatch ...sbatch` | 提交全部 10 个数组任务 |
| `sbatch --array=0 ...sbatch` | 仅提交 GateZO seed57 冒烟测试 |
| `squeue --me` | 查看当前排队和运行任务 |
| `sacct -j JOBID ...` | 查看已结束/失败/超时任务 |
| `tail -f ...out` | 实时查看标准输出 |
| `tail -n 100 ...err` | 查看最近错误 |
| `scancel JOBID` | 取消整个数组 |
| `scancel JOBID_TASK` | 取消一个子任务 |
| `python3 ...summarize...py` | 汇总当前所有可用结果 |
| `tar -tf PACKAGE.tar` | 查看完整 TAR 内容，不解包 |
| `sha256sum PACKAGE.tar` | 计算包校验值 |
| `bash gate_zo_bdnn/build_shd_radlif_r0_bundle_with_container.sh` | 生成包含真实 ARM64 容器的完整 TAR |

## 17. 容器架构注意事项

`bdnn_arm64.sif` 是 ARM64 容器。Arrhenius 登录节点可能是 AMD64，因此在登录节点直接运行：

```bash
apptainer exec bdnn_arm64.sif python ...
```

可能出现：

```text
the image's architecture (arm64) could not run on the host's (amd64)
```

这不表示容器损坏。正常训练通过 Slurm GPU 脚本运行在兼容的 GH200/ARM64 计算节点。不要在 AMD64 登录节点直接执行该容器。

## 18. 科研解释边界

- 隐藏层没有梯度训练；
- GateZO 只用于候选结构选择，不更新冻结权重；
- `fixed16` 不是“没有任何结构筛选”，它仍使用公共预筛和外层验证门；
- test 不参与候选选择、增长接受、停止或 ridge 拟合；
- 最终 test 只用于报告两种预先声明配置的结果；
- 比较时应使用同 seed 配对差值，同时报告五 seed 均值与标准差。

如果后续根据这批 test 结果继续选择新的超参数，那么后续结果应标记为 exploratory test result，而不能再称为未触碰测试集的最终估计。
