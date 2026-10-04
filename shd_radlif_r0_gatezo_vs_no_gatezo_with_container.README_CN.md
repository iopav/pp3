# SHD RadLIF R0：GateZO 与无 GateZO 实验包说明

## 1. 文件身份

本说明对应以下归档文件：

```text
shd_radlif_r0_gatezo_vs_no_gatezo_with_container.tar
```

当前经过校验的归档信息：

```text
显示大小：8.6 GiB
SHA-256：cc8bad28b912eb84564272d0b54941fa06ca17b3a3eb5d6eca0bec9b51cb5314
共享位置：/nobackup/proj/disk/naiss2026-4-902/shared/
```

校验文件：

```text
shd_radlif_r0_gatezo_vs_no_gatezo_with_container.tar.sha256
```

这份 TAR 是一个可移植实验包，用于在 Arrhenius ARM64/GH200 GPU 计算节点上比较：

1. `R0_fixed_prior_gatezo`：使用 GateZO 坐标消融和稀疏裁剪；
2. `R0_fixed_prior_no_gatezo_fixed16`：关闭 GateZO，每轮保留最多 16 个公共预筛候选。

两组配置除 `config_id` 和 `selection_mode` 外完全一致。它们都保持隐藏层无反向传播、候选神经元随机生成后冻结、闭式 ridge 训练输出层的基本原则。

## 2. 当前状态

归档已经复制到共享目录并通过 SHA-256 校验。当前不需要解压。

只检查完整性：

```bash
cd /nobackup/proj/disk/naiss2026-4-902/shared
sha256sum -c shd_radlif_r0_gatezo_vs_no_gatezo_with_container.tar.sha256
```

正确输出应为：

```text
shd_radlif_r0_gatezo_vs_no_gatezo_with_container.tar: OK
```

只查看归档目录而不解压：

```bash
tar -tf shd_radlif_r0_gatezo_vs_no_gatezo_with_container.tar | less
```

不要在只需要保存或传输归档时运行 `tar -xf`。

## 3. 包含内容

最终 TAR 包含：

```text
bdnn_arm64.sif
python_deps/h5py/
python_deps/h5py.libs/
python_deps/h5py-3.16.0.dist-info/
gate_zo_bdnn/实验源码
gate_zo_bdnn/两组匹配配置
gate_zo_bdnn/Slurm数组脚本
gate_zo_bdnn/结果汇总脚本
gate_zo_bdnn/测试
gate_zo_bdnn/中文运行手册
```

运行环境的来源如下：

| 组件 | 来源 |
|---|---|
| PyTorch 2.2.0a0 | `bdnn_arm64.sif` |
| NumPy 1.24.4 | `bdnn_arm64.sif` |
| CUDA 支持 | 容器与 GH200 计算节点 |
| h5py 3.16.0 | 包内 `python_deps/` |
| 项目 Python 源码 | 包内 `gate_zo_bdnn/` |

注意：最初的打包脚本输出文字可能写着 `python_deps ... not included`，但当前最终归档随后已经追加了 `python_deps`，并以本说明记录的 SHA-256 重新校验。判断最终内容应以 `tar -tf` 和当前哈希为准。

## 4. 不包含内容

归档不包含 SHD 数据集和历史实验输出：

```text
data/SHD/shd_train.h5
data/SHD/shd_test.h5
outputs/
```

运行前必须合法地另行准备两份 SHD HDF5 文件。预期目录为：

```text
PROJECT_ROOT/
├── bdnn_arm64.sif
├── python_deps/
├── data/
│   └── SHD/
│       ├── shd_train.h5
│       └── shd_test.h5
└── gate_zo_bdnn/
```

## 5. 实验公共配置

```text
dataset: SHD，700个输入通道
seeds: 57, 58, 59, 60, 61
candidate_pool_size: 512
growth_step: 16
growth_failure_patience: 12
growth_min_delta: 0.002
max_growth_rounds: 120
max_hidden: 0（无显式上限）
ridge: 0.5
temporal_windows: 16
temporal_pooling: window_mean
channel jitter: sigma=20, copies=1
time scale augmentation: 0.90–1.10
input branches: 4
input delays: 0, 10, 25, 50, 100, 200 ms
local frequency widths: 32, 64, 128, 256
heterogeneous tau fraction: 0.5
```

RadLIF 适应参数：

```text
tau_adapt: log-uniform 60–200 ms
a: uniform -0.5–0.5
b: uniform 0.05–0.50
```

异质时间常数：

```text
tau_syn: 50%固定10 ms，50% log-uniform 5–25 ms
tau_mem: 50%固定20 ms，50% log-uniform 15–60 ms
```

GateZO 配置额外执行：

```text
zo_sample_size: 2048
zo_min_keep: 8
zo_sparsity_lambda: 0.000025
```

无 GateZO 配置使用 `selection_mode=fixed16`，跳过坐标消融和 GateZO 稀疏裁剪，但仍保留公共的相关性/多样性预筛和 speaker-validation 外层增长门。因此它是“移除 GateZO”的对照，而不是完全没有结构选择的一次性随机网络。

## 6. 将来需要使用时如何解包

不要直接解压到共享目录根层。应先建立独立工作目录：

```bash
mkdir -p /nobackup/proj/disk/naiss2026-4-902/personal/$USER/shd_r0_gatezo_comparison

tar -xf \
  /nobackup/proj/disk/naiss2026-4-902/shared/shd_radlif_r0_gatezo_vs_no_gatezo_with_container.tar \
  -C /nobackup/proj/disk/naiss2026-4-902/personal/$USER/shd_r0_gatezo_comparison
```

然后把 SHD 数据放到该工作目录的 `data/SHD/` 下。

## 7. 运行前验证

```bash
cd /nobackup/proj/disk/naiss2026-4-902/personal/$USER/shd_r0_gatezo_comparison

test -f bdnn_arm64.sif &&
test -d python_deps/h5py &&
test -f data/SHD/shd_train.h5 &&
test -f data/SHD/shd_test.h5 &&
test -f gate_zo_bdnn/configs_shd_radlif_r0_gatezo_vs_no_gatezo_5seeds.json &&
echo "required files found"
```

`bdnn_arm64.sif` 是 ARM64 容器。AMD64 登录节点不能直接执行它，应通过 Slurm 提交到兼容的 GPU 计算节点。

## 8. 提交、监控和汇总

提交 2 个配置 × 5 个 seed，共 10 个任务，最多同时运行 8 个：

```bash
sbatch gate_zo_bdnn/run_shd_radlif_r0_gatezo_vs_no_gatezo_5seeds_gpu.sbatch
```

查看状态：

```bash
squeue --me
```

查看某个数组任务日志：

```bash
tail -f outputs/slurm-shd-r0-gzcmp-JOBID_TASK.out
tail -n 100 outputs/slurm-shd-r0-gzcmp-JOBID_TASK.err
```

汇总部分或全部结果：

```bash
python3 gate_zo_bdnn/summarize_shd_radlif_r0_gatezo_vs_no_gatezo.py
```

输出根目录：

```text
outputs/shd_radlif_r0_gatezo_vs_no_gatezo_5seeds/
```

更完整的参数、任务映射、checkpoint 恢复和字段解释位于包内：

```text
gate_zo_bdnn/SHD_R0_GATEZO_VS_NO_GATEZO_RUNBOOK_CN.md
```

## 9. 复现与科研边界

- 隐藏层和动力学参数不使用 BPTT；
- 随机候选生成后冻结；
- GateZO 只选择结构，不通过梯度修改候选权重；
- ridge 输出权重使用闭式求解；
- speaker-validation 用于增长接受和停止；
- test 不参与本实验的拟合、增长接受或停止；
- 应报告五个 seed 的均值、标准差和同 seed 配对差值；
- 当前包中的无 GateZO 对照是待运行实验，不能预先假定 GateZO 一定获胜。

## 10. 归档管理

建议始终把以下两个文件放在一起：

```text
shd_radlif_r0_gatezo_vs_no_gatezo_with_container.tar
shd_radlif_r0_gatezo_vs_no_gatezo_with_container.tar.sha256
```

复制到新位置后必须再次运行 `sha256sum -c`。不要修改 TAR 内部内容后继续沿用旧的 `.sha256` 文件。

