# Texture BDNN / Gate-ZO 独立运行包

本包包含当前工程用于 texture 的两阶段代码、所有本地 Python 依赖源码、8 个预处理数据文件、对照实验/参数搜索脚本和原有 Gate-ZO 测试。原始算法源码保持不变；仅打包副本中的 SLURM 脚本将项目目录改为可配置路径，并统一 LF 换行。另附快捷入口 run_texture.py。

## Gate-ZO 功能

- `gatezo`：每轮用原 BDNN 不等式筛出候选神经元，逐个做硬屏蔽，按损失变化排序，再选择带神经元数量惩罚的子结构；也支持 `--zo-loss-tolerance` 的损失约束选择。
- `fixed_nozo`：保留整批候选神经元。
- `adaptive_nozo`：按原候选顺序选择满足损失容差的前缀，不做逐神经元屏蔽。

这里的 Gate-ZO 是结构选择。继承原模型所需的 `zo_refinement.py` 也已包含，但 Gate-ZO 类强制 `zo_steps=0`，不启用旧的连续权重 ZO 优化。

## 数据和两阶段关系

输入维度顺序为 `(samples, timestep, channel)`，所有 X 文件均为 **3775 timestep、9 channel**。第一阶段学习旧 6 类；第二阶段使用第一阶段的 W/U 学习新 6 类，标签加载时减去 6。这是现有两阶段流程，不是一个统一 12 类输出的分类器。

| 文件 | shape | dtype |
| --- | --- | --- |
| `data/new6classes/X_test_6cla_new.npy` | `(118, 3775, 9)` | `float64` |
| `data/new6classes/X_train_6cla_new.npy` | `(482, 3775, 9)` | `float64` |
| `data/new6classes/Y_test_6cla_new.npy` | `(118,)` | `int64` |
| `data/new6classes/Y_train_6cla_new.npy` | `(482,)` | `int64` |
| `data/old6classes/X_test_6cla_old.npy` | `(122, 3775, 9)` | `float64` |
| `data/old6classes/X_train_6cla_old.npy` | `(478, 3775, 9)` | `float64` |
| `data/old6classes/Y_test_6cla_old.npy` | `(122,)` | `int64` |
| `data/old6classes/Y_train_6cla_old.npy` | `(478,)` | `int64` |

数据形状和原始标签详见 `data_info.json`。本包使用现有预处理 `.npy` 数据，无需另外下载。数据来源引用保留在 `readme.md`。

## 安装和直接运行

在解压后的 `texture_bdnn_gatezo_bundle` 文件夹内打开终端：

```bash
python -m pip install -r requirements.txt
python run_texture.py
```

快捷入口默认：seed=22、gatezo、validation_checkpoint、候选池=100、每轮预选=3、至少保留=1、lambda=0.0001，设备自动选择 CUDA/CPU。命令执行完整 exp1 和 exp2；无需提供预训练权重。可以覆盖所有原入口参数：

```bash
python run_texture.py --device cuda --compute-batch-size 128
python run_texture.py --device cpu
python run_texture.py --selection-mode adaptive_nozo --seed 22
python run_texture.py --selection-mode fixed_nozo --seed 22
python run_texture.py --growth-step 5 --min-keep 2 --zo-sparsity-lambda 0.0003
python run_texture.py --help
```

CPU 可运行，但 3775 步的完整训练耗时较长。第三方库由 `requirements.txt` 列出；本机验证时的实际版本见 `environment-tested.json`。GPU 运行需要适配目标机器的 PyTorch/CUDA。`requirements-original.txt` 仅保留旧工程环境记录，不作为本包默认安装清单。运行现有测试：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q gate_zo_bdnn/tests/test_structural_gate.py
```

## 原始可配置入口与输出

```bash
python -m gate_zo_bdnn.run_texture_comparison_task --selection-mode gatezo --growth-protocol validation_checkpoint --seed 22 --data-root data --output-root outputs/texture_gatezo_protocols --device auto
```

输出目录：`outputs/texture_gatezo_protocols/<protocol>/<method>/seed_<seed>/`。`exp1/` 和 `exp2/` 分别保存 W/U/Beta、误差、准确率、耗时与详细 summary；外层 `summary.json` 汇总两个阶段。三种方法完成后：

```bash
python -m gate_zo_bdnn.summarize_texture_comparison --root outputs/texture_gatezo_protocols/validation_checkpoint
```

`validation_checkpoint` 使用训练集内部验证来选 checkpoint，然后用全部训练样本重拟合读出层，最终评估测试集。`legacy_test` 重现原有测试集选 checkpoint 的流程，输出会记录 `test_used_for_selection=true`。

原 Gate-ZO 分阶段入口和种子批量入口也已包含，见 `gate_zo_bdnn/README.md`；这些旧入口默认使用 `legacy_test`。原始无 Gate-ZO 的 `main_exp1.py`、`main_exp2.py` 和 `run_seed_sweep.py` 也已保留，需要先 exp1 再 exp2；原始入口在当前工作目录保存结果，与快捷入口的输出布局不同。

## SLURM / Apptainer 服务器脚本

所有 texture 对照和搜索 `.sbatch` 已包含。脚本仍沿用现有集群 account/partition/resource 设置，使用时按目标集群调整。项目目录默认取提交目录，也可用 `BDNN_PROJECT_DIR` 指定。

Apptainer 容器镜像不在本包内；现有服务器脚本需要已有镜像，通过 `BDNN_CONTAINER_IMAGE` 指定。普通 `python run_texture.py` 入口不依赖容器。在包根目录提交：

```bash
mkdir -p outputs
export BDNN_PROJECT_DIR="$PWD"
export BDNN_CONTAINER_IMAGE=/absolute/path/to/bdnn_arm64.sif
sbatch gate_zo_bdnn/compare_texture_methods_validation_gpu.sbatch
```

其他已包含的服务器实验：legacy 协议、预选 5 个、预选 5 个至少保留 2 个、lambda/min_keep 组合搜索、损失约束搜索。搜索汇总需要先运行对应的 adaptive_nozo 基线，默认目录详见各脚本参数。

## 包内容与验证

`MANIFEST.json` 记录所有打包文件的 SHA-256、大小及源码来自原工程的相对路径。`VERIFY_REPORT.json` 记录交付前检查结果。未包含历史训练输出、SHD/CIFAR 代码、Python 安装或第三方库二进制。完整 texture 训练没有重新执行；检查范围见验证报告。
