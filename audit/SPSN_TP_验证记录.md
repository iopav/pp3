# SPSN、TP 适配验证记录

验证日期：2026-10-03。使用本机 Python 3.10 环境、PyTorch 2.11.0+cu128、NumPy 2.2.6。正式默认 seed 已设置为 **57–61**；本机训练检查只使用 seed 57，没有启动五组完整实验。所有真实数据检查只取极小子集、一个 epoch，不评价学习效果。原算法文件没有在此次实现中改写。

## 已通过的检查

| 检查 | 结果 | 证据 |
|---|---|---|
| SPSN、TP × Spike、Texture，CPU | 四项均完成训练、评估、模型保存、CSV | `implementation_final_cpu/run_20261003_160357_733483/` |
| SPSN、TP × Spike、Texture，GPU | 四项均成功，严格计数没有未知算子 | `implementation_final_gpu/run_20261003_160130_560021/` |
| 保留 Spike 400 步、Texture 3775 步，CPU | 四项均成功；各 split 4 个真实样本，隐藏宽度 8，1 个隐藏层，batch 3，1 epoch | `implementation_full_time/run_20261003_160504_139453/` |
| 人工 SHD，两个模型，CPU/GPU | 每种设备的两个任务均成功，训练 8 个样本（含副本）、测试 4 个，抽取 16 个时间位置 | 上述 final_cpu/final_gpu 中的 `shd/` 任务 |
| 人工 SHD，workers=2 | SPSN、TP 两项均成功 | `implementation_shd/run_20261003_155744_744636/` |
| 故意缺失 SHD，之后运行 Spike | SHD 失败，Spike 继续成功，主控退出码 1 | `implementation_failure_continue/run_20261003_160009_156471/tasks.json` |
| 五个 seed 默认值 | dry-run 确认 Spike 两方法产生 10 个任务，seed 为 57–61 | `tests/test_pp3.py` |
| 数据、手算 FLOPs、未知操作、失败 CSV | 14 个 unittest 通过 | `tests/test_pp3.py`、`tests/test_worker.py` |
| 能耗缺失和功率积分 | 2 个 unittest 通过；功率回退使用模拟 NVML，未冒充真实硬件测量 | `tests/test_energy.py` |
| 开关计数不改变结果 | 两方法在 CPU、CUDA 共 4 项检查通过，逐参数和 RNG 状态完全相同；同时检查末尾 batch 与保存/恢复后输出 | `tests/check_model.py` |
| 输出合计 | 准备/训练/评估之和、每条算子费用、分阶段合计、FFT 子项、模型文件大小、合并 CSV 全部核对通过 | `verify_spsn_tp_outputs.py` |

单元测试覆盖 SHD 的重复事件、空事件、时间边界、扰动重复读取一致，并逐项比较公共处理与 GateZO 原类的输出；目录里没有生成 NPY。人工 HDF5 位于 `fixtures/shd/`，只用于测试，不能当作真实 SHD 准确率证据。正式运行命令没有指向这些人工文件。

## 完整时间轴的计算量记录

以下均为小模型、4 个训练样本加 4 个测试样本的 **一个 epoch**。数字用于复核保存的明细，不代表默认规模网络的计算量，也不用于比较最终学习效果。FFT 估算已包含在对应总数中，不能再加一次。

| 方法 / 数据 | 训练 FLOPs | 评估 FLOPs | 训练中的 FFT 估算部分 |
|---|---:|---:|---:|
| SPSN / Spike | 16,670,316 | 6,939,052 | 10,980,552 |
| SPSN / Texture | 117,499,453 | 47,179,914 | 98,469,241 |
| TP / Spike | 13,996,024 | 3,136,644 | 0 |
| TP / Texture | 42,906,674 | 10,056,988 | 0 |

## 范围限制

真实 Spike、Texture 来自本机已有 Avalanche 数据目录；代码运行不依赖该仓库。未找到真实 SHD 原始文件，所以 **真实 SHD 尚未测试**，没有下载数据。GPU 实测只验证一张卡上的任务依次排队，尚未在多张卡或真实 Slurm 环境运行；提交脚本需要按集群资源设置修改。CPU/GPU 之间不承诺逐位相同，只验证同一设备上开关计数不改变结果。

真实 GPU 运行成功读到 NVML 累计能量。极短评估区间出现过零增量或较大跳变，已经注明硬件更新周期限制；这组极小测试不能用于能耗排名。功率采样回退仅验证代码逻辑，未在不支持累计计量的真实 GPU 上测试。没有做断点续训，也没有进行正式全量训练或收敛验证。

早期失败日志保留在 `implementation_smoke/` 等目录：其中包括本机缺少 h5py，以及严格计数最初报出的 profiler/scatter 元数据覆盖问题。后来已修正并重新验证；这些历史结果不能和本记录列出的成功运行混用。FLOPs 实现的完整边界和规则见工程根目录 `SPSN_TP_FLOPs说明.md`。
