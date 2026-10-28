# 独立 Texture 实验

五个入口：rnn.py、rsnn.py、snnsurrogate.py、ann.py、gatezo.py。总入口 run_all.py 默认以阻塞方式依次运行五个独立进程，同一时间只有一个实验，失败立即停止。五种方法统一默认 seed=57。默认数据集为 texture-mini，本地目录 D:/workspace/Avalanche/avalanche-master/dataset/texture-mini，超算目录 /home/yizhil/pp3/data/texture-mini，读取 texture_spikesT450_T77.npz。训练/测试输入形状为 (960,77,9)/(240,77,9)，为二值脉冲；12 类 one-hot 标签转换为整数标签，不改变原始 train/test 划分，不再次归一化或脉冲编码。输出保存在 outputs/<数据集>/full/<实验>/seed_<seed>/，已成功且配置一致的实验自动跳过，其余已有目录拒绝覆盖，可用 --output 指定新目录。旧结果不迁移、不覆盖；显式使用 --dataset texture 可读取旧 NPY 及 old6classes/new6classes 布局。

## 数据集适配

重复执行相同命令时，五个入口默认自动跳过配置一致且 status.json 标记 success 的已完成实验，输出 `[skip]` 后，总入口继续下一个方法。也检查 model.pt、metrics.csv 非空；GateZO 还要求 summary.json 和 final_metrics.json，完成轮数或隐藏神经元数须匹配。新目录正常训练，已有失败、不完整或配置不同的目录会明确报错并保留，需用新的 --output 重新运行；不自动覆盖、不从中间 epoch 恢复。无需额外开关，单独运行 ann.py 等入口同样生效。配置比较忽略 PyTorch 版本、GPU 名称和 GateZO 实现所在目录，其余已记录字段参与比较；不会对数据文件或模型文件做内容哈希，原路径下更换数据后应使用新的 --output。检查跳过前仍会加载数据及构建当前配置，--dry-run 仅打印命令，不检查完成状态。

五个入口共享 data.py，支持 --dataset texture-mini（默认）、shd、texture，以及 --data-root 覆盖目录。texture-mini 也允许 --data-root 直接指定 NPZ 文件。时间长度、输入维度、输出类别数、MSE 分母和 GateZO one-hot 宽度由数据适配器提供，config.json 记录 dataset、data_root、steps、inputs、classes。完整 Texture-mini 运行使用 77 步，冒烟运行使用前 16 步。

SHD 读取 shd_train.h5、shd_test.h5 中的 labels、spikes/times、spikes/units；需要 h5py。默认预处理参数为 --shd-dt 0.005 秒、--shd-max-time 1.4 秒、--shd-channel-group-size 1，对应 280 步、700 输入、20 类。这些是可调实验配置，随结果保存。时间范围为 [0,max_time)，重复事件合并为二值占用，通道分组后维度为 ceil(700/group_size)，不做数据增强。冒烟运行取前 16 个时间箱，不对完整时域重采样。GateZO 会物化全部输入数组，完整 SHD 运行需评估主机内存；真实 SHD 全量训练与超算执行尚未验证。

```bash
sbatch --partition=gpu texture_experiments/submit.sbatch
sbatch texture_experiments/submit.sbatch --dataset shd --data-root /home/yizhil/pp3/data/shd
python texture_experiments/run_all.py --cpu --dataset texture-mini --smoke
python texture_experiments/run_all.py --cpu --dataset shd --data-root /path/to/shd --smoke
```

Linux 默认按数据集选择 /home/yizhil/pp3/data/<数据集>，提交脚本不固定 Texture 路径。test_data.py 验证真实 Texture NPZ 的输入和标签保真、异常标签与形状拒绝、SHD 合成事件分箱，以及 20 类模型的梯度和指标。

## 启动

```powershell
python texture_experiments/run_all.py --gpu 0
python texture_experiments/run_all.py --gpu 1 --experiments rnn rsnn gatezo
python texture_experiments/run_all.py --gpu 0 --dry-run
python texture_experiments/run_all.py --gpu 0 --smoke
python texture_experiments/gatezo.py --device cuda:0 --max-hidden 49
```

支持 --data-root、--output、--epochs、--batch-size、--seed；--cpu 显式使用 CPU。GPU 默认继承 CUDA_VISIBLE_DEVICES 的第一个设备，否则使用 0；--gpu 接受物理编号或 UUID，不越过调度器设备分配。--max-hidden 是 GateZO 独立入口参数。总入口显式向每个子进程传递同一个 seed，并设置 PYTHONHASHSEED=seed 与 CUBLAS_WORKSPACE_CONFIG=:4096:8。

## Slurm

```bash
# 在仓库根目录提交；作业脚本会初始化 Conda 并激活 avalanche。
sbatch --partition=gpu texture_experiments/submit.sbatch
sbatch texture_experiments/submit.sbatch --experiments rsnn gatezo
```

环境初始化与根目录 submit_experiments.sbatch 一致：优先使用 CONDA_BASE，否则通过 CONDA_EXE 或 conda 查找安装目录，再激活 avalanche。若提交环境无法找到 Conda，先执行 `export CONDA_BASE=/实际的/conda安装目录`。作业启动时记录 Python 路径及 NumPy、PyTorch 版本；依赖导入失败时立即退出。

账户为 naiss2026-4-902-gpu，申请 1 张 GPU、4 个 CPU、32 GB 内存、24 小时；分区采用集群默认，可在 sbatch 命令中按实际要求指定。超算默认数据目录为 /home/yizhil/pp3/data/texture-mini，可用 --data-root 覆盖。无需传 --gpu，自动使用 Slurm 分配的设备。尚未提交正式集群任务。

## 模型与训练

四个梯度模型默认单隐藏层、49 个神经元、100 epoch、batch size 128、完整数据时间长度（当前 texture-mini 为 77 步）。输出层不计入隐藏层数。手写模型的默认值集中在 defaults.json：SGD 默认参数（lr=0.01；已同步用户在旧目录的配置修改）、交叉熵、seed=57、时间平均读出、框架默认初始化。RNN 为 tanh RNN；ANN 对时间步共享 Linear+ReLU 后平均；SNN 为 LIF，beta=0.9、阈值=1、减法 reset（reset 梯度 detach），fast-sigmoid surrogate slope=25；RSNN 增加隐藏脉冲密集递归连接。使用完整反向传播，不截断时间。所有入口设置 Python、NumPy、PyTorch 和全部 CUDA seed，关闭 cuDNN benchmark，启用 cuDNN deterministic 与严格 torch deterministic algorithms；遇到没有确定性实现的算子时直接失败，不静默降级。这些是新增模型的默认定义，不声称来自已有 GateZO 配置。

GateZO 调用相邻 texture_bdnn_gatezo_bundle 的 BDNN1GateZO 与 GATE_ZO_DEFAULTS，使用 validation_checkpoint 协议和统一 seed=57；原生超参数明确设为 nB=1、L_max=49，分别控制每轮候选新增神经元数和成功增长轮数上限。这两个值在 GateZO 适配器内部设置，无需命令行参数；--epochs 仅为 GateZO 兼容参数，四个梯度模型仍默认训练 100 epoch。目标宽度为 49，每轮成功增加一个神经元，共 49 轮；候选搜索失败沿用原生停止行为，未达到目标则实验失败，不无限重试。其余搜索及门控超参保持默认，候选池 T_max=20；nB=1 配合 min_keep=1 时无法删去该轮唯一候选。适配器仍跳过误差及平台期提前停止，并仅保存目标宽度 checkpoint。对所选数据集全部类别单阶段训练，不运行原版两阶段增量协议，因而完整实验流程不与 spec 完全等同，详见 GATEZO_PARITY_AUDIT_CN.md。

运行不依赖 pp3 controller、配置、拆分和输出。energy.py 是本目录独立工具。GateZO Recorder 兼容原实现已有的 _pp3_runtime 回调属性，但不导入 pp3 runtime，counter=None。没有修改原 bundle。

## 输出与指标

四个梯度基线默认关闭每轮训练集和测试集报告评估，ANN 连续训练 10 个 epoch 后刷新 metrics.csv，其他三个梯度模型每 5 个 epoch 刷新；末尾不足窗口长度时按实际轮数测量。仅在窗口起止读取 nvmlDeviceGetTotalEnergyConsumption()，以差值除以 1000 得到窗口焦耳数，再按各 epoch 的训练耗时占比分摊 energy_j。窗口内没有报告评估、CSV 写盘或日志输出，但包含正常训练循环的主机调度开销；分摊值是逐轮估计，各轮之和等于窗口差值。计数器不支持/读取失败时留空，不回退到功率积分；计数器未增长时保留零差值，固定窗口仍不能保证任意硬件上都足够长。

添加 `--eval-each-epoch` 可恢复四个梯度模型每轮的训练集与测试集评估，并自动使用单 epoch 训练能耗窗口，评估在窗口关闭后执行。默认输出不含 mse、test_mse、train_acc、test_acc 四列；训练交叉熵照常保留。GateZO 不采用此开关，默认关闭每个增长轮次的测试集报告以及 Recorder 的重复全量训练/测试评估。其连续 5 个增长尝试（包含失败尝试）构成能耗窗口，按各轮耗时分摊；末尾不足 5 轮单独结算。结构选择所需的训练子集指标和验证计算保留并计入训练能耗，最终重拟合另开窗口，最终训练/测试评估在该窗口关闭后各执行一次。

config.json 保存配置，model.pt 保存最终权重，status.json 记录成功或失败。GateZO 每 5 个增长尝试刷新 metrics.csv，末尾余数在最终重拟合前刷新；最终目标宽度结构恢复与读出重拟合另存 final_metrics.json、summary.json，不混入 epoch 行。final_metrics 的能耗只包含最终重拟合，累计值包含增长和重拟合，准确率来自窗口关闭后的最终评估。

`metrics.csv` 仅保留轮次、模型大小、内存、损失、准确率、能耗、功率与耗时等数值指标。`energy_method`、`energy_reason`、`gpu_uuid`、`auxiliary_mem_method`、`auxiliary_mem_reason` 以及 GateZO 的 `unit`、`source_unit` 不写入结果，也不另存元数据文件。GateZO 的 validation_accuracy 由结构搜索回调填充，属于有效数值指标，予以保留；最终处理仍单独写入 final_metrics.json。已有结果不自动改写，已完成实验仍遵循原有跳过规则。

| 字段 | 口径 |
| --- | --- |
| para_mem_bytes | 最终保留模型参数按各自实际 dtype 计算的字节总和，包括输出层；GateZO 为实际 W、U、Beta，不含 GPU 临时副本、梯度、优化器状态与序列化开销 |
| optimizer_state_bytes | 每轮最后一次 optimizer.step 后的优化器 state 存储；Tensor/NumPy 按实际底层存储去重，CPU/GPU 都计入，数值标量按 8 字节约定计数；不含 param_groups |
| auxiliary_mem_bytes | optimizer_state_bytes + hiddenstatememfor bp；SGD momentum=0.9。不是运行时显存峰值。GateZO 无优化器、BP 或独立可训练门控状态，故为 0；CPU 运行也输出数值 |
| hiddenstatememfor bp | 单位 bytes；autograd 为 BP 保存的非参数张量底层存储的存活峰值，每轮取 batch 最大值。按存储去重并排除模型参数，包含被保存的输入、标签及中间张量。CUDA 训练统计 CUDA 存储，CPU 训练统计 CPU 存储；GateZO 不做 BP，增长和最终记录均为 0 |
| mse / test_mse | 当前模型全训练/测试集 softmax 概率对 one-hot 的 MSE，除以 N×类别数；梯度基线仅启用 --eval-each-epoch 时输出 |
| train_acc / test_acc | 当前模型重新评估的全训练/测试准确率，0..1；梯度基线仅启用 --eval-each-epoch 时输出 |
| train_cross_entropy | 仅梯度模型：训练期间按样本加权的交叉熵 |
| energy_j / power_w | 累计计数器窗口差值按训练耗时分摊与 energy/seconds；GateZO 的窗口为 5 个增长尝试，包含结构选择所需的验证计算 |
| total_energy_j / total_power_w | 累计能量及累计能量/累计测量时长，不对功率求和 |

2026-10-28 按指定口径将 auxiliary 改为“轮末优化器状态 + 本轮 BP 保存张量峰值”。它是两个定义明确的分项之和，不是同时刻存储峰值，也不是最终 checkpoint 文件大小。排除随机数、采样划分、配置、训练进度、梯度和搜索临时空间；因此不能据 GateZO 的 aux=0 推断其运行内存为零。hiddenstatememfor bp 使用 saved_tensors_hooks，按底层存储去重并排除模型参数，包含 autograd 保存的输入/标签，不是整个计算图 CPU 元数据大小；该值已包含在 auxiliary 中，不再重复相加。能耗口径不变。

2026-10-28 起配置增加 bp_memory_version 和 bp_memory_aggregation，以拒绝将旧形状估算结果复用为新 BP 指标。hooks 只持有 detach 后的共享存储别名，不复制或卸载张量，随 autograd 释放保存对象更新计数；统计会增加 Python 调用开销，现有计时与能耗包括该开销。使用新输出目录重新运行，历史 CSV 不改写。`test_bp_memory.py` 检查存储去重、参数排除、retain_graph/释放生命周期，以及四模型 CPU/CUDA 输出、梯度和更新一致性。

### 保存与复用

四个梯度入口默认 SGD momentum=0.9，最终 model.pt 保存 model、optimizer、completed_epochs 和 config，动量缓冲实际落盘。hidden 字段只保存测量值，不序列化计算图。未增加 --resume 或保存 RNG/采样器现场，不能宣称完整精确续跑功能已实现。配置新增 auxiliary_memory_version，旧结果与新口径不兼容，须使用新的 --output；不覆盖历史结果。

GateZO 的选中结构通过裁剪后的 W/U/Beta 保存，没有需跨轮更新的独立 gate logits 或优化器动量；门控阈值属于 config。最终 checkpoint 另存 growth_round 和 hidden_neurons 作为进度元数据，不计入指定的 aux。候选池、屏蔽评分和日志历史不是该口径下的持久学习状态；目标宽度适配器的临时 checkpoint 副本也不计作最终保留的额外模型。原 AuxiliaryMemory 工具仅留作旧测试，训练入口已不再用它计算指标。

config.json 记录 energy_measurement_version、eval_each_epoch 和 energy_window_epochs（GateZO 单位为增长尝试），防止误用不同测量口径的旧结果。重跑请指定新的 --output 目录，旧 CSV 无法据此恢复真实逐轮能耗。GPU 计数器与功率接口说明见 [NVML Device Queries](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)。

## 验证

`python -m unittest discover -s texture_experiments -p 'test_energy_blocks.py'` 检查 ANN 十轮窗口、RNN 五轮窗口、末尾余数、能量守恒和评估开关。`test_gatezo_energy_blocks.py` 检查增长窗口及最终测试边界。仓库根目录执行 `python -m unittest discover -s tests -p 'test_energy*.py'` 检查累计计数器和旧功率积分工具各自的读取与异常处理。

python texture_experiments/test_baselines.py 检查梯度、状态重置、批次加权指标和辅助内存口径。python texture_experiments/test_gatezo.py 检查原生 nB/L_max、轮数上限、搜索失败停止、实际神经元上限、目标宽度与运行时隔离。--smoke 对四个梯度模型取最多 8 个样本、前 16 步；GateZO 每类最多取 4 个样本、前 16 步，仍使用 nB=1、L_max=49，成功运行需达到 49 个隐藏神经元。结果单独存于 outputs/<数据集>/smoke。冒烟结果不能代表完整训练性能。依赖 PyTorch、NumPy，以及原 bundle 所需 SciPy、scikit-learn、matplotlib；NVML 读取需要 nvidia-ml-py。


## 全部五种方法的训练能耗运行

在项目根目录提交以下命令，顺序执行 RNN、RSNN、SNN surrogate、ANN 和 GateZO。不要添加 `--eval-each-epoch`。ANN 每 10 epoch 测量，其他三个梯度模型每 5 epoch 测量，GateZO 每 5 个增长尝试测量，均按时间比例分摊。GateZO 改动仅位于 texture_experiments 的适配器，原 texture_bdnn_gatezo_bundle 文件不修改。

```bash
sbatch --partition=gpu texture_experiments/submit.sbatch \
  --experiments rnn rsnn snnsurrogate ann gatezo \
  --output texture_experiments/outputs_energy_nb1_lmax49
```

默认数据集是 texture-mini，完整 Texture 添加 `--dataset texture`。输出目录与旧测量结果区分；GateZO 新配置使用 `nvml_total_growth_block_time_allocation_v1`，不会把旧逐轮功率积分结果作为匹配结果复用。

`test_gatezo_energy_blocks.py` 使用实际适配后的增长函数验证五轮窗口、余数、缺测、验证保留和最终测试仅调用一次；候选搜索与能耗遥测在该测试中模拟，不构成真实 GPU 能耗精度验证。
