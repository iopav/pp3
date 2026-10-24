# 独立 Texture 实验

五个入口：rnn.py、rsnn.py、snnsurrogate.py、ann.py、gatezo.py。总入口 run_all.py 默认以阻塞方式依次运行五个独立进程，同一时间只有一个实验，失败立即停止。五种方法统一默认 seed=57。默认数据集为 texture-mini，本地目录 D:/workspace/Avalanche/avalanche-master/dataset/texture-mini，超算目录 /home/yizhil/pp3/data/texture-mini，读取 texture_spikesT450_T77.npz。训练/测试输入形状为 (960,77,9)/(240,77,9)，为二值脉冲；12 类 one-hot 标签转换为整数标签，不改变原始 train/test 划分，不再次归一化或脉冲编码。输出保存在 outputs/<数据集>/full/<实验>/seed_<seed>/，已成功且配置一致的实验自动跳过，其余已有目录拒绝覆盖，可用 --output 指定新目录。旧结果不迁移、不覆盖；显式使用 --dataset texture 可读取旧 NPY 及 old6classes/new6classes 布局。

## 数据集适配

重复执行相同命令时，五个入口默认自动跳过配置一致且 status.json 标记 success 的已完成实验，输出 `[skip]` 后，总入口继续下一个方法。也检查 model.pt、metrics.csv 非空；GateZO 还要求 summary.json 和 final_metrics.json，完成轮数或隐藏神经元数须匹配。新目录正常训练，已有失败、不完整或配置不同的目录会明确报错并保留，需用新的 --output 重新运行；不自动覆盖、不从中间 epoch 恢复。无需额外开关，单独运行 ann.py 等入口同样生效。配置比较忽略 PyTorch 版本、GPU 名称和 GateZO 实现所在目录，其余已记录字段参与比较；不会对数据文件或模型文件做内容哈希，原路径下更换数据后应使用新的 --output。检查跳过前仍会加载数据及构建当前配置，--dry-run 仅打印命令，不检查完成状态。

五个入口共享 data.py，支持 --dataset texture-mini（默认）、shd、texture，以及 --data-root 覆盖目录。texture-mini 也允许 --data-root 直接指定 NPZ 文件。时间长度、输入维度、输出类别数、MSE 分母和 GateZO one-hot 宽度由数据适配器提供，config.json 记录 dataset、data_root、steps、inputs、classes。完整 Texture-mini 运行使用 77 步，冒烟运行使用前 16 步。

SHD 读取 shd_train.h5、shd_test.h5 中的 labels、spikes/times、spikes/units；需要 h5py。默认预处理参数为 --shd-dt 0.005 秒、--shd-max-time 1.4 秒、--shd-channel-group-size 1，对应 280 步、700 输入、20 类。这些是可调实验配置，随结果保存。时间范围为 [0,max_time)，重复事件合并为二值占用，通道分组后维度为 ceil(700/group_size)，不做数据增强。冒烟运行取前 16 个时间箱，不对完整时域重采样。GateZO 会物化全部输入数组，完整 SHD 运行需评估主机内存；真实 SHD 全量训练与超算执行尚未验证。

```bash
sbatch texture_experiments/submit.sbatch
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
sbatch texture_experiments/submit.sbatch
sbatch texture_experiments/submit.sbatch --experiments rsnn gatezo
```

环境初始化与根目录 submit_experiments.sbatch 一致：优先使用 CONDA_BASE，否则通过 CONDA_EXE 或 conda 查找安装目录，再激活 avalanche。若提交环境无法找到 Conda，先执行 `export CONDA_BASE=/实际的/conda安装目录`。作业启动时记录 Python 路径及 NumPy、PyTorch 版本；依赖导入失败时立即退出。

账户为 naiss2026-4-902-gpu，申请 1 张 GPU、4 个 CPU、32 GB 内存、24 小时；分区采用集群默认，可在 sbatch 命令中按实际要求指定。超算默认数据目录为 /home/yizhil/pp3/data/texture-mini，可用 --data-root 覆盖。无需传 --gpu，自动使用 Slurm 分配的设备。尚未提交正式集群任务。

## 模型与训练

四个梯度模型默认单隐藏层、49 个神经元、100 epoch、batch size 128、完整数据时间长度（当前 texture-mini 为 77 步）。输出层不计入隐藏层数。手写模型的默认值集中在 defaults.json：SGD 默认参数（lr=0.01；已同步用户在旧目录的配置修改）、交叉熵、seed=57、时间平均读出、框架默认初始化。RNN 为 tanh RNN；ANN 对时间步共享 Linear+ReLU 后平均；SNN 为 LIF，beta=0.9、阈值=1、减法 reset（reset 梯度 detach），fast-sigmoid surrogate slope=25；RSNN 增加隐藏脉冲密集递归连接。使用完整反向传播，不截断时间。所有入口设置 Python、NumPy、PyTorch 和全部 CUDA seed，关闭 cuDNN benchmark，启用 cuDNN deterministic 与严格 torch deterministic algorithms；遇到没有确定性实现的算子时直接失败，不静默降级。这些是新增模型的默认定义，不声称来自已有 GateZO 配置。

GateZO 调用相邻 texture_bdnn_gatezo_bundle 的 BDNN1GateZO 与 GATE_ZO_DEFAULTS，使用 validation_checkpoint 协议和统一 seed=57，其余搜索及门控超参保持默认；对所选数据集全部类别训练（Texture 为 12 类、SHD 为 20 类），不运行两阶段增量协议。最大且目标保留隐藏神经元为 49，每个增长 loop 记录一行。当前适配器按 W 的列数强制增长到目标宽度，--epochs 仅记录为兼容参数，不构成 GateZO 增长上限；剩余容量不足时缩小新增块。候选池仍为默认 100，不属于保留模型宽度。

运行不依赖 pp3 controller、配置、拆分和输出。energy.py 是本目录独立工具。GateZO Recorder 兼容原实现已有的 _pp3_runtime 回调属性，但不导入 pp3 runtime，counter=None。没有修改原 bundle。

## 输出与指标

四个梯度基线默认关闭每轮训练集和测试集报告评估，连续训练 5 个 epoch 后刷新 metrics.csv；末尾不足 5 轮按实际轮数测量。仅在窗口起止读取 nvmlDeviceGetTotalEnergyConsumption()，以差值除以 1000 得到窗口焦耳数，再按各 epoch 的训练耗时占比分摊 energy_j。窗口内没有报告评估、CSV 写盘或日志输出，但包含正常训练循环的主机调度开销；分摊值是逐轮估计，各轮之和等于窗口差值。计数器不支持/读取失败时留空，不回退到功率积分；计数器未增长时保留零差值，固定 5 轮仍不能保证任意硬件上都足够长。

添加 `--eval-each-epoch` 可恢复原先每轮的训练集与测试集评估，并自动使用单 epoch 训练能耗窗口，评估在窗口关闭后执行。例如 `python texture_experiments/run_all.py --experiments ann rnn rsnn snnsurrogate --eval-each-epoch --output /new/output`。默认输出不含 mse、test_mse、train_acc、test_acc 四列；训练交叉熵照常保留。GateZO 不采用此开关，保留原有搜索和评估流程及功率积分计量。

config.json 保存配置，model.pt 保存最终权重，status.json 记录成功或失败。GateZO 仍每轮刷新 metrics.csv，最终最佳结构恢复与读出重拟合另存 final_metrics.json、summary.json，不混入 epoch 行；final_metrics 中累计值包含最终处理。

`metrics.csv` 仅保留轮次、模型大小、内存、损失、准确率、能耗、功率与耗时等数值指标。`energy_method`、`energy_reason`、`gpu_uuid`、`auxiliary_mem_method`、`auxiliary_mem_reason` 以及 GateZO 的 `unit`、`source_unit` 不写入结果，也不另存元数据文件。GateZO 的 validation_accuracy 由结构搜索回调填充，属于有效数值指标，予以保留；最终处理仍单独写入 final_metrics.json。已有结果不自动改写，已完成实验仍遵循原有跳过规则。

| 字段 | 口径 |
| --- | --- |
| para_mem_bytes | 最终保留模型参数按各自实际 dtype 计算的字节总和，包括输出层；GateZO 为实际 W、U、Beta，不含 GPU 临时副本、梯度、优化器状态与序列化开销 |
| auxiliary_mem_bytes | 优化/搜索窗口内 torch.cuda.max_memory_allocated 减去常驻 CUDA 参数字节；包含 hidden activation、递归状态、梯度、输入批次和临时 Torch 张量。CPU 运行留空 |
| hidden_state_mem_bytes (included) | 主隐藏激活 B×T×H×dtype 与持续递归状态之和，作为 hidden working memory 下界；ANN 的递归状态为 0，但隐藏激活非零。该值已包含在 auxiliary_mem_bytes 中，不能再次相加 |
| mse / test_mse | 当前模型全训练/测试集 softmax 概率对 one-hot 的 MSE，除以 N×类别数；梯度基线仅启用 --eval-each-epoch 时输出 |
| train_acc / test_acc | 当前模型重新评估的全训练/测试准确率，0..1；梯度基线仅启用 --eval-each-epoch 时输出 |
| train_cross_entropy | 仅梯度模型：训练期间按样本加权的交叉熵 |
| energy_j / power_w | 梯度基线：累计计数器窗口差值按训练耗时分摊与 energy/seconds；GateZO：搜索/门控/拟合窗口的功率积分估计与 energy/seconds |
| total_energy_j / total_power_w | 累计能量及累计能量/累计测量时长，不对功率求和 |

auxiliary_mem_bytes 是 PyTorch 活跃分配峰值口径，不是 nvidia-smi 进程占用或 CUDA reserved memory；它不包含 NumPy/CPU 内存、CUDA context、allocator 保留但未被活跃张量占用的缓存和非 PyTorch 库分配。hidden_state_mem_bytes 是主隐藏激活加递归状态的下界，用于解释 auxiliary 的一部分；完整 BPTT/autograd 内存以 auxiliary 实测峰值为准。能耗为整张 GPU 的功率积分估计，包含空闲功耗及可能的其他进程，不含 CPU。CPU 运行或读取失败时留空并保留原因，任一轮缺测则累计能量也留空。初始化、训练集报告评估、测试集评估、CSV 和最终 checkpoint 写盘不在积分窗口内；GateZO 的候选搜索、结构门控和参数拟合计入。

config.json 记录 energy_measurement_version、eval_each_epoch 和 energy_window_epochs（后两项仅梯度基线），防止误用不同测量口径的旧结果。重跑请指定新的 --output 目录，旧 CSV 无法据此恢复真实逐轮能耗。GPU 计数器与功率接口说明见 [NVML Device Queries](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)。

## 验证

`python -m unittest discover -s texture_experiments -p 'test_energy_blocks.py'` 检查五轮窗口、末尾余数、能量守恒和评估开关。GateZO 的功率积分仍优先读取 NVML_FI_DEV_POWER_INSTANT，不支持时使用 nvmlDeviceGetPowerUsage；诊断原因和来源不写入结果。仓库根目录执行 `python -m unittest discover -s tests -p 'test_energy*.py'` 检查累计计数器和功率积分各自的读取与异常处理。

python texture_experiments/test_baselines.py 检查梯度、状态重置、批次加权指标和辅助内存口径。python texture_experiments/test_gatezo.py 检查实际神经元上限、目标宽度与运行时隔离。--smoke 对四个梯度模型取最多 8 个样本、前 16 步；GateZO 每类最多取 4 个样本、前 16 步，但仍增长到目标隐藏大小 49，因此其增长行数不固定。结果单独存于 outputs/<数据集>/smoke。冒烟结果不能代表完整训练性能。依赖 PyTorch、NumPy，以及原 bundle 所需 SciPy、scikit-learn、matplotlib；NVML 读取需要 nvidia-ml-py。
