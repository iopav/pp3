# 独立 Texture 实验

五个入口：rnn.py、rsnn.py、snnsurrogate.py、ann.py、gatezo.py。总入口 run_all.py 默认以阻塞方式依次运行五个独立进程，同一时间只有一个实验，失败立即停止。五种方法统一默认 seed=57。默认数据目录 D:/workspace/Avalanche/avalanche-master/dataset/texture；读取原始 *_texture.npy，不使用 *_texture_img.npy。输出保存在本目录 outputs/full/<实验>/seed_<seed>/，重复目录拒绝覆盖，可用 --output 指定新目录。兼容 old6classes/new6classes 数据布局，合并为 12 类联合训练。

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
# 在仓库根目录提交；先激活包含依赖的 Python 环境。
sbatch texture_experiments/submit.sbatch
sbatch texture_experiments/submit.sbatch --experiments rsnn gatezo
```

账户为 naiss2026-4-902-gpu，申请 1 张 GPU、4 个 CPU、32 GB 内存、24 小时；分区采用集群默认，可在 sbatch 命令中按实际要求指定。服务器需显式提供服务器数据路径，本地 Windows 路径不能用于服务器。无需传 --gpu，自动使用 Slurm 分配的设备。尚未提交正式集群任务。

## 模型与训练

四个梯度模型默认单隐藏层、49 个神经元、100 epoch、batch size 128、完整 3775 步。输出层不计入隐藏层数。手写模型的默认值集中在 defaults.json：SGD 默认参数（lr=0.01；已同步用户在旧目录的配置修改）、交叉熵、seed=57、时间平均读出、框架默认初始化。RNN 为 tanh RNN；ANN 对时间步共享 Linear+ReLU 后平均；SNN 为 LIF，beta=0.9、阈值=1、减法 reset（reset 梯度 detach），fast-sigmoid surrogate slope=25；RSNN 增加隐藏脉冲密集递归连接。使用完整反向传播，不截断时间。所有入口设置 Python、NumPy、PyTorch 和全部 CUDA seed，关闭 cuDNN benchmark，启用 cuDNN deterministic 与严格 torch deterministic algorithms；遇到没有确定性实现的算子时直接失败，不静默降级。这些是新增模型的默认定义，不声称来自已有 GateZO 配置。

GateZO 调用相邻 texture_bdnn_gatezo_bundle 的 BDNN1GateZO 与 GATE_ZO_DEFAULTS，使用 validation_checkpoint 协议和统一 seed=57，其余搜索及门控超参保持默认；对完整 12 类训练，不运行两阶段增量协议。最大且目标保留隐藏神经元为 49，每个增长 loop 记录一行。当前适配器按 W 的列数强制增长到目标宽度，--epochs 仅记录为兼容参数，不构成 GateZO 增长上限；剩余容量不足时缩小新增块。候选池仍为默认 100，不属于保留模型宽度。

运行不依赖 pp3 controller、配置、拆分和输出。energy.py 是本目录独立工具。GateZO Recorder 兼容原实现已有的 _pp3_runtime 回调属性，但不导入 pp3 runtime，counter=None。没有修改原 bundle。

## 输出与指标

每轮刷新 metrics.csv；config.json 保存配置，model.pt 保存最终权重，status.json 记录成功或失败。GateZO 的最终最佳结构恢复与读出重拟合另存 final_metrics.json、summary.json，不混入 epoch 行；final_metrics 中累计值包含最终处理。

| 字段 | 口径 |
| --- | --- |
| para_mem_bytes | 最终保留模型参数按各自实际 dtype 计算的字节总和，包括输出层；GateZO 为实际 W、U、Beta，不含 GPU 临时副本、梯度、优化器状态与序列化开销 |
| auxiliary_mem_bytes | 优化/搜索窗口内 torch.cuda.max_memory_allocated 减去常驻 CUDA 参数字节；包含 hidden activation、递归状态、梯度、输入批次和临时 Torch 张量。CPU 运行留空 |
| hidden_state_mem_bytes | 主隐藏激活 B×T×H×dtype 与持续递归状态之和，作为 hidden working memory 下界；ANN 的递归状态为 0，但隐藏激活非零。该值已包含在 auxiliary_mem_bytes 中，不能再次相加 |
| mse / test_mse | 当前模型全训练/测试集 softmax 概率对 one-hot 的 MSE，除以 N×12 |
| train_acc / test_acc | 当前模型重新评估的全训练/测试准确率，0..1 |
| train_cross_entropy | 仅梯度模型：训练期间按样本加权的交叉熵 |
| energy_j / power_w | 当轮优化、结构搜索、门控及参数拟合的 GPU 能量与 energy/seconds；训练集和测试集的报告评估均排除在测量窗口之外 |
| total_energy_j / total_power_w | 累计能量及累计能量/累计测量时长，不对功率求和 |

auxiliary_mem_bytes 是 PyTorch 活跃分配峰值口径，不是 nvidia-smi 进程占用或 CUDA reserved memory；它不包含 NumPy/CPU 内存、CUDA context、allocator 保留但未被活跃张量占用的缓存和非 PyTorch 库分配。hidden_state_mem_bytes 是主隐藏激活加递归状态的下界，用于解释 auxiliary 的一部分；完整 BPTT/autograd 内存以 auxiliary 实测峰值为准。NVML 能量为整张 GPU 在测量窗口中的能量，可能包含其他进程，不含 CPU；不可用或计数器未更新时留空并保留原因，任一轮缺测则累计能量也留空。初始化、训练集报告评估、测试集评估、CSV 和最终 checkpoint 写盘均不计入；GateZO 的候选搜索、结构门控和参数拟合计入，随后用于报告的训练/验证/测试指标不计入。

## 验证

python texture_experiments/test_baselines.py 检查梯度、状态重置、批次加权指标和辅助内存口径。python texture_experiments/test_gatezo.py 检查实际神经元上限、目标宽度与运行时隔离。--smoke 对四个梯度模型取最多 8 个样本、前 16 步；GateZO 每类最多取 4 个样本、前 16 步，但仍增长到目标隐藏大小 49，因此其增长行数不固定。结果单独存于 outputs/smoke。冒烟结果不能代表完整训练性能。依赖 PyTorch、NumPy，以及原 bundle 所需 SciPy、scikit-learn、matplotlib；NVML 读取需要 nvidia-ml-py。
