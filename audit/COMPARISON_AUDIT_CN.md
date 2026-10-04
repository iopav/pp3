# PP3 三数据集、七方法适配审计

日期：2026-10-01。范围：阅读本地源码、方法说明及极小离线探针；没有下载 SHD，没有启动正式训练，没有修改七个方法的算法源码。这里给出适配与计量方案，不代表七方法已完成适配，也不代表断点恢复已经验证。

## 1. 数据与比较协议

读取的两份说明是根目录 `GATEZO_BDNN_RSNN_PROJECT_OVERVIEW_CN.md` 与 `README_CN.md`，并对照实际调用链。数据尺寸由当前 `.npy` 实读确认，见 `results/inventory_spike.json`。

| 数据 | 训练 / 测试样本 | 原生 X 形状（不含 N） | 标签 | 注意 |
|---|---:|---|---|---|
| Spike | 3219 / 781 | 1600×64，另有 400×64 | 0–19 | 两个时间版本须明确选择；不可各方法自行选不同版本 |
| Texture | 960 / 240 | 3775×9 | 0–11 | float64 连续有符号值，加载转 float32；不是二值脉冲 |
| SHD | 本次没有读数据文件 | 700 通道事件，分箱后 T×700 | 本地代码配置为 20 类 | 本次不下载；正式实验需已有本地数据及统一分箱 |

Spike 的现成图像版为 `(N,3,160,160)` uint8；Texture 为 `(N,3,185,185)` float32，包含负值。不能对 Texture 直接做 uint8/PIL 截断。图像版与原生时序版的编码差异必须记录，不能把不同表示的结果解释成训练规则的纯粹优劣。

方案暂按普通全类别分类讨论：20 / 12 / 20 输出类别。用户尚未确认是否采用旧类→新类流程。现有 Texture bundle 明确是 6+6 两阶段、各自六类输出，不能将两阶段准确率当成统一十二类准确率。原 BDNN 的 `main_exp1` 可作为全类别适配起点，不必调用依赖旧 W/U 的 `main_exp2`。

所有方法共用训练/验证/测试索引；归一化只在训练子集拟合；SHD 保留 speaker-disjoint validation；超参和停止用验证集。报告 validation accuracy per epoch/round；如果另存 test 曲线，冻结所有配置后生成，不能用其最大值选模型。统一记录时间分箱、通道分组、幅度缩放、截断与随机增强。16 步探针设置不能直接成为正式实验设置。

## 2. GateZO 的含义与两个实现分支

SHD R0：冻结随机 RadLIF 参数及递归结构；旧节点不接受新节点反馈，因此提交新节点不改变旧节点特征。候选池经过残差相关性/多样性预筛，再逐坐标硬消融，按校准损失变化排序，比较带节点数惩罚的前缀；外层 speaker-validation 决定接受和停止；最终完整训练集重拟合 ridge。这里的 ZO 优化离散结构，没有更新隐藏权重的反向传播。无 ZO fixed16 仍有公共预筛和外层增长门，适合做 GateZO 增量贡献消融，不能替代独立原 BDNN 基线。

Texture bundle：原 BDNN 不等式预筛 + 结构 GateZO，连续权重 ZO 被强制关闭。它与 SHD R0 的候选生成、动力学、时间读出不同。若论文要比较同一种 GateZO-RadLIF 在三数据集上的结果，应把 SHD R0 泛化；若沿用各自版本，应分别命名并说明结构差异，不能写成完全相同模型只改参数。

本地 `spike_bdnn_gatezo_bundle` 的 `data_info.json`、README 和入口仍指向 Texture 6+6；其目录下也没有 `data/`。这只能证明本地这份包尚未核实为 Spike 版本，不否定服务器存在另一份可用实现。

## 3. 入口、修改范围和阻塞

### BDNN

入口：`BDNNversion1-main/main_exp1.py` → `inputsdata.load_data` → `bdnn1.classification` → `sc_Search/addNodes/upgradeSCN`。第二阶段为 `main_exp2.py` → `bdnn2.py`。

输入原生 `(B,T,C)`，`sc_Search` 按 `X.shape[2]` 推断输入维数，类别维来自 one-hot。Spike/Texture 不需替换输入层，主要换数据路径、classes=20/12、original_classes=0，并调整阈值、输入尺度、alpha/beta、候选池和停止条件。SHD 需先构造事件分箱加载器。`T_max` 是候选个数，不是序列长度。

但是原代码不能只改超参后正式使用：

1. `main_exp1.py:15` 起路径元组为空，直接运行会解包失败。
2. `bdnn1.py:209` 的 `nC=len(C)` 数到类别维，应该按候选列数计数；探针只让一个候选合格、要求三个，实际仍返回成功，已复现。
3. `bdnn1.py:224` 的 `C.argsort(axis=1)[::-1]` 翻转行序，不是列内降序；应先明确跨类别贡献汇总，再排序候选列。候选 U 子矩阵还需保留原候选索引，不能把筛选后位置直接当原始神经元编号。
4. `bdnn1.py:472` 起按 test 最大准确率选结构，有测试集泄漏；用训练内部 validation 替代。
5. 随后的 `0:index_of_max_value` 用零基日志下标直接切节点，可能切成零节点；应保存最佳轮的完整 W/U/Beta 快照，不能按重复扩展的日志下标猜结构。
6. 大量 NumPy/pinv/CPU 张量，不会因为 main 里定义了 CUDA device 就全部运行在 GPU。不能以近零 GPU 能耗宣称 BDNN 整体低能耗。

### Integer-RSNN

入口：`Integer-Arithmetic-SNN-Learning-main/experiments/shd/rec_snn_experiment.py` → `models/rec_snn/trainer_shd.biograd_snn_training` → `BioGradNetwork.train_online` → `train_update_parameter` → `update_inference_weights`。

当前 SHD 配置：700 通道按四通道求和成 175，10 个时间帧，20 类；这不同于 SHD GateZO 的时间分箱。增加 `.npy` loader，向训练器注入 DataLoader，保留 batch `(B,T,C)`，调用核心前 `permute(0,2,1)`。输入层可直接使用 C=64/9，输出 K=20/12，无需换 Linear 结构；同时参数化 norm、soft_error_start 和序列步数。默认 `lr_rec=0`，递归权重不按常规非零学习率学习；要保留还是改变必须在实验配置说明。

Texture 还需要决定输入表示：直接实数电流能跑通接口，但不能称全整数输入运算；若研究整数方法，使用仅训练集拟合的有符号量化尺度并报告截断范围，或所有方法统一采用正负双通道编码（此时 C=18）。不能直接裁掉负值。

`BioGradNetwork` 不是 `nn.Module`，没有可直接覆盖全部状态的标准 `state_dict()`；训练器中的模型 pickle 保存语句被注释。需要专门导出各 cell 的训练/推理/反馈/递归权重、参数字典、量化尺度、学习率、epoch 与 RNG。本次 integer mode 探针中权重 dtype 为 float32；数值量化位宽不等于实际存储位宽或整数硬件指令。

### SLTT

入口：`SLTT-main/train_SLTT.py`；另有 `train_SLTTK.py` 和 BPTT 对照。当前 ResNet/VGG 面向二维图像/事件帧，不能仅改第一层通道数后直接读 `(B,T,C)`。

建议增加时序全连接骨干：每步 `Linear(C,H) → 原 SLTTNeuron → … → Linear(H,K)`，保留其膜电位 detach、逐时间步 loss/backward、一个样本批次后的 optimizer.step 和 batch 间状态重置。`train_SLTT.py:307` 的数据分支要显式接入时序数据并传 `x[:,t,:]`，避免落入静态图像分支反复喂整段序列。此方案是 SLTT 的时序骨干移植，应明确与原图像网络不同。

### SNN-NDOT

入口：静态图像 `SNN-NDOT-main/train_cifar_main.py` → `functions.py`；时序切片参考 `train_cifar10dvs.py` → `functions_cifar10dvs.py`。NDOT_A 与 `--online_update` 的 NDOT_O 更新时机不同，需要固定变体或分别报告。

同 SLTT，需要时序 Linear 骨干及 NPY loader，不是只换分类头。保留 NDOT 神经元的动态梯度实现、首步 `init=True`、每步输入 `x[:,t]`、相应参数差异正则和更新时机；不能改成普通 LIF 后仍称 NDOT。当前多个文件 import `data.utils/data.datapool/data.autoaugment`，但此副本缺 `data/` 包；现有 `datasets/` 也没有一一对应实现，不能简单全局重命名 import 即认为修好。先补齐作者原依赖/辅助模块，或在独立适配入口替换数据和统计辅助函数并核验等价性。

### SNN-FF

入口：`SNN_FF-main/FF_SNN/SNNFF_SHD.py`。顶层即创建 Tonic SHD 数据集，不能直接 import 以免触发下载；应先提取定义、加 main guard 和本地数据加载入口。

修改不止 input/output：此方法以标签候选的 goodness 分类，没有普通 K 维分类头。应追加 K 个标签通道，`x'=[label_one_hot,x]`，首层输入 C+K；遍历标签、负样本生成、标签注入都改成 K。现在的前二十通道覆盖法会覆盖真实特征，Texture 的 C=9 还会越界。追加标签通道是明确的输入适配变化，需三数据集一致采用并记录。

两个 RLeaky 写死 `linear_features=1000`，但 `n_units=500`，已复现矩阵维度错误；改为 output_size。第二层通过全局 `spk1_rec_global` 读取轨迹，正/负样本调用会覆盖它，不能保证各自对应的轨迹；应显式传递各自的时间轨迹并在层间 detach。当前训练确实调用 `loss.backward()`，因此不能照 README 说它完全没有梯度计算；应表述为局部 FF 学习而非端到端跨层 BP。学习率分段仅改 Python 变量，没有更新 Adam param_groups，应接 scheduler。训练中的逐类负样本评分和测试中的 K 次前向都必须计费。

### EfficientTrain++

入口：`EfficientTrain-main/_ET_pp_main_res50.py` 等调度脚本 → `main_buffer.py` → `engine_buffer.py`，数据入口 `datasets_buffer.py`。当前调度脚本写死 ImageNet 路径和八 GPU，不能原样提交超算。

该仓库是视觉骨干课程训练方法，不是 SNN 模型。省工作方案是明确选 ResNet-50 等一个骨干，增加 NPY 图像 Dataset、K 类输出及适合这些表示的变换，然后保留分辨率/课程阶段；现有 Spike/Texture 图像数组可作为输入来源，但须核验构造方式、数值范围，SHD 仍需本地事件转二维表示。禁用无意义的颜色增强，不能悄悄截断负值。若改成原生一维时序，则要重新设计低频/分辨率课程和骨干，已超出 input/output 修改，应标为方法扩展。[作者论文，Wang et al., 2024，DOI:10.1109/TPAMI.2024.3401036](https://arxiv.org/abs/2405.08768)

### GateZO-BDNN

SHD：`gate_zo_bdnn/run_shd_adlif_gatezo_task.py` → `train_shd_radlif_residual_proposal.py` → `train_shd_individual_pool_structural_zo.py`，动态后端为 residual/advanced RadLIF。Texture：`texture_bdnn_gatezo_bundle/run_texture.py` → `gate_zo_bdnn/run_texture_comparison_task.py` → `BDNN1GateZO/BDNN2GateZO` → `structural_gate.run_gatezo_growth`。

Texture 分支自身的矩阵维数大多自适应，改全十二类主要是绕开六类两阶段 loader；但若把 SHD R0 搬到 Spike/Texture，还需替换 HDF5/speaker loader，参数化 trainer 的 700 输入与 `shd_gatezo_support.py` 中的多个固定 20 类，修改频带宽度、延迟单位、jitter 与验证划分。尤其九个传感器通道不能直接套用 SHD 的连续频带与 sigma=20 通道抖动。这一分支移植不能概括成“只改几个超参”。

## 4. 最省工作的计量结构

保留每仓库原训练循环，只共享一个轻量工具模块：阶段起止、能耗读数、JSONL 输出、原子 checkpoint 写入、RNG 保存。每方法单独提供 `export_training_state/import_training_state/export_inference_state`，不建立统一 Trainer，也不强制所有算法继承同一个模型类。模型适配、完整恢复、FLOPs 特殊算子仍必须逐方法处理，外层 shell 包装无法正确替代这些工作。

### 能耗

对受支持且独占的 NVIDIA GPU，在训练区间前后 synchronize，并读取同一 GPU UUID 的累计能量：`E_gpu_J=(end_mJ-start_mJ)/1000`。NVML 读数属于设备，不是进程；CUDA_VISIBLE_DEVICES 重编号后要按 UUID/PCI 映射，不能把 cuda:0 当物理 GPU 0。累计能量不可用时再轮询功率，用梯形积分得到估计值；读数不可用/计数器重置填 null 和原因，不填 0。[NVIDIA NVML 官方接口说明](https://docs.nvidia.com/deploy/archive/R535/nvml-api/group__nvmlDeviceQueries.html)

逐段记录 train、validation、final_refit、test、checkpoint_io。GPU 能耗不能代替整节点能耗，尤其原 BDNN 的 CPU/NumPy 求解占比高；若集群支持节点能耗采集且能独占节点，额外记录 node_energy_J，否则结论限定为 GPU 能耗。Slurm 作业级能耗用于整体核对，不能直接按 epoch 平分成逐 epoch 测量。[Slurm 能耗采集配置](https://slurm.schedmd.com/acct_gather.conf.html)

### FLOPs

没有对这七个异构实现同时完整准确的一行自动统计器。建议正式能耗运行关闭重型 profiler；另做算子审计/成本标定，再按每轮实际形状和调用次数累计。计数约定：一个浮点乘法和一个浮点加法各一 FLOP，dense MAC=2 FLOPs；矩阵乘 `[m,k]@[k,n]` 使用约 `2mkn`。前向、局部/全局反向、更新、候选提取、筛选、失败轮、ridge/refit 均纳入实际阶段，不能简单用 forward FLOPs×3。

PyTorch profiler 的 `with_flops=True` 仅覆盖特定算子，官方主要列出矩阵乘和二维卷积。不能直接把它的和叫完整 training FLOPs；NumPy pinv/SVD、ridge 求解、FFT、神经元逐元素状态、比较/位运算及自定义算子需分别核对或解析补计。初期可如实输出 `flops_counted`、`coverage=partial` 和 omitted_ops，完整 FLOPs 留 null；不能用不完整数字作最终跨方法排序。[PyTorch profiler 官方文档](https://docs.pytorch.org/docs/stable/profiler)

真正的整数加乘应另报 int_ops/bit_ops；该 RSNN 当前 float32 模拟整数数值的运行路径应按实际运算类型记录，并把逻辑精度另列。理论神经形态 MAC/AC 能耗估算不能冒充超算实测 Joule。只有固定结构、固定时间长度和固定训练路径时才可标定 batch 成本后外推，末尾小 batch、FF 的 K 次评分、GateZO 每轮变宽、EfficientTrain 每阶段分辨率都需单独处理。

### 准确率与模型大小

accuracy 使用固定 eval 模式的 `correct/total`，区别于训练中变化权重所得 online train accuracy。普通分类逐 epoch 报 validation accuracy；增长方法逐 attempted round 报，拒绝轮也记录累计成本和当前准确率。建议主图使用 accuracy 对累计 Joule、累计 FLOPs和累计 wall time，另保留各自 epoch/round 曲线。一轮 GateZO 含多次训练数据处理、消融和 ridge，绝不能宣称它等价于一次 epoch。

大小分别存 `inference_model_bytes` 与 `resume_checkpoint_bytes`，读取实际文件字节数（MiB=bytes/2^20）。推理文件包含结构、权重、偏置、动力学、量化尺度和归一化统计；GateZO 需冻结连接/延迟/tau、ridge beta/feature mean/scale；不含 optimizer、RNG、历史日志和缓存。BDNN W/U/Beta 不是标准注册参数，不能只数 model.parameters。RSNN 只有实际导出 int8 并验证预测等价，才能称 int8 文件大小，不能按配置位宽假算。

推荐逐阶段记录：

```json
{"run_id":"...","unit":"epoch|growth_round","index":1,"phase":"train","status":"committed","accepted":true,"n_samples":0,"time_steps":0,"hidden_size":0,"wall_seconds":0.0,"gpu_energy_j":null,"node_energy_j":null,"flops_counted":null,"flops_coverage":"unmeasured","validation_accuracy":null,"checkpoint":"..."}
```

此外配置文件保存数据身份与 split 哈希、源码/依赖版本、输入表示、seed、硬件、计量范围。失败或重跑尝试单独记录，以便区分训练轨迹累计成本和实际作业总耗费。

## 5. 断点续训的现状与实现边界

| 方法 | 当前状态 | 必须补齐 |
|---|---|---|
| EfficientTrain | utils_buffer 有模型、optimizer、scaler、EMA 等恢复；main 有 epoch | RNG、数据顺序、原子写；课程阶段/分辨率/replay 状态和恢复语义；保存间隔固定为每 epoch |
| SLTT | `-resume` 恢复 net/optimizer/scheduler/epoch | AMP scaler、RNG、数据 sampler/generator、原子写；验证集选择替代 test |
| NDOT | `--resume` 从 latest 恢复模型/optimizer/scheduler/epoch | AMP scaler、RNG、原子写；先补缺包与数据适配 |
| Integer-RSNN | 没有完整 epoch 续训路径 | 专门 cell 状态序列化、epoch/学习率/量化状态/RNG；关闭交互绘图 |
| SNN-FF | 没有完整 epoch 续训路径 | model+Adam+scheduler+epoch+RNG；修复全局轨迹传递与入口 |
| 原 BDNN | 最终保存 W/U/Beta，非增长过程恢复 | 当前/最佳结构、误差与残差、停止窗口、round、RNG、数据 split |
| Texture GateZO | 内存中 best snapshot 与训练结束文件 | 每轮磁盘快照、selector RNG/history、最佳结构、停止状态 |
| SHD GateZO | 有 `--resume`，实际读写 `latest_growth.pt` | 配置/数据身份强校验、原子写、恢复一致性测试；说明文档的 latest_structural_zo.pt 名称与代码不同 |

SHD R0 的候选随机数按 seed+round 构造，数据增强也是确定性按样本构造，不能仅凭未存全局 RNG 就断言其恢复必然随机漂移；仍需验证中断前后结构、ridge 和历史一致。现有恢复代码主要校验 structural_partitions，不能当成完整配置身份校验。

语义应明确为：完成 epoch e 后提交快照；若在 e+1 中途断掉，从 e 的快照重新运行 e+1。要恢复半个 epoch 中的具体 batch，则还需 sampler 游标、部分指标、累积梯度等，当前需求可先避免这层复杂度。梯度训练的 optimizer 状态必须随模型恢复，不能只加载权重。[PyTorch 保存与恢复官方教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

最稳妥的边界提交方式：每个 epoch/round 保存一个含该轮指标的临时文件，flush/fsync 后在同一文件系统原子替换为不可变 `epoch_0001.pt`；扫描最后一个有效 checkpoint 恢复，JSONL/CSV 可由已提交 checkpoint 重建或按 run+unit+index 去重。恢复文件与 inference 文件分开；保留上一份有效快照，避免最后写入中断毁掉唯一文件。Slurm 发预终止信号时只设置 stop flag，在边界结束并提交；不能在 signal handler 里直接复杂序列化，也不能假设 SIGKILL 会给保存机会。

必须做两项反向验证：同种子连续两轮与一轮保存后新进程恢复一轮比较结构/权重/optimizer/scheduler/RNG；在第二轮中途强制退出，检查重新运行后没有漏轮、重复指标或损坏唯一 checkpoint。确定性条件下应精确一致，否则说明容差和非确定性来源。本次没有实施这些恢复测试，因此不宣称续训已可用。

## 6. 本机探针与交付

环境：`D:\anaconda3\envs\py310\python.exe`，实测 torch 2.11.0+cu128，CUDA 可用，但探针采用 CPU、单线程。现有 scipy/sklearn/snntorch/torchvision/tensorboard；缺 tonic/spikingjelly/timm。没有安装依赖或下载数据。

| 探针 | 范围与结果 |
|---|---|
| Integer-RSNN Spike | 4 个真实样本、从全时间跨度抽 16 步、hidden=8、一 batch 的一次 epoch；权重更新、有限、输出维正确 |
| Integer-RSNN Texture | 同样最小范围，直接实数输入；接口更新成功，不代表整数编码协议成立 |
| BDNN | 单次候选搜索，受控不等式只让一个候选合格；复现要求三个却返回一个并报成功；不是完整训练 |
| SNN-FF | AST 提取原定义，避开顶层下载；复现 500/1000 维度冲突与标签 11 对九通道越界 |
| Texture GateZO 核心用于 Spike | 最多一次增长尝试、四样本十六步；没有合格 checkpoint，保留异常 |
| Texture GateZO 核心用于 Texture | 四样本不足以满足内部类别划分，明确拒绝；这是探针样本约束，不据此判定正式训练算法错误 |
| SLTT / NDOT / EfficientTrain | 调用链静态核对；未完成适配训练，不能声称单 epoch 已通过 |
| SHD | 未运行，未下载 |

复现实例：

```powershell
& D:\anaconda3\envs\py310\python.exe audit/minimal_probe.py rsnn --dataset spike
& D:\anaconda3\envs\py310\python.exe audit/minimal_probe.py rsnn --dataset texture
& D:\anaconda3\envs\py310\python.exe audit/minimal_probe.py bdnn
& D:\anaconda3\envs\py310\python.exe audit/minimal_probe.py ff --dataset texture
```

全部实测 JSON 在 `audit/results/`，失败也保留 traceback。下一步实施顺序：确定全类别/两阶段与输入表示 → 修 BDNN/FF 的可复现问题并核实 Spike 包 → 各仓库独立适配入口 → 每方法恢复往返测试 → 接入统一阶段计量 → 超算小作业验证能耗权限与 FLOPs 覆盖 → 正式实验。无需统一大框架，但不能跳过这些方法级差异。
