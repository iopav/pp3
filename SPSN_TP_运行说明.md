# SPSN、TP 三数据集运行说明

> 历史说明：本文对应旧公共 worker。当前主控直接启动原仓库脚本，方法范围、修改位置与 FLOPs 完整性以 [原脚本运行说明](原脚本运行说明.md) 和 [当前 FLOPs 审计](原脚本_FLOPs审计.md) 为准。

入口是 `run_experiments.py`，参数放在 `constants.py`。默认运行 SPSN、TP，三个数据集和 **57、58、59、60、61** 五个 seed，共 30 个独立任务。原模型目录没有修改；数据读取、训练适配和指标记录放在新建的 `pp3/` 目录。没有断点续训，重复启动会创建新的结果目录。

## 1. 数据与目录

把工程放到超算后，在工程根目录运行命令。运行时不需要 Avalanche。保留下面这些代码目录：`pp3/`、`gate_zo_bdnn/`、`Stochastic-Parallelizable-Spiking-Neuron-SPSN-main/`、`traces_propagation-main/`，以及顶层 `constants.py`、`run_experiments.py`。`gate_zo_bdnn/` 用于直接复用当前 SHD 读取、转换和扰动代码，不会启动 GateZO 训练。

| 数据集 | 正式输入 `[时间, 通道]` | 类别 | 数据目录中的文件 |
|---|---:|---:|---|
| Spike | 400 × 64 | 20 | `X_train_20_cla_400steps.npy`、`Y_train_20_cla_400steps.npy`、`X_test_20_cla_400steps.npy`、`Y_test_20_cla_400steps.npy` |
| Texture | 3775 × 9 | 12 | `X_train_texture.npy`、`Y_train_texture.npy`、`X_test_texture.npy`、`Y_test_texture.npy` |
| SHD | 2000 × 700 | 20 | `shd_train.h5`、`shd_test.h5` |

NPY 输入必须是 `[样本, 时间, 通道]`，标签必须是从 0 开始的整数向量。Texture 转为 float32，保留负值。缺文件或形状不符直接报错。SHD 在取样本时读取原始事件，训练扰动后再按 0.5 毫秒转换；测试不扰动，同一位置重复事件记 1。不下载、不生成 NPY、不生成磁盘缓存，也不在主控进程中展开整套数据。

SHD 直接调用现有 GateZO 的类，因此保留 float32 时间边界行为。默认保留原样本，另加一份扰动副本，通道扰动标准差 20，时间缩放 0.9–1.1。扰动由 seed、样本编号和副本编号决定，每轮重新读取仍得到相同扰动，**没有把随机结果写入文件，也没有改为每轮换一套扰动**。SPSN 的 SHD 神经元时间间隔设为 0.0005 秒；Spike、Texture 保留原方法的 0.001 秒设置。

## 2. 安装与启动

使用 Python 3.10 或更新版本，先在超算环境中准备与驱动匹配的 PyTorch 2.11，再安装本次依赖。GPU 能耗依赖 `nvidia-ml-py` 和超算是否开放读数；读不到时训练仍可进行，能耗字段留空并写明原因。

```bash
python -m pip install -r requirements-experiments.txt

# 先看实际配置，不训练，不创建输出文件。
python run_experiments.py --dry-run --data-root /path/to/dataset

# 正式运行：默认五个 seed，全部方法、全部数据集。
# 在 Slurm 已分配 GPU 时，省略 --gpus，继承 CUDA_VISIBLE_DEVICES。
python run_experiments.py --data-root /path/to/dataset --output outputs/formal

# 自己管理机器上的两张 GPU；每张卡同时只有一个任务。
python run_experiments.py --gpus 0 1 --data-root /path/to/dataset --output outputs/formal

# 只选一部分；数据也可以分开放，不要求放进同一个目录。
python run_experiments.py --methods tp --datasets spike shd \
  --spike-path /data/spike --shd-path /data/shd --output outputs/tp
```

`--data-root` 下应有 `spike/`、`texture/`、`shd/` 三个子目录；单独的 `--spike-path` 等参数优先。未在命令行指定时使用 `constants.py`，其中也可以通过 `PP3_DATA_ROOT` 环境变量设置数据根目录。路径按启动命令时的目录解析。主控给每个子进程设置 GPU，子进程内部统一使用 `cuda:0`。已限制 GPU 可见范围时，显式 `--gpus` 必须使用该范围内的标识，不能填另一张卡的编号。

Slurm 示例文件是 `submit_spsn_tp.sh`；需要按所在超算修改分区、时间、内存和环境加载命令。单个主控管理当前节点上的可见 GPU，不负责跨节点分配。五个 seed 会分别运行，不会合并成一次训练。

## 3. 修改超参

`constants.py` 的 `DATASETS` 保存路径、输入数、类别数、时间步和 SHD 处理参数；`METHODS[方法][数据集]` 保存各自训练参数。示例修改写在 `METHODS` 定义之后。正式训练默认 SPSN 200 epoch、batch 64、隐藏宽度 128、3 层、Adamax 学习率 0.001；TP 100 epoch、batch 128、隐藏宽度 450、1 层、Adam 学习率 0.0001。TP 保留每时间步更新，不把完整序列改成一次更新。

```python
METHODS['spsn']['texture']['batch_size'] = 16
METHODS['tp']['shd']['hidden_size'] = 450
METHODS['tp']['spike']['lr'] = 0.0001
```

只改某次运行的 epoch 可用 `--epochs 10`。数据形状参数不能随意修改来掩盖文件不匹配：Spike 当前适配的就是 400 步文件。增加其他模型选项后，若出现未覆盖算子，FLOPs 检查会报错，不能将部分计数当作完整结果。

## 4. 最小测试

`--smoke` 使用训练/测试各最多 4 个原始样本、均匀抽取 16 个时间位置、隐藏宽度 8、1 个隐藏层、batch 3、1 epoch。SHD 训练仍添加扰动副本，因此最多为 8 个训练样本。该模式只验证能运行，不能拿准确率或耗能比较模型。`--smoke-full-time` 保留完整时间轴，其他缩小设置相同。正式实验不要带这两个开关。

```bash
# 超算或本机 CPU 最小测试；明确指定一个 seed，避免重复五遍。
python run_experiments.py --cpu --smoke --seeds 57 \
  --datasets spike texture --data-root /path/to/dataset --output outputs/smoke

# 检查完整的 400/3775 时间步，仍然只有极少样本和一个 epoch。
python run_experiments.py --cpu --smoke --smoke-full-time --seeds 57 \
  --datasets spike texture --data-root /path/to/dataset --output outputs/full_time_check

# 有原始 SHD 时进行真实数据测试。
python run_experiments.py --cpu --smoke --datasets shd --seeds 57 \
  --shd-path /path/to/shd --output outputs/shd_check

python -m unittest tests.test_pp3 tests.test_worker tests.test_energy -v
python -m tests.check_model --method spsn
python -m tests.check_model --method tp
```

Windows 本机的数据根目录可写成 `"D:\workspace\Avalanche\avalanche-master\dataset"`。本次验证使用 `D:\anaconda3\envs\py310\python.exe`；缺少的 h5py/NVML Python 包只装到了工程内 `python_deps/`，没有修改全局环境。本机复测前可在 PowerShell 设置 `$env:PYTHONPATH = "$PWD\python_deps;$PWD"`。超算正常安装依赖后无需这个设置，也无需复制 Windows 的 `python_deps/`。

## 5. 输出怎么看

每次运行创建 `输出目录/run_日期_时间/`，其中 `metrics.csv` 合并各任务的逐 epoch 结果，`tasks.json` 列出任务退出码。任务目录按 `方法/数据集/seed_编号/` 排列，含生效配置、日志、状态、模型文件、逐阶段 FLOPs 明细。一个任务失败不会阻止其他任务；只要有一项失败，主控最终退出码就是 1。

| 字段或文件 | 含义 |
|---|---|
| `train_accuracy` / `test_accuracy` | 当前 epoch 准确率，范围 0–1；不用于自动调参或早停 |
| `train_flops` / `test_flops` | 训练/评估计算量，包含各自损失计算 |
| `train_complete` / `test_complete` | 当前计数规则是否覆盖实际执行的操作；不等于测得硬件指令数 |
| `*_known_flops` | 已有规则覆盖的部分；遇到未知操作时仍保留 |
| `*_fft_flops_estimated` | 已包含在对应 FLOPs 中的 FFT 估算部分，不能再加一次 |
| `*_seconds` / `*_data_wait_seconds` | 阶段总时间、等待取数的时间 |
| `*_gpu_energy_j` / `*_energy_method` / `*_energy_reason` | GPU 焦耳数、测量方法、缺失或读数限制说明 |
| `epoch_flops` / `epoch_seconds` / `epoch_gpu_energy_j` | 训练与评估之和；不含一次性模型准备和文件保存 |
| `cumulative_flops_including_prepare` | 一次性模型准备加上已完成的训练/评估 |
| `final_model_bytes` | 最后保存的 `model.pt` 实际文件字节数，包括配置和文件格式开销 |
| `prepare_flops.json` | 模型初始化，包括 SPSN 预先计算 FFT 核的开销 |
| `epoch_XXXX_*_operations.json` | 算子名、输入输出形状、调用次数、规则和分阶段合计 |
| `environment.json` | Python/PyTorch/NumPy、设备、样本数和数据加载器准备时间 |

`model.pt` 用于推理，含必要参数和配置，不含优化器及每批次的临时轨迹；TP 不保存推理不用的目标传播分支。使用 `pp3.models.load_inference(path, device)` 恢复模型。这不是断点续训文件。模型创建开销单列，训练/评估区间包含取数等待和计数工具开销。CPU 数据处理的耗能没有测量，不能用 GPU 能耗代替整机能耗；也没有偷偷再跑第二遍训练来测量。

## 6. FLOPs 与本次测试范围

详细公式和边界见 `SPSN_TP_FLOPs说明.md`，实际验证记录见 `audit/SPSN_TP_验证记录.md`。所有 FLOPs 都按明确的运算规则计数，FFT 使用长度公式估算；对未识别操作保留明细、留空完整总数并让任务失败。短至毫秒的 GPU 区间可能受硬件计量更新周期影响，出现零增量或读数跳变；最小测试只用于检查读数通路，不能据此做能耗排名。

FFSNN、ESPP、EfficientTrain 本次不接入，也没有修改它们的学习规则。此前 FFSNN 的“500/1000”指的是隐藏层宽度写成 500、递归矩阵另一处写成 1000，属于尺寸没有对应；Texture 的 9 通道不能直接容纳原脚本的 12 个类别注入位置，则属于换数据集必需的接口适配。二者都不影响本次 SPSN/TP。EfficientTrain 更换 Spiking ResNet 的工作按确认范围暂缓，没有把未适配的入口声称为可直接运行。
