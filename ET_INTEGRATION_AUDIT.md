# EfficientTrain++ 接入与方法身份检查（2026-10-14）

## 修正结论

旧的 `efficienttrain` 分支直接运行 `main_buffer.py`，使用固定尺寸浮点输入，遗漏上层课程和原图像变换，实际是普通 ResNet50 基线。这些历史结果不能作为 ET 结果。现在统一入口调用 `_ET_pp_main_res50.py`，读取该入口的十阶段分辨率表，经过 `pp3_et.launch` 将课程交给原 `main_buffer.main`，继续使用原 ResNet50、优化器创建函数、`train_one_epoch` 和 `evaluate`。

本地仓库是 EfficientTrain++。参考：[官方代码](https://github.com/LeapLabTHU/EfficientTrain)、[Wang et al., EfficientTrain++, TPAMI, arXiv:2405.08768](https://arxiv.org/abs/2405.08768)，DOI: 10.48550/arXiv.2405.08768。这里接入其 ResNet50 发布配方，并非重新搜索课程，也不是 ICCV 2023 原版的独立复现。

## 课程与预算

- 十阶段尺寸为 `[96,96,160,160,160,160,224,224,224,224]`，增强调用原 `datasets_buffer.build_transform`，使用 `rand-m0-mstd0.5-inc1` 到 `rand-m9-mstd0.5-inc1`。低分辨率阶段保留发布代码的 PIL Lanczos 缩放；没有把普通双线性缩放冒充 Fourier cropping。
- 阶段总调度长度为 `int(E * (224*224)/(size*size))`，阶段结束坐标及跨尺寸起始坐标分别沿用上层脚本和 `utils_buffer.auto_load_model` 的取整公式。`E=100` 对应 **226 个实际训练 epoch**；`--epochs` 对 ET 表示全尺寸等效的面积预算参数，不能与其他方法的原始 epoch 数直接相等比较。面积只是原配方的近似预算，实际 FLOPs/耗时仍以记录为准。
- 学习率与权重衰减使用原 cosine scheduler 对应阶段坐标的切片。模型和优化器只创建一次，跨阶段不重置；替代的是原来磁盘 checkpoint 的传递方式。短预算的 warmup 上限设为该阶段调度总长减一，避免原调度数组越界；默认 100 预算不受影响。
- 测试固定使用最终尺寸 224 和确定性原评估变换；测试准确率仅记录，不控制课程、提前停止或选择模型。当前统一配方仍保留配置中的 batch size、mixup=0、cutmix=0、关闭 AMP/EMA 等选择，因此这是脉冲数据上的 ET++ 适配实验，不声称逐项复现 ImageNet 超参。
- `--smoke` 会执行完整十阶段，每阶段一个实际 epoch，尺寸缩放为 32/46/64，测试固定 64。它只验证机制，不能用于速度或准确率结论。完整运行要求预算至少 10，输入基准尺寸为 224。

## 数据表示边界

读取原 Spike/Texture/SHD 公共 Dataset，将时间×通道平面双线性缩放至基准尺寸。为真正调用原 PIL RandAugment 和 Lanczos 管线，使用 **只由训练输入计算的全局 min/max** 映射为 0–255、量化为 8 位并复制为 RGB，随后进入原 transform。所有样本共用同一范围，不逐样本归一化，不使用测试样本拟合范围。负值先平移缩放，不直接清零；测试超出训练范围的值会被截断。常量训练集映射为零图像。

这会引入量化、插值、幅值截断和图像增强，不能宣称保留原始脉冲时间语义。原图像增强的标签保持假设仍需通过正式实验验证；本次完成的是接通原 ET++ 流程。模型仍为 ANN ResNet50，不是脉冲 ResNet。训练集范围扫描和数据处理的时间/能耗纳入 prepare/train 测量，CPU/PIL 数据预处理 FLOPs 沿用原统计口径排除。

## 证据与历史结果隔离

每次运行生成 `et_curriculum.json`（预算、阶段、编码范围及变换）、`et_stages.jsonl`（阶段入口与优化器累计步数），每条 `metrics.csv` 包含阶段、计划尺寸、实际观测尺寸、增强和调度坐标。原训练函数对每批实际输入形状执行检查。`pp3.completed.completed_status` 要求完整十阶段的逐轮记录和协议标识；没有这些证据的旧固定尺寸结果不会被跳过复用，也不会被新汇总器纳入 ET 对比。原始历史文件不做修改。

运行方式：`python run_experiments.py --methods efficienttrain --datasets spike texture shd --seeds 57 --output result_et`。沿用既有数据路径配置，或添加 `--data-root` / 各数据集路径。正式执行前要为 ET 的实际 epoch 数和训练集范围扫描预留资源。

## 其他方法的实际调用链审查

以下是当前统一入口的代码路径检查，不代表各论文的完整复现认证，也不代表所有方法重新完成收敛实验。

| 方法 | 确认执行的核心路径 | 判断与限制 |
|---|---|---|
| SPSN | `run.py` 强制 `SPSN-<spike_mode>`，`network.create_network` 创建 `neurons.spsn.SPSN`，原训练执行反向传播与 Adamax | 没有切换到代码中另有的 Non-Spiking/LIF 基线 |
| TP | `utils.run` 设置 `algorithm=TP`，构建 `TP_MLP`；`train(custom_grad=True,T=1)` 每时间步调用 `model.update()` | 没有走普通 BP 分支 |
| FFSNN | `SNNFF_SHD.py` 调用 `make_examples` 注入正负标签，逐层 `loss_fn(h_pos,h_neg)`、反向与局部更新 | 第一层 Forward-Forward 路径存在，但第二层 `LeakyLayer2.forward` 忽略传入的正负表示，读取同一全局 `spk1_rec_global`；最小复现得到相同正负输出和零 swish loss，梯度仅剩约 1.9e-6 的数值残差。另有分段学习率只赋值 Python 变量、未同步 optimizer 的问题；均未在本次 ET 修改中重写 |
| ESPP | 原 `classwise_loader.next_item` 交替同类/异类，`utils.train` 调用网络局部梯度和 SGD 更新 | 未替换为普通监督 BP；准确率另用原读出器拟合，不能把读出训练成本忽略 |
| BDNN | `bdnn1.classification → sc_Search → inequalityEq → addNodes → upgradeSCN/computeBeta` | 候选筛选与增长均在；原代码末尾仍按测试准确率决定截取结构，存在测试泄漏及截取索引问题，不能声称严格无泄漏比较；本次未改变原算法 |
| GateZO（Spike/Texture） | `run_texture_comparison_task` 强制 `selection_mode=gatezo`，`BDNN1GateZO.classification → run_gatezo_growth → StructuralGateZO`，执行逐候选硬屏蔽评分 | `zo_steps=0` 关闭的是旧连续权重 ZO，不是结构 GateZO；核心屏蔽查询存在 |
| GateZO（SHD） | R0 配置 `selection_mode=gatezo`，经 `train_shd_radlif_residual_proposal → train_shd_individual_pool_structural_zo` 调用门控选择 | `proposal_mode=prior` 是当前明确选用的 R0 固定先验版本，不应描述为残差自适应 proposal 实验 |

上述六种其他方法未发现与本次 ET 相同的上层入口遗漏，但 FFSNN 第二层正负分支失效和 BDNN 测试集选结构是实质问题，不能宣称“其余全部正确”。FFSNN 的最小复现直接提取并执行当前类与损失函数，证据在 `audit/ffsnn_shared_state_check.json`；本轮只修改 ET，实现问题保留供独立修正，避免把更改原算法与接入 ET 混成一个未说明的变更。

## 验证

自动测试覆盖原脚本阶段预算/恢复取整、原学习率坐标、原 RandAugment/Lanczos 变换、worker 获取更新后变换、确定性评估及拒绝旧基线结果。运行验证覆盖真实 ResNet50 的十阶段 GPU 训练和三类数据入口；Spike/Texture 使用专门生成的人工数组，SHD 使用本地小型 fixture，均仅用于验证执行。正式 96/160/224 路径另用人工 Spike 输入和 10 预算执行，不能把这些准确率当作真实数据性能。

GPU 检查结果见 `audit/et_execution_checks.json`：三个 smoke 各十个 epoch；正式尺寸检查共 21 个实际 epoch，依次观测到 96/160/224。十次阶段入口的优化器步数严格递增，低尺寸每轮已计数训练 FLOPs 小于高尺寸。运行状态均为 success。回归测试日志为 `audit/et_regression_tests.log`；这些检查不构成正式数据集的精度或墙钟加速实验。
