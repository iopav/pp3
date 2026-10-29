# 独立 Texture 实验

总入口 `run_all.py` 顺序执行 RNN、RSNN、SNN surrogate、ANN、GateZO、普通 BDNN 六个独立进程；失败即停止，同一时间只运行一个方法。梯度模型默认单隐藏层49单元、100 epoch、batch size 128、seed 57，使用 Adam lr=0.001、betas=(0.9,0.999)、eps=1e-8。模型结构配置位于 defaults.json。GateZO 和普通 BDNN 保留候选搜索和闭式求解，不使用 Adam。

```bash
python texture_experiments/run_all.py --gpu 0
python texture_experiments/run_all.py --experiments rnn rsnn snnsurrogate ann gatezo bdnn
python texture_experiments/run_all.py --cpu --smoke --output /path/to/new-output
python texture_experiments/run_all.py --dry-run
python texture_experiments/bdnn.py --device cpu --max-hidden 49
sbatch texture_experiments/submit.sbatch --output texture_experiments/outputs_adam_bdnn
```

## 数据与模型

默认读取 Texture-mini 的 texture_spikesT450_T77.npz，保留原 train/test 划分、二值脉冲和全部时间步，不重复归一化或编码。输入形状为 (960,77,9)/(240,77,9)，12类 one-hot 标签转整数。Windows 默认目录为 D:/workspace/Avalanche/avalanche-master/dataset/texture-mini，Linux 为 /home/yizhil/pp3/data/texture-mini；可用 --data-root 指定目录或 NPZ 文件。保留 --dataset texture 和 shd 适配，维度由数据适配器提供。

RNN 使用 tanh；ANN 对时间步共享 Linear+ReLU 后求平均。SNN 使用 LIF，beta=0.9、阈值1、减法 reset，reset 梯度 detach，fast-sigmoid surrogate slope=25；RSNN 增加隐藏脉冲递归连接。四个梯度模型保留完整时间反向传播。Python、NumPy、PyTorch 使用统一 seed，禁用 cuDNN benchmark，启用严格确定性算法；不支持确定性实现时明确报错。

GateZO 调用相邻 texture_bdnn_gatezo_bundle，使用既有 validation_checkpoint 协议，nB=1、L_max=49、T_max=20，目标49个实际隐藏单元；候选失败或未达到目标判失败。保留结构搜索所需的训练侧验证，最终测试集不参与结构选择。详细增长规则由现有 growth_policy.py 适配层控制，原 bundle 不修改。

普通 BDNN 的 bdnn.py 调用相邻 BDNNversion1-main 原生 sc_Search、addNodes、upgradeSCN、getH 和 computeBeta；不修改源目录。固定 nB=1、目标49单元，可用 --max-hidden 覆盖；取消误差和平台期提前停止，候选失败、非有限结果或宽度不符判失败。候选搜索保留原实现始终使用训练目标 T 的规则。适配器绕过原 classification() 的逐轮测试和按测试准确率裁剪，最终重拟合后只测试一次，保存实际 W/U/Beta。

普通 BDNN 原代码使用 CPU 张量及 NumPy，--device 仅指定其 GPU 遥测设备，不会将计算迁移到 GPU。该方法的 GPU 板卡能耗不等于 CPU 训练能耗。--epochs、--batch-size 对它仅作总入口兼容，实际使用全数组计算。GateZO 的 --epochs 同样是兼容参数，增长规模由隐藏单元目标决定。

## 指标与测量边界

梯度模型每轮 train_acc 使用训练批次更新前预测的正确数除以样本数，不额外评估训练集。测试集只在训练结束后评估一次，final_test_acc 仅填入最后一行。GateZO 使用既有增长训练子集准确率，最终行记录重拟合后全训练集准确率；普通 BDNN 记录各增长轮当前模型训练准确率。不同方法的训练准确率口径不完全相同。

ANN 每10轮、其余梯度模型每5轮形成连续能耗窗口；窗口中不执行报告评估、CSV 写盘或主动日志打印。使用 NVML 累计毫焦耳计数器差值除以1000获得焦耳，按各轮训练耗时分摊。兼容参数 --eval-each-epoch 仅改为单轮能耗窗口，不增加逐轮测试。GateZO 和普通 BDNN 每5个增长尝试形成窗口，最终重拟合另开窗口并追加最后一行，沿用最后增长轮编号。

| 字段 | 定义 |
| --- | --- |
| para_mem_bytes | 最终保留模型参数的实际 dtype 字节总和，包含读出；BDNN/GateZO 为 W/U/Beta |
| optimizer_state_bytes | 每轮最后更新后的优化器 state 存储，按底层存储去重，不含 param_groups |
| hiddenstatememfor bp | 每轮各 batch 中 autograd 为 BP 保存的非参数张量底层存储存活峰值，包含保留的输入、标签及中间量 |
| auxiliary_mem_bytes | optimizer_state_bytes 与本轮 BP 保存存储峰值之和，不是同时刻运行时内存峰值 |
| train_cross_entropy | 梯度模型的训练批次交叉熵，按样本数加权 |
| energy_j / power_w | 窗口能量按耗时分摊的逐轮估计 / 逐轮测量时长 |
| total_energy_j / total_power_w | 累计训练能量 / 累计训练测量时长，不对功率求和 |

BDNN/GateZO 无优化器、无 BP，其对应两项及 auxiliary 为0，这不表示运行时无需内存。梯度模型使用 saved_tensors_hooks 记录存储，不克隆或卸载张量；计量本身的开销包含在耗时和能耗中。遥测不可用留空，不回退到功率积分；相等计数器读数保留零差值，短窗口可能小于传感器更新分辨率。

总入口训练开始前约3秒、每0.1秒采样所选 GPU 的板卡功率，只打印一次均值；独立方法启动也测量一次，总入口启动的子进程不重复。该值标记为训练前板卡功率，不保证设备独占空闲，不从训练能耗中扣除。CPU 或接口不可用明确打印原因。

## 保存、跳过与验证

输出目录为 outputs/数据集/full/方法/seed_编号，冒烟使用 smoke 子目录。config.json 保存配置及指标版本，metrics.csv 保存数值指标，model.pt 保存模型；梯度模型还保存 Adam 一阶矩、二阶矩、step 和 completed_epochs。status.json 记录成功或失败；BDNN/GateZO 另存 summary.json 与 final_metrics.json。没有实现中断现场恢复或精确续跑。

仅配置相同、status 成功且必要文件非空的实验自动跳过；既有失败、不完整或配置不同的目录拒绝覆盖，需新 --output。Adam 与旧 SGD 配置不匹配。历史 CSV 不改写，不把原始运行显存、checkpoint 文件大小或旧估计量当作本次内存指标。

梯度模型冒烟最多8样本、前16步、1轮；BDNN/GateZO 每类最多4样本、前16步，仍需达到目标49单元。冒烟不表示完整训练收敛。相关回归测试包括 test_bdnn.py、test_startup_power.py、test_energy_blocks.py、test_gatezo_energy_blocks.py、test_bp_memory.py、test_state_memory.py、test_completion.py 和 test_gatezo.py。旧 test_gatezo_metric_parity.py 另外需要本地参考 spec 目录，SHD 测试需要 h5py；缺少这些依赖时不能声称整个测试发现流程通过。
