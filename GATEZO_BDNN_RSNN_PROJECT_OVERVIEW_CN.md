# GateZO-BDNN-RSNN 项目说明

## 1. 项目目标

本项目研究一种不依赖反向传播的增量式脉冲神经网络，用于 Spiking Heidelberg Digits（SHD）20 类语音识别。

核心约束是：

```text
不使用BPTT训练隐藏层
随机生成候选神经元及其动力学参数
候选生成后立即冻结
通过结构筛选决定保留哪些候选
通过闭式ridge求解输出权重
网络按轮自然增长
```

项目目前的中心方法称为 `GateZO-BDNN-RSNN`。其中：

- `BDNN` 表示按候选块逐轮增长的随机网络框架；
- `RSNN` 表示带递归连接的脉冲神经网络；
- `GateZO` 表示利用无梯度的坐标硬消融估计候选神经元的结构贡献；
- 当前主要隐藏神经元采用冻结的 RadLIF 动力学。

“无反向传播”不等于“没有监督信息”。标签仍用于 ridge 拟合、候选评价和 speaker-validation 增长决策，但不存在通过链式法则更新隐藏层权重或动力学参数的过程。

## 2. 数据与评估协议

数据集：SHD，包含 700 个输入通道和 20 个类别。

主要五 seed 实验使用：

```text
57, 58, 59, 60, 61
```

训练集合内部按说话人划分为：

```text
structure-training
speaker-disjoint structure-validation
```

典型流程为：

```text
训练数据生成/筛选随机候选
→ speaker-validation决定本轮增长是否接受
→ 在完整训练数据上重新拟合ridge
→ test只做最终报告
```

早期和探索性实验曾直接比较多组 test 准确率，因此相关结果应明确标记为 exploratory，而不能统一宣称为完全未触碰测试集的最终泛化估计。

## 3. 网络动力学

### 3.1 RadLIF 状态

每个隐藏神经元维护：

```text
突触电流 s[t]
膜电位 u[t]
脉冲 z[t]
慢适应电流 A[t]
```

适应状态更新为：

```text
A_i[t] = rho_i A_i[t-1]
       + (1-rho_i) a_i u_i[t-1]
       + b_i z_i[t-1]
```

其中：

```text
rho_i = exp(-dt / tau_adapt,i)
tau_adapt ~ LogUniform(60, 200) ms
a ~ Uniform(-0.5, 0.5)
b ~ Uniform(0.05, 0.50)
```

突触和膜电位更新为：

```text
s_i[t] = alpha_i s_i[t-1] + I_i[t]

u_tilde_i[t] = beta_i u_i[t-1]
             + (1-beta_i)(s_i[t] - A_i[t])
```

达到阈值后产生脉冲并执行硬重置。RadLIF 的慢适应状态为语音事件提供了比普通 LIF 更长的时间记忆。

### 3.2 冻结随机参数

以下参数均由预先声明的随机分布生成，然后冻结：

```text
输入权重
旧神经元到新神经元的递归权重
候选自递归权重
bias
tau_syn
tau_mem
tau_adapt
a
b
输入频带和输入延迟
```

候选神经元可以接收旧神经元上一时刻的脉冲，但新候选不会反向改变已提交的旧神经元：

```text
W_new,old != 0
W_old,new  = 0
```

因此，添加候选时旧隐藏特征保持不变，适合增量增长和坐标结构评估。

## 4. 多分支稀疏延迟输入

当前 R0 配置为每个候选使用四个冻结输入分支：

```text
分支1：全700通道，0 ms延迟
分支2–4：随机连续局部频带，使用非零稀疏延迟
```

局部频带宽度从以下集合选择：

```text
{32, 64, 128, 256}
```

延迟从以下集合选择：

```text
{0, 10, 25, 50, 100, 200} ms
```

这一结构让单个神经元可以同时观察当前全频活动和多个历史局部频带。它是项目中的定制组合设计，不应表述成对某一篇论文网络结构的逐项复现。

## 5. 数据增强与异质时间常数

训练阶段使用：

```text
通道抖动：sigma=20，copies=1
时间缩放：gamma ~ Uniform(0.90, 1.10)
```

验证集和测试集不执行这些随机增强。

候选中 50% 使用固定时间常数：

```text
tau_syn = 10 ms
tau_mem = 20 ms
```

另外 50% 使用异质时间常数：

```text
tau_syn ~ LogUniform(5, 25) ms
tau_mem ~ LogUniform(15, 60) ms
```

异质性使随机候选池覆盖不同的时间尺度，由结构搜索选择更适合任务的动力学，而不是通过梯度学习时间常数。

## 6. GateZO 与无 ZO 对照

### 6.1 GateZO 结构选择

每轮生成 512 个随机候选，公共预筛先根据候选与当前分类残差的相关性及候选之间的多样性，提交最多 16 个候选给结构选择阶段。

GateZO 的核心过程为：

```text
临时加入整个候选块
→ 逐个将候选坐标硬关闭
→ 测量移除该候选造成的loss变化
→ 按结构贡献排序
→ 评价候选前缀
→ 最小化 calibration loss + lambda × 保留数量
→ 提交选中的冻结神经元
```

当前典型设置：

```text
candidate_pool_size = 512
growth_step = 16
zo_min_keep = 8
zo_sample_size = 2048
zo_sparsity_lambda = 0.000025
diversity_strength = 0.7
```

GateZO 不计算反向梯度，也不更新候选的权重。它优化的是离散结构选择。

### 6.2 无 ZO：fixed16

无 ZO 对照配置为：

```text
config_id = R0_fixed_prior_no_gatezo_fixed16
selection_mode = fixed16
```

它与 GateZO 配置使用相同的 RadLIF、数据增强、四分支稀疏延迟、异质时间常数、候选池、随机种子、ridge 和停止规则。每轮仍然先生成 512 个冻结随机候选，再执行公共的残差相关性与多样性预筛；区别是预筛出最多 16 个候选后，不再执行逐坐标硬消融、ZO 排序或稀疏前缀选择，而是把这批预筛候选整体交给外层增长门。

因此无 ZO 方案中：

```text
zo_steps_per_growth = 0
zo_coordinate_queries_per_growth = 0
每轮最多提交16个预筛候选
```

无 ZO 方案仍然保留 speaker-validation 外层增长接受与停止，所以它不是“完全无筛选”的随机网络。这个对照专门测量 GateZO 坐标结构选择带来的增量贡献。

## 7. 自然增长和停止

候选被 GateZO 选中后，不会自动永久加入。外层 speaker-validation 增长门要求验证指标达到预先声明的改进阈值。

典型设置：

```text
growth_min_delta = 0.002
growth_failure_patience = 12
max_growth_rounds = 120
max_hidden = 0
```

`max_hidden=0` 表示没有人为指定最终隐藏层规模。最终 hidden 数量由随机候选质量、GateZO 筛选和验证集停止共同产生，因此称为自然增长。

## 8. 时间读出

基础 R0 读出不是 max-over-time，也不是累积和。当前方式是：

```text
隐藏轨迹
→ 去除warmup
→ 划分16个时间窗口
→ 每窗求均值
→ 拼接
→ 标准化
→ 闭式ridge分类
```

累积和读出已进行过对照，没有成为当前主配置。

## 9. 代表性实验结果

本说明只记录当前打包方法的主结果：

| 方法 | 五 seed 平均测试准确率 | 说明 |
|---|---:|---|
| `R0_fixed_prior_gatezo`（历史 `R0_fixed_prior`） | 83.065% | 当前打包的 GateZO 配置 |
| `R0_fixed_prior_no_gatezo_fixed16` | 待测 | 当前打包的无 ZO 对照配置 |

## 10. 代码结构

主要模块：

| 文件 | 作用 |
|---|---|
| `train_shd_bdnn_rsnn_pure.py` | SHD 读取、事件分箱和基础动力学支持 |
| `train_shd_bdnn_rsnn_blocks.py` | speaker split、数据增强和 DataLoader |
| `shd_adlif_gatezo_core.py` | adLIF/RadLIF 状态与适应动力学 |
| `shd_radlif_delay_gatezo_core.py` | 延迟 RadLIF 基础实现 |
| `shd_radlif_advanced_gatezo_core.py` | 多分支、局部频带和异质 tau |
| `shd_gatezo_support.py` | ridge、采样、相关性和多样性预筛 |
| `train_shd_individual_pool_structural_zo.py` | 自然增长、GateZO/fixed16 和 checkpoint |
| `train_shd_radlif_residual_proposal.py` | R0/残差提议系列训练入口 |
| `summarize_*.py` | 各实验系列结果汇总 |

训练调用链通常为：

```text
Slurm数组脚本
→ task dispatcher
→ 训练入口
→ 自然增长/GateZO控制器
→ RadLIF动力学后端
→ final checkpoint + summary.json
```

## 11. 当前 R0 GateZO/无 GateZO 对照包

共享归档：

```text
/nobackup/proj/disk/naiss2026-4-902/shared/
shd_radlif_r0_gatezo_vs_no_gatezo_with_container.tar
```

它专门比较：

```text
R0_fixed_prior_gatezo
R0_fixed_prior_no_gatezo_fixed16
```

历史 `R0_fixed_prior` GateZO 五 seed 结果为：

| Seed | 测试准确率 | Hidden |
|---:|---:|---:|
| 57 | 81.8463% | 166 |
| 58 | 83.2597% | 240 |
| 59 | 83.4364% | 295 |
| 60 | 82.8622% | 248 |
| 61 | 83.9223% | 203 |
| 平均 | 83.0654% | 230.4 |

包中的 no-GateZO `fixed16` 是新对照配置，应实际运行后再填写结果。它仍保留公共预筛和外层验证门，因此实验回答的是“GateZO 坐标筛选的增量贡献”，而不是“整个结构学习是否有用”。

## 12. 输出与复现记录

每个 seed 目录通常保存：

```text
search_config.json
latest_structural_zo.pt
final_structural_zo.pt
summary.json
```

汇报结果至少应记录：

```text
Git/源码快照或归档SHA-256
容器身份
数据文件身份
配置ID和seed
held-out speakers
test accuracy
hidden size
accepted/attempted rounds
运行时长
是否使用test做选择
```

任务中断时，可使用同一配置、seed 和输出目录从 `latest_structural_zo.pt` 恢复。不要跨 seed 或跨配置复制 checkpoint。

## 13. 项目的科学主张

当前证据支持的表述是：

> 随机冻结的 RadLIF 候选可以通过无梯度结构搜索逐步组成有效的 SHD 表征；多分支稀疏延迟和异质时间常数能够配合闭式 ridge 完成 SHD 分类。

当前证据不支持以下过度表述：

- 隐藏层完全无监督；
- GateZO 是连续参数优化算法；
- 当前系统复现了某篇论文的全部结构；
- 当前结果已经稳定达到 90%。
