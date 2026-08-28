# CEGSP-P6-A：全层 centered/affine score-validity 实验报告

日期：2026-08-28  
云端：RTX 4090 24GB，`torch 2.5.1+cu124`，`bf16`  
原始结果：[result.json](../results/remote-runs/cegsp_p6a_score_validity_opt350m_20260828_42168/result.json)  
完整日志：[screen.log](../results/remote-runs/cegsp_p6a_score_validity_opt350m_20260828_42168/screen.log)

## 1. 实验目的

P6-A 不测试新的 CEGSP 模块，也不测试最终模型级 PPL 优势。它专门验证一个基础机制：在健康的 ordinary ternary initialization 上，量化点交叉熵梯度是否能够为**合法、同组、保持 support cardinality 的一次 relocation**提供有效排序信号；同时检查该信号是否同时存在于 centered 与 affine ternary 表示中。

候选分数固定为：

`score = - <G, Delta Q>`

其中 `G` 是 fit split 上 deployed ternary point 的 Q/K CE 梯度，`Delta Q` 是一次 support exchange 造成的实际权重变化。每个候选的 `mu/alpha` 或 scale 均冻结，不允许用 validation 或 untouched 数据反选候选。

## 2. 固定协议与完整性审计

| 项目 | 固定值 |
|---|---|
| 模型 | `facebook/opt-350m`，OPT decoder layers 0--23 |
| 投影 | 每层 Q/K；全模型共 24 层 |
| 数据 | Wikitext-2 train fit 8 batches；validation 8；disjoint untouched W2 16；C4 16 仅 baseline |
| 序列/批量 | `seq_len=128`，`batch_size=2`，offset=0 |
| group | group size 128 |
| centered | threshold factor 0.70 |
| affine | `Q = mu + alpha*T`，`T in {-1,0,+1}`，threshold factor 0.75，`mu/alpha` frozen |
| 候选 | 每个 layer/representation 32 个固定 gradient pool，取 rank `{0,1,2,4,8,16,24,31}` 的 8 个；另 8 个 matched-random |
| 总评估量 | 每种表示 192 gradient + 192 random = 384 rows；两种表示共 768 rows |
| 训练/优化 | 无 optimizer step、无 QAT teacher/checkpoint、无 PT²、无 module search |

完整性结果：两种表示均覆盖全部 24 层，均为 192 个 gradient 与 192 个 random 候选；每种表示的 384 行数值全部 finite；Q/K layout 与 group 内 cardinality 约束在脚本中保持不变。实验状态为 `complete`，耗时 930.9 秒。该结果不是由 untouched test 选择产生的。

## 3. 量化 baseline（仅用于说明测试难度）

| 初始化 | validation NLL | untouched W2 NLL | C4 NLL |
|---|---:|---:|---:|
| FP16 reference | 3.8039 | 3.8900 | 3.5628 |
| centered ternary | 8.6946 | 8.8477 | 8.1776 |
| affine ternary | 8.6915 | 8.8053 | 8.0545 |

两种 ordinary ternary initialization 都明显偏离 FP16，因此本轮存在足够的 residual error 可供候选排序机制检测。P6-A 不据此声称 centered 或 affine 是强 PTQ 基线；它只研究候选 score 的有效性。

## 4. 主要结果

表中 `Delta NLL < 0` 表示 relocation 改善对应 split。`rho` 是 score 与 `-Delta NLL` 的 Spearman 相关。

| 表示 | rho（validation） | rho（untouched W2） | gradient 平均 Delta（val） | top-20% 平均 Delta（val） | random 平均 Delta（val） | top-20% 改善率 / random |
|---|---:|---:|---:|---:|---:|---:|
| centered | +0.6982 | +0.7678 | -0.002436 | -0.005323 | -0.000044 | 1.000 / 0.510 |
| affine | +0.7199 | +0.7388 | -0.002431 | -0.005588 | +0.000286 | 1.000 / 0.271 |

untouched W2 的对应平均 Delta 为：centered 的 gradient `-0.002103`、top-20% `-0.004884`、random `+0.000049`；affine 的 gradient `-0.003078`、top-20% `-0.005924`、random `-0.000487`。每种表示的预注册 gate 均为 true：

1. validation 上 `rho > 0`；
2. top-20% 平均 validation Delta 优于全部 gradient candidates；
3. top-20% validation improvement rate 高于 matched-random。

固定 score bins 也呈单调趋势。以 affine 为例，5 个由高分到低分的 bin 的 validation 平均 Delta 分别为 `-0.005540, -0.003064, -0.001815, -0.001085, -0.000550`；centered 对应为 `-0.005235, -0.002593, -0.001859, -0.001555, -0.000859`，整体方向同样是高 score 候选更可能产生真实 held-out improvement。

## 5. 结果解释

### 发现 1：全层 score-validity gate 通过

centered 和 affine 都通过预先固定的 score-validity gate，而且不是只在少数幸运层成立。24 层全量候选的相关系数均约为 0.70--0.77，top-20% 候选的 validation improvement rate 均为 1.0。

这支持一个较窄但重要的机制结论：

> 在 ordinary centered/affine ternary state 中，量化点 CE 一阶分数可以作为一次合法 support relocation 的有效优先级信号。

### 发现 2：该信号不是简单的随机 relocation 效应

centered 的 random validation 平均 Delta 接近 0，affine 的 random validation 平均 Delta 为正；两者的 top-20% gradient 候选都显著更优。尤其 affine 中，top-20% 的 validation 平均改善约为 `-0.005588`，而 random 为 `+0.000286`。

因此，当前证据不只是“任意改变一个 ternary index 偶尔有帮助”，而是支持 quantized-point gradient 对候选排序具有独立信息量。

### 发现 3：affine 不是 P5-A 的局部特例

affine 在全 24 层的 score-validity gate 通过，说明 `mu/alpha` 固定时，CE gradient 仍能在 affine ternary index-space 中识别 task-relevant support relocation。这把 P5-A 的单层合法性与机制信号扩展成了全层机制证据。

### 发现 4：机制证据不等于模型级性能证据

本轮每个候选只做一次局部 relocation，没有把候选组合成最终 whole-model patch，也没有与 Strong PTQ 或 PT² 的最终 PPL 竞争。因此不能把 P6-A 写成“CEGSP 已优于强三值 PTQ”，也不能据此证明大模型 scaling。affine 与 centered baseline 的 NLL 仍很高，说明后续需要把“score validity”与“候选预算组合后的全局泛化”分开验证。

另外，affine random 在 untouched W2 上也多数为改善（improvement rate 0.8854），而 validation improvement rate 只有 0.2708，提示不同 split 存在分布偏移；这正是为什么本轮没有把单个 untouched 结果当作候选选择依据，也说明下一阶段不能只依赖单一校准流。

## 6. 对论文主张的更新

### 可以支持

- CEGSP 的量化点 CE score 在 centered 与 affine ternary index-space 都具有全层、固定候选协议下的排序有效性。
- 一次保持组内 support cardinality 的 relocation 可以通过 gradient score 被有效优先级化；该现象不是 matched-random control 能解释的。
- affine CEGSP 的机制不是某个单独 layer 的偶然现象。

### 尚不能支持

- CEGSP 已经优于 PT² 或其他最新 strong ternary PTQ。
- 全层一次性 patch 一定改善模型级 W2/C4 PPL。
- score validity 自动等价于 QAT gap closure 或大模型收益。

## 7. 下一步建议（本报告不自动启动）

最小且有判别力的后续是 P6-B：在不改变候选定义的前提下，用 3 个预注册 seed/offset 对 centered 与 affine 各重复一次 score-validity 汇总，重点报告 `rho`、top-20% 相对 random 的效应方向及跨 split 一致性。只有该机制在不同采样条件下仍稳定，才进入 P6-C 的固定小预算 whole-model patch；不建议根据本轮结果重新扫 threshold、sign rule、group size 或 layer。

如果 P6-B 通过，再做一个固定预算的组合实验：全 24 层统一候选池、按 score 全局排序、固定 top-k（不使用 untouched 选 k），并保留 matched-random whole-model control。若 P6-B 不稳定，则把论文主张收窄为“局部、表示依赖的 candidate-ranking mechanism”，而不是继续堆叠模块。
