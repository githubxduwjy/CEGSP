# P5-C0 实验报告：官方 PT² 数值健康与评测一致性审计

日期：2026-08-28  
模型：`facebook/opt-350m`  
设备：NVIDIA RTX 4090 24 GiB  
Run：`cegsp_p5c0_pt2_numeric_health_opt350m_20260828_42143`

## 1. 结论先行

本轮结果为：

> **`P5-C0_NUMERICAL_HEALTH_FAIL`：官方 PT² 状态 finite 且结构合法，但数值健康性不通过，不应继续作为当前论文的主 strong-PTQ baseline。**

这不是 CEGSP 的新失败实验，也不是评测器单独造成的异常。P5-C0 没有运行 CEGSP、没有使用 QAT checkpoint/logits、没有搜索任何 CEGSP budget/sign/layer rule；它只按官方 PT² 配置运行 ATQ 与 ATQ+SSR，并对同一个量化状态同时使用官方 evaluator 和 compact evaluator。

## 2. 固定协议

- 官方校准：Wikitext-2 train，`nsamples=128`，`calib_seqlen=2048`，seed `0`。
- 官方 ATQ：`method=atq`，`percdamp=0.01`，`group/block size=128`，`num_p=1`，`salient_metric=hessian`，GPTQ enabled。
- 量化顺序：`k_proj -> v_proj -> q_proj -> out_proj -> fc1 -> fc2`。
- 模型权重 dtype：`torch.float16`。
- 两个固定状态：`ATQ (ssr=False)` 和官方 `ATQ+SSR (ssr=True)`。
- 官方评测：Wikitext-2/C4，sequence length 2048。
- compact 评测：固定 Wikitext validation/untouched 与 C4 validation batches，sequence length 128；只用于方向/数值一致性，不与官方 PPL 数值直接等同。

## 3. 运行完整性

第一次启动因同步辅助模块缺少 C4 loader 接口退出；第二次启动发现官方 evaluator 迁移模型后 compact evaluator 的设备状态未复位。两次均发生在量化结果产生前，未产生可用指标。只修复了这两个明确的 harness 问题，保留同一 run-id、模型和预注册配置；最终运行完整结束并写出原始结果。

最终审计记录：

| 检查项 | 结果 |
|---|---:|
| 预期线性模块 | 144 = 24 layers × 6 modules |
| 实际线性模块 | 144 |
| 实际量化 block | 1728 |
| 记录值 finite | 是 |
| inferred T 非法比例 | 0 |
| inferred T nonfinite block | 0 |
| 官方/compact 指标 finite | 是 |
| CEGSP/QAT 是否调用 | 否 |

因此这是一个**执行完成、结构合法、数值健康失败**的结果，而不是 harness failure。

## 4. 评测结果

### 4.1 Clean FP16 reference

| 状态 | Official W2 PPL | Official C4 PPL | Compact W2 untouched NLL | Compact C4 NLL |
|---|---:|---:|---:|---:|
| Clean FP16 | 22.0046 | 22.5898 | 3.8903 | 3.5622 |

### 4.2 官方 ATQ

| 状态 | Official W2 PPL | Official C4 PPL | Compact W2 untouched NLL/PPL | Compact C4 NLL/PPL | Quant time |
|---|---:|---:|---:|---:|---:|
| ATQ, SSR off | 13044.43 | 11384.02 | 9.5009 / 13372.08 | 9.0396 / 8430.31 | 87.8 s |
| ATQ, SSR on | 15917.38 | 13408.74 | 9.8585 / 19120.27 | 9.3881 / 11945.14 | 104.2 s |

官方和 compact evaluator 的绝对数值不同，这是因为数据切片和序列长度不同；但二者都相对 clean FP16 同向恶化。因此异常不能归结为 compact evaluator 单独造成的。

## 5. 数值健康诊断

### 5.1 全局诊断

| 指标 | ATQ | ATQ+SSR |
|---|---:|---:|
| 全局 `max|Q|` | 1166.0 | 11696.0 |
| block `p99|Q|` 中位数 | 0.05078 | 0.05063 |
| block `p99|Q|` 最大值 | 998.72 | 11118.49 |
| 最大/中位数 p99 | 1.97×10⁴ | 2.20×10⁵ |
| 最大 output reconstruction MSE | 85807.74 | 4149223.25 |
| 平均 output reconstruction MSE | 957.75 | 43191.03 |

预注册的 10× 相对 outlier 诊断在两个状态均触发。注意：10× 不是为了筛掉结果而事后设置的性能阈值，而是本轮启动前固定的数值诊断标记；这里同时报告原始最大值和分布中位数。

### 5.2 异常位置

异常高度局部化在 layer 0 的 attention projection：

| 状态 | layer-0 Q max | layer-0 K max | layer-0 V max | 代表性正常层最大值 |
|---|---:|---:|---:|---:|
| ATQ | 1166.0 | 1152.0 | 289.75 | layer-1 fc2 = 2.10 |
| ATQ+SSR | 11696.0 | 7844.0 | 482.75 | layer-1 fc2 = 0.34 |

输出重构误差也集中在 layer 0：

- ATQ：layer-0 K MSE `85807.74`，Q MSE `48932.10`，V MSE `3063.88`。
- ATQ+SSR：layer-0 Q MSE `4149223.25`，K MSE `2066358.75`，V MSE `3836.03`。

这说明问题不是所有层均匀变差，而是官方 pipeline 中早期 Q/K 的局部数值失真被放大；SSR 状态比不启用 SSR 更严重。

### 5.3 三值结构本身仍然合法

从量化器内部更新得到的 T 统计中，`-1/0/+1` 均为合法状态。例如 ATQ layer-0 K 的代表性 block 约为 `27.6% / 44.8% / 27.6%`，ATQ+SSR layer-0 Q 的代表性 block 约为 `25.4% / 49.4% / 25.3%`。因此本轮揭示的是：

> **“T 合法、codebook 可解释”并不等于“部署状态数值健康”。**

CEGSP 的 state-parity gate 因此仍然成立，但不能把 state legality 当作 PT² baseline quality 的充分证据。

## 6. 对 P5-C 的重新解释

P5-C 原先得到的 `STATE_PARITY_PASS / PT2_COMPATIBILITY_FAIL` 仍然有效，但现在可以进一步拆开：

1. **接口层面通过**：CEGSP 可以合法读取官方 PT² 的 affine ternary state，group/layout/T legality 均无问题。
2. **性能层面失败**：在这个 PT² state 上，CEGSP 没有改善 baseline。
3. **基线健康层面失败**：P5-C0 复现了同一类 layer-0 Q/K 数值异常，并且官方/compact evaluator 都观察到极高的模型损失。

因此不能把 P5-C 的负结果表述成“CEGSP 在健康 strong PTQ 上无效”；当前证据只支持：

> **在本次官方 OPT-350M PT² 配置产生的、数值不健康的状态上，CEGSP 没有带来收益；CEGSP 与该 PT² 状态的 performance compatibility 不成立。**

## 7. 研究路线决策

按照预注册分支，当前进入路线 B：

- 停止把当前 PT² reproduction 放在论文主表的 strong baseline 位置。
- 不再扫描 CEGSP budget、sign rule、layer subset 或 epsilon 来挽救 P5-C。
- P5-C 保留为“state compatibility passed, performance compatibility failed”的 negative/limitation evidence。
- PT² 的官方复现结果与 layer-0 异常放入 appendix/reproduction limitation，并明确报告官方配置、实现版本和数值诊断。
- 下一阶段若仍需要 strong-baseline compatibility，应先寻找一个稳定且能导出 `mu/alpha/T/group layout` 的 ternary PTQ，并先做同样的 protocol/health audit；在此之前不做新的 CEGSP 兼容性实验。

当前主张应收窄为：

> **CEGSP is a task-aware discrete refinement method for direct/ordinary affine ternary PTQ initializations.**

不应声称：

> **CEGSP consistently improves an optimized strong ternary PTQ pipeline.**

## 8. 局限

本轮只审计了 OPT-350M、seed 0 和官方 PT² 版本，不能据此断言所有 PT² 实现或所有模型都会出现同样问题。审计已经定位到稳定、可重复的 layer-0 Q/K 数值异常，但尚未进一步隔离其内部来源（例如 GPTAQ compensation、SSR reorder 或官方实现的特定交互）。这正是选择替代 strong ternary PTQ 之前需要记录的 reproduction limitation，而不是继续对 CEGSP 做后验调参的理由。

## 9. 原始产物

- [P5-C0 result.json](/home/x1shan/文档/ChatGPT/PTQ_paper/results/remote-runs/cegsp_p5c0_pt2_numeric_health_opt350m_20260828_42143/result.json)
- [P5-C0 screen.log](/home/x1shan/文档/ChatGPT/PTQ_paper/results/remote-runs/cegsp_p5c0_pt2_numeric_health_opt350m_20260828_42143/screen.log)
- [P5-C0 preregistration](/home/x1shan/文档/ChatGPT/PTQ_paper/refine-logs/EXPERIMENT_PLAN_CEGSP_P5C0_PT2_NUMERICAL_HEALTH_20260828.md)
