# 1.58-bit PTQ 实验总结报告

日期：2026-08-22  
模型：LLaMA-2-7B  保护位宽：三值权重（1.58-bit PTQ）  
主要基线：官方 PT²/ATQ，无 SSR，除特别说明外使用 8 个校准样本、2048 序列长度、128 block size。

## 一、今日研究问题

今天集中验证一个核心假设：能否借鉴 PT²/ATQ 的离散结构设计，在量化过程中更新三值分配 `T`，并通过验证集门控保留真正有利于模型函数保持的更新。

今天没有把某一个失败实验直接解释为“硬 T 更新方向错误”，而是分成三层检查：

1. 块级未见激活上的机制是否成立；
2. 机制能否迁移到真实 GPTQ 量化；
3. 迁移失败究竟来自局部目标、量化上下文，还是数据分布依赖。

## 二、原始结果汇总

| 实验 | 对照与变量 | 主要结果 | 判定 |
|---|---|---|---|
| R042b | 固定 `T`、无门控更新、验证门控更新；56 blocks | 门控相对固定 `T` 的未见测试 NMSE 中位改善 5.6872%，胜率 96.43%；但未优于无门控更新 | 机制门控失败 |
| R042c | 公平 matched 对照：fit-only 更新后均 refit `alpha/mu` | 门控相对固定 `T` 中位改善 5.8289%、均值改善 12.1401%、胜率 96.43%；matched ungated 中位改善 5.0223%、胜率 87.50% | 机制门控通过 |
| R043a | 在 layer 0/10/20/31 的官方 GPTQ 中更新 `T` | WikiText2 为 NaN；C4=56.0225，相比官方 66.7370 改善 16.06%；1112 blocks，fallback=0 | 模型迁移失败 |
| R044a | 仅 layer 0 更新 `T`，其余保持官方 GPTQ | W2=26.2403、C4=57.9333；相对官方 W2=25.8104、C4=66.7370，W2 恶化 1.67%，C4 改善 13.19% | 局部迁移门控失败 |
| R045 | layer-0 fixed `T` vs hard `T`，其余模型保持 FP16 | hard `T` 在 4 个留出序列的 layer NMSE、cosine drift、mean NLL、CVaR10 均优于 fixed `T` | 说明局部候选本身并非简单有害 |
| R046 | 同一候选放入完整量化上下文，分别评估 W2/C4 | W2：hard `T` 的 mean NLL 增加 +0.0479、CVaR10 +0.1644；C4：mean NLL -0.1143、CVaR10 -0.1328 | 分布选择性，严格 gate 失败 |

注：R042c 的 matched ungated-fit-refit 结果为中位改善 5.0223%、胜率 87.50%，因此门控版本在该公平对照下胜率更高；报告不把 R042c 解读为已经证明模型级收益。

## 三、关键发现

### 1. 离散 `T` 更新在局部函数保持层面确实有信号

R042c 在 56 个 block、fit/validation/untouched-test 分离的设置下，验证门控更新能稳定降低未见激活上的加权输出误差：中位改善 5.8289%，96.43% 的 block 获胜。这说明研究方向不是空想，也不是单纯依赖校准集拟合。

更准确的表述是：验证门控的硬三值结构更新具备“局部机制有效性”，但这不是“模型级 PPL 已经提升”的证据。

### 2. 局部 NMSE 的改善不能直接推出端到端 PPL 改善

R043a 和 R044a 表明，`T` 更新进入逐层 GPTQ 后，误差会沿后续量化层传播。R044a 即使只更新 layer 0，也出现 W2/C4 方向相反：W2 恶化 1.67%，C4 改善 13.19%。因此当前 block-level objective 还没有充分表达“量化后模型轨迹”的风险。

这不是硬 `T` 更新被整体否定，而是说明接受准则的层级不够高：需要考虑量化上下文和跨分布稳定性。

### 3. R045 排除了一个过早结论：候选本身并非在所有函数指标上都更差

在 FP16 其余层的隔离环境中，R045 的 hard `T` 候选在四个留出序列上同时降低了 layer-output NMSE、cosine drift、平均 NLL 和尾部 token 的 CVaR10 NLL。由此可排除“只要更新 `T` 就会局部破坏模型”的解释。

但该环境没有复现后续层量化误差，所以不能作为 R044 的反证；它只说明 R044 的问题产生于“候选 + 后续量化上下文”的组合。

### 4. R046 揭示了当前最重要的研究对象：分布鲁棒性

将同一 layer-0 hard `T` 候选放回完整量化模型后，W2 和 C4 的方向相反：

| 数据流 | mean NLL 相对官方 | CVaR10 NLL 增量相对官方 | 解释 |
|---|---:|---:|---|
| WikiText2 | +0.0479 | +0.1644 | hard `T` 变差 |
| C4 | -0.1143 | -0.1328 | hard `T` 变好 |

因此，候选的价值不是单一标量“好/坏”，而是依赖校准数据和测试分布。下一版方法必须把多分布函数保持作为接受条件，否则很容易在 C4 上获得收益、同时牺牲 W2。

## 四、今天可以写入研究结论的边界

目前可以支持的结论：

- 三值 `T` 的验证门控离散更新，在 block-level、未见激活测试上具有稳定收益；
- 该收益不能直接迁移为模型级 PPL 收益；
- 迁移失败不是由 grid fallback 或明显数值异常主导，R043a 的 1112 个 refined blocks 均无 fallback，R044a 也通过了针对退化行的稳定性检查；
- hard `T` 候选具有明显的数据分布选择性，完整量化上下文中的跨分布鲁棒性是下一步核心问题。

目前不能支持的结论：

- 不能说硬 `T` 更新已经优于 PT²/ATQ；
- 不能说 layer 0 是普适的最佳更新位置；
- 不能把 C4 的改善外推到 WikiText2 或通用任务；
- 不能继续扩大到全层 hard `T` 搜索，因为当前接受准则尚未解决 W2/C4 冲突。

## 五、对研究主线的重新定位

今天的结果把原先较宽泛的“函数感知三值 PTQ”收敛成一个更可检验的核心命题：

> 三值结构 `T` 不应只由单个 block 的重构误差决定，而应被视为量化模型轨迹中的离散控制变量；候选更新需要在多个校准分布和量化上下文下通过鲁棒验证。

对应的方法雏形可以称为“robust trajectory-gated ternarization”：

1. 用 Hessian/激活加权目标生成少量、稀疏的 `T` 候选；
2. 在短序列和多个数据流上评估候选造成的量化后轨迹变化；
3. 采用 minimax 或 Pareto 接受规则，只接受不会显著伤害任一主要分布的候选；
4. 对不稳定候选回退到 fixed `T`，而不是强制更新。

这比继续堆叠坐标变换、SSR 或全层枚举更接近今天实验真正暴露的瓶颈。

## 六、下一步实验建议

下一轮建议做 R047，优先使用现有 R046 候选验证接受规则，而不是扩大搜索空间：

1. 预注册跨分布 gate：W2-like 与 C4-like 留出流同时评估 mean NLL、NLL increase、CVaR10；
2. 先做离线 dry-run，确认该 gate 能拒绝 R046 这种“C4 改善但 W2 恶化”的候选；
3. 再生成更小步长或更稀疏的 `T` proposal，例如限制 changed fraction、按 block/tranche 接受；
4. 只有当候选在两类分布上都不恶化，才进入 layer 0 的完整量化测试；
5. 若 R047 找不到同时不伤害两类分布的候选，再研究“分布条件化的 `T`”或报告方法的 Pareto trade-off，而不是继续无条件扩大到全层。

## 七、产物索引

- 实验总表：[EXPERIMENT_TRACKER.md](/home/x1shan/文档/ChatGPT/PTQ_paper/refine-logs/EXPERIMENT_TRACKER.md)
- 研究发现：[findings.md](/home/x1shan/文档/ChatGPT/PTQ_paper/findings.md)
- R042c 原始摘要：[summary.json](/home/x1shan/文档/ChatGPT/PTQ_paper/results/remote-runs/hessian_gated_r042c_ns12_20260822/summary.json)
- R043a 原始结果：[selective_gated_t_gptq.json](/home/x1shan/文档/ChatGPT/PTQ_paper/results/remote-runs/gated_t_r043a_l0_10_20_31_ns8_20260822_retry/selective_gated_t_gptq.json)
- R044a 原始结果：[selective_gated_t_gptq.json](/home/x1shan/文档/ChatGPT/PTQ_paper/results/remote-runs/gated_t_r044a_l0_ns8_20260822/selective_gated_t_gptq.json)
- R045 原始结果：[metrics.json](/home/x1shan/文档/ChatGPT/PTQ_paper/results/remote-runs/r045_trajectory_gate_20260822/metrics.json)
- R046 原始结果：[metrics.json](/home/x1shan/文档/ChatGPT/PTQ_paper/results/remote-runs/r046_contextual_sequence_gate_20260822/metrics.json)
