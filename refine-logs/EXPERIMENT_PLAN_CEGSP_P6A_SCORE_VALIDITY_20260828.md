# P6-A：全层 centered/affine score-validity

## 状态

本方案在 2026-08-28 云端启动前固定。它扩展既有 P4 的 OPT-350M score-validity：P4 只测了少数指定层；P6-A 测量全部 24 个 decoder layers，并首次在相同协议下比较 centered 与 affine 两种 ternary initialization。它不是新的 CEGSP 模块搜索，也不使用 PT² checkpoint。

## 研究问题

在健康的 direct/ordinary ternary initialization 上，量化点 CE 一阶分数

`score = - <G, Delta Q>`

能否预测一个合法 support relocation 的真实 validation/untouched NLL 改变量？该预测关系是否依赖 centered 或 affine codebook？

## 冻结协议

- 模型：`facebook/opt-350m`。
- 设备：单张 RTX 4090 24 GiB。
- 数据：Wikitext-2 train fit 8 batches、validation 8 batches、validation untouched 16 batches；C4 validation 16 batches仅作 baseline/transfer 记录。
- `seq_len=128`，`batch_size=2`，所有 token offset 为 0，seed `20260828`，dtype `bf16`。
- 候选层：OPT decoder layers `0--23` 全部；每层只观察 Q/K projection。
- group size：128。
- centered 初始化：全模型 Linear 使用既有 direct ternary 规则，`threshold_factor=0.70`。
- affine 初始化：全模型 Linear 使用固定 `Q=mu+alpha*T`，`T∈{-1,0,+1}`，`threshold_factor=0.75`；`mu/alpha` 固定。
- 梯度：只用 fit split 的 1 个 batch，在对应 deployed ternary point 计算 Q/K CE gradient。
- 每个 layer/representation：先产生固定的 gradient candidate pool 32 个，再按固定 rank `{0,1,2,4,8,16,24,31}` 取最多 8 个做真实评估；另取 8 个合法 matched-random relocation。所有候选在评估前一次性生成，validation/untouched 不参与生成、排序或筛选。
- 每个候选只执行一次合法 support exchange：一个 active state 变为 zero，一个 zero state 变为 signed nonzero，保持 group 内 cardinality；不重新估计 scale/offset。
- 真实评估：对每个候选记录 validation NLL 与 untouched Wikitext-2 NLL 改变量；不以这些结果选择最终模型。记录 baseline C4 NLL，但不逐候选评估 C4。
- 无 QAT teacher、无 QAT checkpoint/logits、无 optimizer step、无 PT²、无 budget/sign/layer sweep、无 post-hoc epsilon。

## 预注册统计

对 gradient-ranked candidates 报告：Spearman(`score`, `-Delta NLL`)、top-20% 与全部候选的平均 `Delta NLL`/improvement rate、random matched improvement rate，以及按 score 降序的 5 个 fixed bins。对 validation 和 untouched 分别计算，不能用 untouched 反向改规则。

## Gate

每种 initialization 单独判定机制 gate：

1. `rho_val > 0`，且 score top-20% 的平均 validation delta 优于全部 gradient candidates；
2. top-20% 的 validation improvement rate 高于 matched-random rate；
3. untouched 统计作为 transfer check，不因为单个 representation 的 untouched 负结果否定另一 representation。

若 centered 与 affine 均通过，支持“score-validity 跨 representation”；若只有一个通过，报告 representation-dependent mechanism；若均不通过，关闭“当前一阶分数可泛化预测”的强表述，但不否定已经完成的 CEGSP 结果。

## 解释边界

本实验验证的是候选排序机制，不是最终 PPL SOTA gate。P4 的 3-layer score-validity 作为历史参考，不与本轮全层统计合并。任何后续方法性能实验都必须使用本轮预先固定的候选定义，而不能根据 score-bin 结果重新设计规则。
