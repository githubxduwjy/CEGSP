# CEGSP-P5-B：整体 affine CEGSP compatibility test

日期：2026-08-28

## 研究问题

P5-A 只在 OPT-350M 的第 13 层验证了：把 CEGSP 从 centered ternary codebook 提升到 affine ternary index space 后，support relocation 可以保持合法，并在冻结 `mu/alpha` 时产生独立于随机扰动的局部机制信号。

P5-B 不再把某一层作为主实验单位，而是检验：

> 在所有候选 attention Q/K 层上计算一次量化点 CE 梯度，按照预先冻结的 layer-ranking 和 budget rule 选择层与 relocation，并一次性联合应用，整体 affine CEGSP 是否仍能改善 stronger affine ternary initialization。

本轮仍不是 PT² checkpoint 的最终兼容性实验；PT² 的可复现 baseline/state-export protocol 仍单独处理。

## 预注册协议

- 模型：`facebook/opt-350m`，OPT decoder layers 0--23 全部作为候选层。
- 目标模块：每个候选层的 Q/K projection；不改 V/O。
- codebook：每个 row-group 一个 affine ternary codebook：`Q = mu + alpha*T`，`T in {-1, 0, +1}`。
- group size：128。
- threshold factor：0.75，沿用 P5-A 的 PT²-style affine initialization。
- `mu` 和 `alpha`：从 FP 权重初始化后冻结；relocation 后不重估。
- 校准/评估：Wikitext-2 fit 8 batches、validation 8 batches、untouched Wikitext-2 8 batches、untouched C4 8 batches；sequence length 128、batch size 2、offset 全为 0。
- 梯度：仅使用 fit split 的 1 个 batch 计算 deployed affine ternary point 的 CE gradient；不使用 validation 或 untouched 数据选层、选预算或选 relocation。
- 主预算：`top-6 layers`，每个入选层 64 个 relocation pairs，共预期 384 pairs / 768 changed coordinates。
- 次预算：`top-4 layers`，每个入选层 64 个 relocation pairs，共预期 256 pairs / 512 changed coordinates。该预算用于观察预算敏感性，不根据结果改主预算。
- layer score：每层 Q/K 候选按一阶 CE gain 降序排列，取前 8 个合法 candidate 的 score 之和；按该分数降序选 top-6 或 top-4，平分时按 layer index 升序。
- layer 内 relocation：在已选层中，按同一 CE score 选前 64 个互不冲突的合法 candidates。
- random control：固定同一 selected layer set 和每层相同的 64 个 relocation 数，只随机 donor、receiver 和 receiver sign；不重新选层。
- 无 QAT teacher、无 latent FP optimizer、无多步训练、无后验 threshold/epsilon/budget 调整。

## 比较组

1. affine ternary baseline：所有候选 Q/K 都使用冻结的 affine codebook，不做 relocation。
2. affine CEGSP top-6：预注册 primary。
3. affine CEGSP top-4：预注册 secondary。
4. random affine relocation top-6 / top-4：严格匹配 selected layers、每层 relocation 数和合法 codebook。

## Gate

### Gate A：合法性

所有 variant 必须满足：

- 所有有效位置仍落在 `{mu-alpha, mu, mu+alpha}`；
- padding 位置保持 zero state；
- 每个 row-group 的 active support cardinality 不改变；
- changed coordinates 等于预注册数量；
- 所有评估 NLL 为 finite。

### Gate B：primary整体改善

top-6 必须同时满足：

`L_val(CEGSP_6) < L_val(affine baseline)`

并且

`L_W2(CEGSP_6) < L_W2(affine baseline)`

以及

`L_W2(CEGSP_6) < L_W2(random_6)`。

满足时，P5-B 记为 `PASS_OVERALL_AFFINE_COMPATIBILITY`。top-4 作为预算稳定性证据单独记录；C4 是 transfer 指标，不作为硬 gate。

若 top-6 失败而 top-4 通过，记录为 `SMALL_BUDGET_ONLY`，不事后把 top-4 改写成 primary。

## 结果解释边界

- 通过：说明 affine extension 不只在单层可行，而且可在预注册的全候选空间、固定整体预算和 matched random control 下运行。
- 失败但合法：说明 affine codebook 与整体 relocation 的兼容性对预算或层选择敏感，需要诊断 layer-score 分布；不能据此否定 CEGSP 的 centered 结果或 P5-A feasibility。
- 失败且不合法/非 finite：优先判定为实现或 harness 故障；只有明确故障才允许修复后重跑同一协议。
- 本轮不声称优于 PT² 或其他 strong ternary PTQ，也不把 C4 单独的正结果升级为通用结论。
