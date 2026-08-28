# CEGSP-03A：WikiText 选择到 C4 untouched 的跨数据迁移验证

## 固定研究方向

本实验不改变 `RESEARCH_DIRECTION_LOCK_CEGSP_20260826.md` 中冻结的主线：使用 deployed ternary weights 处的 CE gradient 做三值离散编辑。上一批实验只更新模块置信度，不改变核心问题。

## 问题

CEGSP-02A 已显示，在三个 WikiText offset 上，top-k CE-gradient 三值编辑对 untouched WikiText NLL 稳定改善；但这还不能排除“只拟合 WikiText 分布”的风险。CEGSP-03A 检查同一套由 WikiText fit/val 决定的编辑，是否能迁移到 C4 validation。

## 预注册设置

- 模型：`facebook/opt-350m`
- 设备：RTX 4090 24GB
- 量化：direct ternary PTQ，group size 128，threshold factor 0.7
- 编辑对象：24 层 attention Q/K
- 梯度：只在 deployed ternary model 上用 WikiText fit split 计算 CE gradient；不使用 QAT checkpoint、QAT logits、QAT latent weights 或 optimizer update
- 选择：只允许使用 WikiText validation 的 single-layer delta 排序
- 测试：
  - `untouched_w`：未参与选择的 WikiText validation 后续片段
  - `untouched_c4`：C4 validation streaming 前 8 batches，仅报告迁移，不参与选择或调参
- k：只测试 CEGSP-02A 预先确定的稳定区域 `k ∈ {4, 6}`，不做事后扩展
- 方法族：
  - support top-k
  - signflip top-k
  - joint top-k

## Gate

Primary transfer gate：

1. `ksweep-joint-top4-qk` 或 `ksweep-joint-top6-qk` 在 `untouched_w` 与 `untouched_c4` 上的 NLL delta 均 `<= 0`；
2. 同一候选的 `val` delta 也 `<= 0`；
3. runtime 保持在 4090 可承受的 PTQ-diagnostic 级别，本实验不要求端到端 PPL。

Secondary interpretation：

- 若 WikiText 改善但 C4 退化：说明 CEGSP 当前选择规则有分布过拟合风险，下一步需要加入跨分布 holdout 或梯度正则，而不是改主线。
- 若 support/signflip 单独迁移而 joint 不迁移：说明“joint 逐层择优”可能过拟合 val，需要引入更保守的选择规则。
- 若二者均迁移：下一步进入更大样本/第二 seed 或更大模型的确认。

## 禁止项

- 不使用 QAT teacher。
- 不根据 C4 结果重新选 k、调 threshold、调 max-edits。
- 不把 C4 缺失时的 fallback 文本当作 C4。
