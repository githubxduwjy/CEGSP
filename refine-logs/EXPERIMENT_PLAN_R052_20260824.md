# R052 实验计划：`(20,21)` 层窗口复制

**Primary claim**：R051 的双分布 PASS 不是 layer 11 单点偶然；在更深的相邻层窗口，选择性 local hard-T 候选仍可能被冻结 gate 稳定识别。

**Anti-claim**：R051 的改善来自特定评分窗口或层 11 偶然。

## 冻结设计

- 候选：`official`, `hard_l20`, `hard_l21`, `hard_l20_l21`。
- 全新 windows 40–55：40–47 gate，48–55 untouched test。
- WikiText2/C4 mean NLL、CVaR10、nonfinite 零退化 gate 完全不变。
- 校准、seed、blocksize、local 75/25 validation 与 R051 一致。

## 决策

- **PASS**：非 official 被选中且 untouched 四指标全非退化；再进入最后 `(30,31)` 边界复制。
- **SAFE FALLBACK**：选择 official；记录该深度无改进证据，仍执行 `(30,31)` 以完成位置对照。
- **FAIL**：非 official 选中但 untouched 退化；不在 `(20,21)` 调参，进入 `(30,31)` 完成最后位置对照。
