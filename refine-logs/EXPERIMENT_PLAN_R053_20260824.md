# R053 实验计划：`(30,31)` 末层边界复制

**Primary claim**：hard-T 的可传递性具有深度依赖；在完成 `(10,11)` 成功与 `(20,21)` safe fallback 后，末层窗口可判别改善是否只存在于特定中层。

**Anti-claim**：R051 只是 layer 11 或其评分窗口的偶然。

## 冻结设计

- 候选：`official`, `hard_l30`, `hard_l31`, `hard_l30_l31`。
- 全新 windows 56–71：56–63 gate，64–71 untouched test。
- WikiText2/C4 mean NLL、CVaR10、nonfinite 零退化 gate 完全不变。
- 校准、seed、blocksize、local 75/25 validation 与 R051/R052 一致。

## 决策

- **PASS**：非 official 被选中且 untouched 四指标全非退化。
- **SAFE FALLBACK**：选择 official；记录末层无可验证改善。
- **FAIL**：非 official 被选中但 untouched 退化。
- 三种结果都不在本窗口调 epsilon 或枚举 projection mask。R053 后结束预注册的三窗口位置复制，转入深度机理综合。
