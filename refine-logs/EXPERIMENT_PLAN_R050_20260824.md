# R050 实验计划：全新样本上的 8+8 双分布门控

**Primary claim**：R048/R049 的两折选择不一致主要是 4-window 方差；增加样本量后，零退化门控能稳定选择一个在 untouched W2/C4 同时非退化的候选。

**Anti-claim**：改善来自重用 R047/R048 的已见窗口。

## 冻结设计

- 候选不变：`official`, `hard_l0`, `hard_l1`, `hard_l0_l1`。
- 校准不变：WikiText2 nsamples=8, seed=0, local fit/validation=75/25。
- 评分从 sequence window 8 开始，不重用 R047/R048 的 0–7。
- gate：windows 8–15，WikiText2/C4 各 8 个。
- untouched test：windows 16–23，WikiText2/C4 各 8 个。
- epsilon 仍为 0；两分布 mean NLL/CVaR10 均非退化，且无新 nonfinite，才可选中。

## 成功/失败决策

- **PASS**：非 official 候选被选中，且 untouched 四个功能指标均 `<=0`。下一步扩展到后续层窗。
- **SAFE FALLBACK**：gate 选择 official。关闭 `(0,1)` 的改进声明，保留门控安全性。
- **FAIL**：选中非 official 但 untouched 任一指标退化。关闭当前 0/1 层 gate 形式，不再扩充这条样本量路线。
