# R051 实验计划：中层 `(10,11)` 双分布门控

**Primary claim**：R047–R050 的 WikiText2/C4 分裂可能由早层误差的强传播敏感性导致；在中层窗口，local validation-gated hard-T 候选可能更稳定地保持双分布功能。

**Anti-claim**：hard-T 的 C4 改善只是普遍的数据集偏置，而非层位置互作用。

## 冻结设计

- 候选：`official`, `hard_l10`, `hard_l11`, `hard_l10_l11`。
- 仍保留 block-local 75/25 fit-validation hard-T 约束，只改 layer window。
- 评分数据为完全未见 windows 24–39：24–31 作 gate，32–39 作 untouched test。
- WikiText2/C4 mean NLL、CVaR10、nonfinite 的零退化规则不变。
- 模型、校准集、seed、blocksize 与 R050 一致。

## 成功准则

- 非 official 候选被 gate 选中，且 untouched W2/C4 mean/CVaR 全部 `<=0`，nonfinite 不增加。
- 若 gate 选择 official，记为安全回退，中层无改进证据。
- 若 gate 选中 hard-T 但 untouched 失败，记录分布/位置签名，下一轮移至 `(20,21)`，不在 `(10,11)` 调参。
