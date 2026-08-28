# R050 分析：8+8 全新样本门控

## 完整性

- config: score_start=8, gate=8, test=8, layers=(0,1), epsilon=0。
- 4 个候选×2 数据集×16 序列=128 行，sequence 严格为 8–23。
- nonfinite 总数=0。
- 用时 746.86 s，峰值显存 4063.27 MiB。

## 原始结果

gate 选中 `hard_l1`：

| split/dataset | mean NLL delta | CVaR10 delta | gate pass / wins |
|---|---:|---:|---:|
| gate WikiText2 | -0.028932 | -0.128973 | pass |
| gate C4 | -0.066620 | -0.201311 | pass |
| untouched WikiText2 | +0.032156 | +0.069191 | 2/8; 3/8 |
| untouched C4 | -0.106941 | -0.353334 | 7/8; 7/8 |

## 解读与决策

R050 为 FAIL。增加到 8 个 gate 样本仍无法让 `(0,1)` 层候选在 untouched WikiText2 上泛化，且 mean 与 CVaR 同时变差。因此不再扩大该窗口样本量、不调 epsilon，也不将 C4 大幅改善解读为通用胜利。

但 R042c 的 block-local 机制改善仍成立，所以不否定 hard-T 整体方向。R051 转向层窗口 `(10,11)`，检验早层分布分裂是否具有位置特异性。
