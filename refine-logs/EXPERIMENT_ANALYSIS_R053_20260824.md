# R053 分析：末层 `(30,31)` 边界复制

## 完整性

- score_start=56, layers=(30,31), gate/test=8/8, epsilon=0。
- 4 候选×2 数据集×16 序列=128 行，sequence 为 56–71。
- 所有浮点指标有限，nonfinite 总数=0。
- 用时 747.50 s，峰值显存 4063.27 MiB。

## Gate 结果

gate 选择 `official`，属于预注册的 SAFE FALLBACK。

| candidate | objective | W2 mean/CVaR | C4 mean/CVaR | rejection reason |
|---|---:|---:|---:|---|
| hard_l30 | -0.006752 | -0.000864 / +0.005778 | +0.002548 / -0.034470 | W2 CVaR, C4 mean |
| hard_l31 | -0.025262 | +0.004858 / -0.046726 | +0.003325 / -0.062504 | W2 mean, C4 mean |
| hard_l30_l31 | -0.023558 | -0.005166 / -0.041405 | +0.007918 / -0.055578 | C4 mean |

## 解读

`hard_l30_l31` 的平均 objective 是正向的，但 C4 mean 仍退化 +0.007918，冻结 Pareto gate 正确拒绝。结合 R050–R052，四个深度窗口中只有 layer 11 严格通过；不再扫描其他层。下一步转向 R054，用全部四个已冻结窗口评估误差相消机制。
