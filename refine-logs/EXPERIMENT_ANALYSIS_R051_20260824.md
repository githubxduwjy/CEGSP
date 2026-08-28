# R051 分析：中层 `(10,11)` 位置泛化

## 完整性

- score_start=24, layers=(10,11), gate/test=8/8, epsilon=0。
- 4 候选×2 数据集×16 序列=128 行，sequence 为 24–39。
- nonfinite 总数=0。
- 用时 750.56 s，峰值显存 4063.27 MiB。

## 原始结果

gate 选中 `hard_l11`：

| split/dataset | mean NLL delta | CVaR10 delta | pass / wins |
|---|---:|---:|---:|
| gate WikiText2 | -0.024505 | -0.084505 | pass |
| gate C4 | -0.019011 | -0.100810 | pass |
| untouched WikiText2 | -0.036275 | -0.168679 | 5/8; 6/8 |
| untouched C4 | -0.049213 | -0.191124 | 6/8; 7/8 |

`hard_l10` 在 gate 上因 W2 CVaR 和 C4 mean/CVaR 退化被拒；`hard_l10_l11` 因 C4 CVaR 退化被拒。这说明门控不只是在 hard-T 与 official 之间选择，还能拒绝“相邻层全开”的过度更新。

## 解读

R051 严格 PASS。与 `(0,1)` 多次在 untouched W2 失败相比，`hard_l11` 在两分布的 mean 和 tail risk 上均泛化，支持“hard-T 收益具有层位置依赖性”，不支持对所有层统一开启。

但当前只是单个中层窗口。R052 将冻结设计复制到 `(20,21)` 和全新窗口，检验这是否为可复制的深度结构，而非 layer 11 单点偶然。
