# R048/R049 分析：双分布门控的 untouched 泛化与 cross-fit

## R048 原始结果

- 选择集选中：`hard_l1`。
- 总用时：712.50 s；峰值显存：4063.27 MiB；所有候选 nonfinite delta 为 0。

| untouched test | mean NLL delta | CVaR10 delta | wins |
|---|---:|---:|---:|
| WikiText2 | +0.004538 | -0.048344 | 2/4 mean; 2/4 CVaR |
| C4 | -0.111372 | -0.331174 | 4/4 mean; 4/4 CVaR |

R048 严格 gate 失败：WikiText2 mean NLL 有小幅正退化。这否定的是“4-window 双分布 gate 足以泛化”，不是 hard-T 候选生成机制。

## R049 交换折 cross-fit（无需重复 GPU 量化）

将 R048 后 4 个窗口改为 gate，按同一零退化规则重新选择：

| candidate | W2 mean | W2 CVaR | C4 mean | C4 CVaR | eligible |
|---|---:|---:|---:|---:|---|
| `hard_l0` | +0.016567 | +0.099493 | -0.107785 | -0.197554 | no |
| `hard_l1` | +0.004538 | -0.048344 | -0.111372 | -0.331174 | no |
| `hard_l0_l1` | -0.026517 | +0.104399 | -0.183353 | -0.371631 | no |

交换折选择 `official`。R048 选择 `hard_l1`，R049 选择 `official`，候选身份不稳定。数据显示 W2 是决定性约束，C4 改善对三个 hard-T 候选都更一致。

## Auto Research 决策

不调整 epsilon，不使用 test 事后重加权。R050 使用完全未见的窗口 8–23：8 个 gate + 8 个 untouched test，检验样本量增加是否能稳定选择。若仍失败，关闭 `(0,1)` 当前 gate 设计并转向其他层窗或新的机制约束。
