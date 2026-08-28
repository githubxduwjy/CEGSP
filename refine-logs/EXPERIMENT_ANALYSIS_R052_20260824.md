# R052 分析：深中层 `(20,21)` 位置复制

## 完整性

- score_start=40, layers=(20,21), gate/test=8/8, epsilon=0。
- 4 候选×2 数据集×16 序列=128 行，sequence 为 40–55。
- 所有数值有限，nonfinite 总数=0。
- 用时 748.91 s，峰值显存 4063.27 MiB。

## 原始结果

gate 选择 `official`，因此 untouched test 与 official 的四项 delta 均为 0，属于预注册的 **SAFE FALLBACK**。

| candidate | gate objective | WikiText2 mean/CVaR delta | C4 mean/CVaR delta | eligible |
|---|---:|---:|---:|---:|
| hard_l20 | -0.012959 | -0.013115 / -0.037043 | +0.001831 / -0.003508 | no |
| hard_l21 | +0.047482 | +0.010867 / +0.046175 | +0.033722 / +0.099164 | no |
| hard_l20_l21 | +0.050083 | +0.012726 / +0.103478 | +0.011907 / +0.072222 | no |

`hard_l20` 很接近可用，但 C4 mean NLL 仍回归 +0.001831；冻结的零退化规则正确拒绝了它，不事后调 epsilon。

## 解读

R052 不否定 hard-T 的整体方向；它否定的是“R051 在 layer 11 的严格改善会自然复制到 layer 20/21”。当前证据更支持稀疏、位置依赖的可用层，不支持对所有中层统一开启 hard-T。按预注册进入最后的 `(30,31)` 边界复制，然后再综合深度图，不在 `(20,21)` 上调参。
