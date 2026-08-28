# Review Summary

**Problem**: 三值 hard-`T` 局部改善无法稳定转化为全模型函数改善。  
**Rounds**: 2  
**Final verdict**: REVISE — 可进入机理实验，不可直接宣称方法成立。

| Round | Main concern | Resolution | Remaining risk |
|---|---|---|---|
| 1 | Fisher + CVaR gate 像工程 wrapper | 删除 Fisher，聚焦 per-layer no-cancellation constraint | 相消机制尚未被实证 |
| 2 | initializer、cancellation、fair budget 定义不精确 | 固定 PT² local initializer，定义 `C_S`，匹配 proposals/steps/tokens | 零退化约束可能 safe-but-vacuous |

最终方案保留一个主贡献：在每层局部映射不退化的可行域内做 hard ternary 跨层联合优化。R054 是强制的机理 gate；若其失败，不进入全模型方法。
