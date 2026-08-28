# R054 结果：error-cancellation 风险假设失败

## 完整性

- 四个冻结窗口 `(0,1)`, `(10,11)`, `(20,21)`, `(30,31)` 全部完成。
- 每个窗口包含 W2/C4 各 16 条新序列（72--87）；合计 128 条机制记录。
- 配置一致：nsamples=8, blocksize=128, validation=25%, max_steps=4, seed=0，epsilon 固定为 0。
- 所有浮点指标有限，`nonfinite_delta_total=0`。
- 本地重新运行分析器与远端 `summary.json` 字节级一致，SHA-256 为 `da6034135453da1fa45a51ad30b639e2790c468982678d8b820a802ab8c833a1`。
- 独立跨模型审计后端本轮不可用；以上为机械完整性审计，不声称外部审计通过。

## 预注册 Gate

| split | mechanism cases | risk rho: mean/CVaR | boundary rho: mean/CVaR | risk avg | boundary avg | pass |
|---|---:|---:|---:|---:|---:|---|
| gate | 47/64 | 0.5531 / 0.2501 | 0.5901 / 0.2833 | 0.4016 | 0.4367 | FAIL |
| test | 48/64 | 0.5783 / 0.1655 | 0.6388 / 0.2243 | 0.3719 | 0.4316 | FAIL |

机制样本数量充足，但取消项风险没有比简单 boundary-NMSE proxy 高 0.05；相反，两 split 的平均相关都更低，且 test CVaR rho 未达到 0.20。因此按预注册判定为 `FAIL`。

## 解释边界

该结果否定的是“独立 hard-T 更新组合的误差相消是主要风险源”及其特定风险分数，不否定 R042c 的局部 hard-T 改善，也不否定跨层约束的一般方向。当前 `hard_lx_ly` 仍不是联合求解器；由于 R054 未支持取消机制，不能继续实现旧 R055 的 no-cancellation joint solver。

两个 split 中 boundary checkpoint NMSE 与 mean-NLL harm 的相关反而稳定较高（0.5901/0.6388）。下一步只检验这个更简单、结果直接支持的 proxy 是否能形成有用而非过度保守的接受门。
