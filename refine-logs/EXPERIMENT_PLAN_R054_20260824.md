# R054 预注册：hard-T 组合的 error-cancellation 机制审计

## 目标和边界

R054 不宣称当前 `hard_lx_ly` 是联合求解器；它是两个独立 hard-T 更新的组合。本轮只检验：组合后的边界误差是否因两个更新在小样本上相消而显得过度乐观。真正的 unconstrained/constrained joint solver 只在 R054 支持机制后于 R055 实现。

## 冻结设计

- 窗口：`(0,1)`, `(10,11)`, `(20,21)`, `(30,31)`；全部来自 R050–R053 的预注册深度对照，不新增层搜索。
- 候选：official, hard first, hard second, hard both。
- 新评分 windows 72–87：72–79 gate，80–87 test；W2/C4 完全同步。
- 量化配置：nsamples=8, blocksize=128, validation=25%, max_steps=4, seed=0。
- 不调 epsilon，不根据当前输出增减窗口。

## 指标

\[
C_S=
\frac{\|u_l\|_2^2+\|u_{l+1}\|_2^2-\|u_{joint}\|_2^2}
{\|u_l\|_2^2+\|u_{l+1}\|_2^2+\epsilon}.
\]

其中所有 `u` 是同一量化 official initializer 边界隐状态的变化，不是 FP16 绝对误差。风险分数预注册为：

\[
R_S=\max(C_S,0)\max(\Delta NMSE_{first},\Delta NMSE_{boundary},0).
\]

对 gate/test 分别计算 `R_S` 与 mean-NLL/CVaR10 harm 的 Spearman 相关，并与 boundary-NMSE delta 这个简单 proxy 比较。

## 机器可判定 Gate

两个 split 都必须满足：

1. `C_S>0` 且存在 checkpoint NMSE regression 的 sample-level 机制样本至少 8 个；
2. risk 对 mean NLL 的 Spearman rho >=0.20；
3. risk 对 CVaR10 的 Spearman rho >=0.20；
4. risk 的两指标平均 rho 至少比 boundary-only proxy 高 0.05；
5. 无新 nonfinite。

- **SUPPORT**: gate 和 test 都通过，进入 R055。
- **INCONCLUSIVE**: 任一 split 机制样本少于 8；停止 NC-PTQ，不增加层或事后放宽。
- **FAIL**: 机制样本充足但相关 gate 失败；否定 error-cancellation 解释，不否定 R042c hard-T 局部机制。

## 计算成本

四个窗口串行，预计 50–70 分钟，峰值 GPU 约 4–5 GiB，CPU 内存临时保留边界隐状态。
