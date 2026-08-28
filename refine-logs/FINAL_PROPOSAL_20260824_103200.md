# Research Proposal: NC-PTQ — No-Cancellation Path-Constrained Ternarization

## 问题锚点

我们要解决的不是“如何让三值权重更像 FP16 权重”，而是“如何在严格 PTQ 中更新 `{-1,0,+1}` 离散结构，且不破坏预训练模型函数”。

现有实验已经证明：hard-`T` 在 block 层面确实有效（R042c: untouched NMSE 中位改善 5.83%，96.43% blocks 获胜），但局部改善、FP16 suffix 和普通两层边界误差都不能稳定预测全量化 NLL。`(0,1)` 在 W2/C4 上分裂，`hard_l11` 严格泛化，`(20,21)` 与 `(30,31)` 均回退 official。因此 hard-`T` 不应全层开启，也不应通过测试结果手工挑层。

## 核心命题

已有 CAT-Q 完成 sliding-layer reconstruction，新的 cross-layer error compensation 也已联合优化多层离散码。所以“跨层”不是新意。

NC-PTQ 的核心是：

> 在三值量化中，无约束跨层重构可能依赖 calibration-specific error cancellation：通过恶化某一层映射，让相邻层的误差在小校准集上相消。应当在“每一层都不比强三值 initializer 更差”的可行域内，再最小化跨层轨迹误差。

这正好对应“保留单层约束，同时引入跨层约束”：单层是可行性约束，跨层是优化目标，不用人工 `lambda` 把多个 loss 混成一个弹性标量。

## 方法

对第 `j` 层：

\[
\widehat W_j=\alpha_jT_j+\mu_j,
\qquad T_j\in\{-1,0,+1\}^{m_j\times n_j}.
\]

### 1. 强 initializer

使用 PT² 或 CAT-Q 产生 `T_j^0,alpha_j^0,mu_j^0`。initializer 必须是单层/单 block 稳定优化的三值基线，不能是已经通过跨层相消得到的模型。主实验先用 PT² fixed-`T` AGA initializer。

### 2. 真实量化上下文

对窗口 `S={l,...,b}`，所有候选使用已提交量化前缀产生的 `h_{l-1}^q`，不再使用 R045 已证伪的 FP16-rest proxy。

局部误差：

\[
L_{loc}^{j,d}(T_j)=
\mathbb E_{x\sim D_d}
\frac{\lVert f_j(h_{j-1}^{q};W_j)-f_j(h_{j-1}^{q};\widehat W_j)\rVert_2^2}
{\lVert f_j(h_{j-1}^{q};W_j)\rVert_2^2+\epsilon}.
\]

窗口路径误差：

\[
L_{path}^{S,d}(T_S)=
\mathbb E_{x\sim D_d}
\frac{\lVert F_S(h_{l-1}^{q};W_S)-F_S(h_{l-1}^{q};\widehat W_S)\rVert_2^2}
{\lVert F_S(h_{l-1}^{q};W_S)\rVert_2^2+\epsilon}.
\]

### 3. No-cancellation constrained objective

\[
\min_{T_S,\alpha_S,\mu_S}
\max_{d\in\mathcal D_{fit}}L_{path}^{S,d}(T_S)
\]

subject to

\[
L_{loc}^{j,d}(T_j)\le L_{loc}^{j,d}(T_j^0),
\quad \forall j\in S,\ d\in\mathcal D_{fit}.
\]

第一版固定为零退化，不调 epsilon。若约束导致候选接受率近似 0，结论是“safe but vacuous”，而不是用事后放宽拯救。

### 4. Hard discrete solver

复用 R042c 已验证的 Hessian-aware hard-coordinate proposal。在固定 `w=2` 窗口内，只接受同时满足两个条件的 ternary flip：

1. 不违反任一层局部约束；
2. 降低 worst-distribution path loss。

`D_fit` 用于候选生成，`D_val` 用于早停与整个窗口 rollback，final test 永不参与选择。冻结 `T` 后，再在 fit+validation 上重拟合 `alpha,mu`。

## 可检验的相消定义

对两层窗口 `(l,l+1)`，分别构造只量化前一层、只量化后一层和两层联合量化的边界残差 `u_l,u_{l+1},u_joint`。定义归一化相消指数：

\[
C_S=
\frac{\lVert u_l\rVert_2^2+\lVert u_{l+1}\rVert_2^2-\lVert u_{joint}\rVert_2^2}
{\lVert u_l\rVert_2^2+\lVert u_{l+1}\rVert_2^2+\epsilon}.
\]

`C_S>0` 表示联合边界误差小于两个单层残差能量之和，存在相消。核心机理需要预先证明：当无约束 joint 同时出现高 `C_S` 和某层 local regression 时，其 untouched/跨分布 NLL 更容易退化。如果没有这个关系，NC-PTQ 的主命题不成立。

## 创新边界

- PT² 提供强 initializer，但 AGA 冻结 `T`，无跨层约束。
- CAT-Q 联合重构 sliding window，但没有 per-layer no-harm feasible set。
- Cross-Layer Error Compensation 已经联合优化全模型离散码，NC-PTQ 不争夺“首次跨层”；它只主张“局部信任域可防止三值联合优化的校准特异性误差相消”。
- 方法不新增推理参数或 bpw，实质是新的离散可行域与求解规则。

## 最小验证路线

1. **R053**: `(30,31)` 已安全回退 official；四个位置对照完成，停止验证其他层。
2. **R054 机理审计**: 在冻结的 `(10,11)` 正对照和 `(20,21)` 安全回退对照上，比较 initializer、independent hard-T、unconstrained joint hard-T；验证 `C_S + local regression` 是否预测 untouched NLL 伤害。
3. **R055 方法隔离**: 在全新 fit/val/test 上比较 initializer / independent / unconstrained joint / NC-PTQ。四者使用相同 initializer、tokens、ternary proposal set、coordinate passes 和 max steps；额外报告实际 wall time。
4. **R056 全模型**: 只在 R055 通过后，以固定 `w=2` 从 0–31 层自动运行，不挑层；比较 PT²、CAT-Q-style unconstrained joint 和 NC-PTQ。

R055 的严格 gate 是：NC-PTQ local violation rate=0，accepted flip ratio 不近似 0，且在 untouched WikiText2/C4 的 worst-distribution NLL 优于 unconstrained joint。R056 的 gate 是 W2/C4 PPL 均不退化且至少一者显著改善，然后才扩展到 3 seeds、第二模型家族与 zero-shot 任务。

## 预期研究价值

如果成功，论文不是“PT² 加一个 loss”，而是给出一个更清晰的三值 PTQ 原则：在极端离散编码下，跨层补偿必须受局部保真约束，否则容易把校准集上的误差相消误当成函数保持。如果 R054 机理审计失败，则及时停止这一具体机制，但不否定 R042c 已证实的 hard-`T` 候选生成能力。
