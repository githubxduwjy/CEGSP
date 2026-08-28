# Research Proposal: Bi-level Path-Safe Ternarization (BiPaST)

## Problem Anchor

- **Bottom-line problem**: 在严格 PTQ 约束下，将已预训练 LLM 的权重编码为 `{-1,0,+1}`，保留模型函数，而不只是拟合 FP16 权重。
- **Must-solve bottleneck**: 局部 Hessian/激活误差改善不能预测全量化轨迹改善；离散 `T` 更新在少样本下具有层位与分布依赖。
- **Non-goals**: 不做 projection-mask 枚举，不事后调 epsilon，不依赖 QAT/LoRA/蒸馏，不把单一好层当作方法。
- **Constraints**: 少量通用校准数据，单张 RTX 4090 24GB，完整三值权重表示，量化成本须显著低于 QAT。
- **Success condition**: 不用测试集挑层，算法自动接受可泛化的 hard-`T` 更新、拒绝有害更新，并在 WikiText2/C4 和下游任务上稳定超过强三值 PTQ 基线。

## Evidence Synthesis

| Evidence | Observation | What it rules out |
|---|---|---|
| R014–R019/R033 | 局部 Haar/配对最多只有小幅稳定改善 | 坐标变换不是主线 |
| R034–R041 | activation-ordered Hadamard 局部和部分 PPL 有收益，全层/SSR 插入却不稳定 | 不能用局部误差或单层胜利推出全模型方法 |
| R042c | hard-`T` 在 untouched block NMSE 中位改善 5.83%，胜率 96.43% | 离散 `T` 不是不可更新，候选生成机制有效 |
| R043–R046 | 局部和 FP16-rest 均预测失败；全量化上 W2/C4 分裂 | 局部目标、FP16 suffix 和单分布 gate 均不充分 |
| R047–R052 | `(0,1)` 不泛化，`hard_l11` 严格 PASS，`(20,21)` 回退 official | 更新应是自动稀疏的，不应全层开启或手工挑层 |

## Technical Gap and Literature Boundary

PT² 在 AGA 中冻结 `T`，避免激活目标下的少样本过拟合。CAT-Q 已经使用 sliding-layer reconstruction 和软三值化，因此“跨层重构”本身不是新意。2026 年的 cross-layer error compensation 甚至对全模型离散码和尺度做联合优化；KronQ 则表明输出梯度协方差可以补足只看输入激活的 Hessian。

因此真正未解的缺口不是“要不要跨层”，而是：

> 如何在少样本严格 PTQ 中，让跨层功能目标决定 hard ternary code 是否可以被提交，同时用每层信任域防止跨层补偿牺牲局部映射，并用拟合/验证分离控制离散过拟合。

## Method Thesis

**BiPaST** 将三值化重写为“局部可行域中的离散提案 + 量化轨迹上的外层提交”：内层只生成不破坏单层映射的 hard-`T` 候选，外层用未参与拟合的多分布校准样本、在已量化前缀中评估路径风险，逐窗口接受或回退。

### Dominant contribution

一个面向 hard ternary assignments 的 **bilevel Pareto trust-region commit rule**，而不是新的旋转、新的量化网格或简单的跨层 loss 相加。

### Explicit non-contributions

- 不声称首次跨层量化或首次输出感知 PTQ。
- 不把 layer 11 作为手工 mask。
- 不在当前阶段同时追求 W1.58A4、新 kernel 和新坐标变换。

## Proposed Method

### 1. Representation and baseline

对第 `l` 层权重：

\[
\widehat W_l = \alpha_l T_l + \mu_l,\qquad T_l\in\{-1,0,+1\}^{m_l\times n_l}.
\]

PT²/CAT-Q 只作为强 initializer 和 baseline。方法的新变量是 hard code proposal `T'_l` 的提交决策，不增加推理参数。

### 2. Inner problem: locally feasible hard-T proposal

在 fit split `D_fit` 上用 R042c 已验证的 Hessian-aware 坐标下降生成 `T'`，但必须满足每层相对于 initializer 的信任域：

\[
\mathcal L_{\text{loc}}^{(l)}(T'_l;D_{fit})
\leq
\mathcal L_{\text{loc}}^{(l)}(T_l^0;D_{fit}).
\]

它保留用户要求的单层约束，也防止 joint optimization 用下游补偿掩盖某一层的严重破坏。

### 3. Path score: downstream-sensitive boundary risk

对相邻窗口 `S=[l,l+w]`，在当前已量化前缀中得到窗口边界误差

\[
\delta h_b(x)=h_b(x;T'_S,T_{<l}^{committed})-h_b(x;T^0_S,T_{<l}^{committed}).
\]

不再单纯用 `||delta h||^2`，而用一次 teacher backward 得到的输出侧 Fisher/K-FAC 草图 `G_b`：

\[
\mathcal R_{path}(S)=
\mathbb E_{x\sim D}\left[\delta h_b(x)^\top G_b(x)\delta h_b(x)\right].
\]

这一项只用于廉价排序/淘汰候选，不直接作为可事后调权的最终加权 loss。首先必须通过 R054 证明它比 local NMSE 和普通 window NMSE 更能预测全模型 NLL 符号；若不能，直接删除 Fisher 路线。

### 4. Outer problem: validation-only Pareto commit

`D_fit` 仅生成候选；`D_val` 仅决定 commit。对至少两个通用校准分布 `d`，定义：

\[
\Delta_{d}^{mean}=\mathbb E[\ell_{T'}-\ell_{T^0}],\qquad
\Delta_{d}^{tail}=\operatorname{CVaR}_{0.1}(\ell_{T'}-\ell_{T^0}).
\]

提交规则为：

\[
\text{commit}(T'_S)=1
\iff
\begin{cases}
\mathcal L_{loc}^{(l)}(T'_l)\le \mathcal L_{loc}^{(l)}(T_l^0),&\forall l\in S,\\
\Delta_d^{mean}\le 0,\ \Delta_d^{tail}\le 0,&\forall d,\\
\text{nonfinite}(T'_S)=0.
\end{cases}
\]

若不通过，回退 initializer。这使层稀疏性成为算法输出，而不是手工搜索。

### 5. Sequential integration

从浅到深以固定宽度 `w=2` 滑动：

```text
initializer ternary model
  -> fit split generates locally feasible hard-T proposals
  -> Fisher path score prunes obviously harmful proposals
  -> quantized-prefix validation computes mean/tail risk
  -> commit or rollback the whole window
  -> refit alpha,mu after T is frozen
  -> continue to next non-overlapping window
```

开始只用 `w=2`，不扫 window size。若该设计不能超过 CAT-Q 类普通 sliding reconstruction，不增大窗口来掩盖失败。

## Why This Is Not a Loss Pile

1. local loss 是 hard feasibility constraint，不是 lambda 加权项。
2. Fisher path score 是候选排序器，必须先独立验证预测性。
3. mean/tail multi-distribution NLL 是外层 commit rule，不反向传播到 fit split。
4. 三者属于“提案—筛选—提交”的不同角色，超参只保留预注册的零退化边界。

## Claim-Driven Validation

### C1: downstream-sensitive path score can predict hard-T transfer

- **R054**: 在已有 `(0,1)/(10,11)/(20,21)/(30,31)` 候选上回放，比较 local NMSE、window NMSE、Fisher path risk 对 untouched W2/C4 NLL 方向与排序的预测能力。
- **Gate**: leave-one-window-out 下，Fisher score 对“四项全非退化”的 balanced accuracy >=0.75，且 Spearman 明显优于两个简单 proxy。
- **Failure meaning**: 不是整个 hard-T 方向失败，而是删除 Fisher proxy，直接使用更贵但可靠的 quantized-context outer gate。

### C2: bilevel commit beats independent and unconstrained joint ternarization

- **R055 mechanism test**: 冻结两个对照窗口 `(10,11)` 和 `(20,21)`，用全新 calibration windows；比较 initializer、independent hard-T、普通 two-layer reconstruction、BiPaST。
- **Gate**: BiPaST 应保留已知的正候选类型并拒绝有害候选；在 untouched 上每层 local constraint 和双分布 mean/CVaR 同时非退化。
- **Failure meaning**: 若普通 joint reconstruction 等价，则 bilevel/local trust-region 没有新增价值，不进入全模型方法。

### C3: automatic full-model ternarization without layer picking

- **R056**: 将冻结 BiPaST 从 layer 0 自动运行到 layer 31，所有决策只看 fit/validation calibration，之后一次性解封 W2/C4 测试。
- **Baselines**: PT² official matched-budget、CAT-Q/sliding reconstruction（可复现配置）、hard-T all-on；`hard_l11` 仅作 oracle diagnostic，不列为公平方法。
- **Primary metrics**: W2/C4 PPL、zero-shot average、accepted-window fraction、PTQ wall time、peak VRAM、实际 bpw。
- **Go gate**: 两个 PPL 都不退化，至少一个有明显改善；若只改善 C4 又伤 W2，不声称通用方法。

## Execution Order and Stop Rules

1. 完成 R053，冻结四个位置对照；不再验证其他层。
2. R054 先验证 predictor，防止在错误的 Fisher 直觉上耗费算力。
3. R055 验证“local constraints + outer commit”是否比普通 cross-layer reconstruction 更有效。
4. 只有 R055 过 gate 才运行 R056 全模型。
5. R056 过 gate 后再做 3 seeds、第二模型家族和任务表；否则转为负结果/机理论文，不继续扫层。

## Compute Estimate

- R054: 约 1–2 GPU-hours，主要是回放与一次 backward 统计。
- R055: 约 4–8 GPU-hours，取决于是否复用候选。
- R056: 约 8–16 GPU-hours，并严格记录 PTQ 成本。
- 主线首轮总预算：约 13–26 RTX 4090 GPU-hours。

## Highest-Risk Assumptions

1. Fisher path risk 在 1.58-bit 的大离散扰动下仍有足够的排序能力。
2. validation-only commit 不会因零退化过于保守而全部回退。
3. 相邻两层路径足以暴露主要害处；若不足，也只在机理证据支持后扩大窗口。
