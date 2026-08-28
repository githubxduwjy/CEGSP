# Round 1 Refinement: No-Cancellation Path-Constrained Ternarization (NC-PTQ)

## Problem Anchor

- **Bottom-line problem**: 在严格 PTQ 约束下，将已预训练 LLM 权重编码为 `{-1,0,+1}`，保留模型函数，而不只是拟合 FP16 权重。
- **Must-solve bottleneck**: 局部 Hessian/激活误差改善不能预测全量化轨迹改善；离散 `T` 更新在少样本下具有层位与分布依赖。
- **Non-goals**: 不做 projection-mask 或层 mask 枚举，不事后调 epsilon，不依赖 QAT/LoRA/蒸馏，不同时堆叠旋转、混合精度和新 kernel。
- **Constraints**: 少量通用校准数据，单张 RTX 4090 24GB，三值权重表示，量化成本显著低于 QAT。
- **Success condition**: 不用测试集挑层，联合优化在保持每层局部映射的前提下改善跨层轨迹，并在未见分布上超过强三值 PTQ 基线。

## Anchor Check

- 原始瓶颈是三值离散表示与全模型函数的失配，不是旋转或网格不够精细。
- 新方法直接更新 hard `T`，但将“保持每层映射”作为可行性约束，将“保持跨层函数”作为优化目标。
- 删除 Fisher ranker 不会造成问题漂移；反而避免转向敏感度分配论文。

## Simplicity Check

- **唯一主贡献**: 三值跨层联合优化中的 per-layer no-cancellation trust region。
- **删除**: Fisher/K-FAC 排序器、独立 CVaR 筛选模块、多坐标系竞争。
- **保留**: fit/validation 分离只作为少样本离散优化的常规泛化控制，不宣称为独立创新。

## Core Hypothesis

CAT-Q 类的 sliding-layer reconstruction 和全局 error compensation 可能在校准集上利用层间误差相消：窗口边界误差很小，但某一层的量化残差变大。三值扰动远大于 3/4-bit 扰动，这种 calibration-specific cancellation 更易在分布变化后失效。

**NC-PTQ 的命题**：只允许在每一层都不比强 initializer 更差的可行域内做跨层联合优化，可以减少层间误差相消导致的过拟合，使 hard ternary code 更能迁移到未见文本。

## Mathematical Formulation

对窗口 `S={l,...,b}`，强 initializer 产生

\[
\widehat W_j^0=\alpha_j^0T_j^0+\mu_j^0,
\qquad T_j^0\in\{-1,0,+1\}.
\]

在当前已量化前缀的真实输入 `h_{l-1}^q` 上，定义每层局部残差：

\[
e_j(x;T_j)=f_j(h_{j-1}^{q};W_j)-f_j(h_{j-1}^{q};\widehat W_j).
\]

局部目标是对分布 `d` 归一化的输出误差：

\[
L_{loc}^{j,d}(T_j)=
\mathbb E_{x\sim D_d}
\frac{\lVert e_j(x;T_j)\rVert_2^2}
{\lVert f_j(h_{j-1}^{q};W_j)\rVert_2^2+\epsilon}.
\]

窗口路径目标为：

\[
L_{path}^{S,d}(T_S)=
\mathbb E_{x\sim D_d}
\frac{\lVert F_S(h_{l-1}^{q};W_S)-F_S(h_{l-1}^{q};\widehat W_S)\rVert_2^2}
{\lVert F_S(h_{l-1}^{q};W_S)\rVert_2^2+\epsilon}.
\]

核心优化是：

\[
\min_{T_S,\alpha_S,\mu_S}
\max_{d\in\mathcal D_{fit}}L_{path}^{S,d}(T_S)
\]

subject to

\[
L_{loc}^{j,d}(T_j)
\le L_{loc}^{j,d}(T_j^0)+r_{j,d},
\quad \forall j\in S,\ d\in\mathcal D_{fit}.
\]

`r_{j,d}` 不是看到 test 后调的 epsilon；它由 fit/validation 上 initializer 的 paired bootstrap 不确定性预先给出。第一版使用最保守的 `r=0`，只在显示小样本噪声导致全部不可行时，才在新预注册实验中切换到置信上界。

### Why the constraint targets cancellation

对局部线性化 `A_j=partial f_j/partial h`，窗口终点误差近似为

\[
\delta h_b\approx
\sum_{j=l}^{b}
\left(\prod_{k=j+1}^{b}A_k\right)e_j.
\]

只最小化 `||delta h_b||` 允许不同项方向相反而抵消。局部 trust region 限制每个 `e_j` 的大小，不禁止有益补偿，但禁止通过恶化某层来换取校准窗口上的偶然抵消。

## Optimization Algorithm

1. 用 PT² 或 CAT-Q 生成 `T^0,alpha^0,mu^0`。
2. 在 `D_fit` 上以 R042c hard-coordinate update 生成单层可行移动集。
3. 对固定 `w=2` 窗口做 constrained block-coordinate descent：只接受降低 minimax path loss 且不破坏任一 local constraint 的 ternary flips。
4. 在独立 `D_val` 上用同一组约束早停；不用 final W2/C4 test 做决策。
5. 冻结 `T`，在 fit+validation 上重拟合 `alpha,mu`，提交该窗口并继续。

没有新的可训练推理模块，没有增加 bpw。如果 hard coordinate solver 过慢，可以以 soft ternary 作求解器，但这只是 implementation variant，不改变方法命题。

## Novelty Boundary

- **vs PT²**: PT² 的 activation-aware 阶段冻结 `T`且局部优化；NC-PTQ 在可行域内联合更新 hard `T`。
- **vs CAT-Q**: CAT-Q 最小化 sliding-window boundary reconstruction；NC-PTQ 的关键增量是每层无退化约束，直接防止 calibration-specific cancellation。
- **vs cross-layer error compensation**: 对方强调全局累积误差和特征统计；NC-PTQ 不声称首次联合离散优化，而是声称“局部 no-harm 可行域减少三值联合优化的误差相消过拟合”。
- **vs KronQ**: 不依赖输出梯度协方差、旋转或混合精度，方法问题不同。

## Minimal Validation

### R054: cancellation mechanism audit

- 固定 `(10,11)` 和 `(20,21)` 两个对照窗口，使用全新校准/验证数据。
- 比较 initializer、independent hard-T、unconstrained two-layer hard-T。
- 必须观察：unconstrained 方法是否出现“boundary 更好但某层 local 更差”，以及该 cancellation gap 是否在 untouched/OOD 上对应 NLL 退化。
- 若没有这种现象，NC-PTQ 的核心机理缺少证据，不继续全模型实现。

### R055: constrained optimization isolation

- 比较 initializer / independent / unconstrained joint / NC-PTQ，候选数、求解步数与校准 token 匹配。
- 主指标：untouched 双分布 path NLL、CVaR10、每层 local violation rate、accepted flip ratio。
- Gate：NC-PTQ 的 local violation=0，且在两窗口的 worst-distribution NLL 优于 unconstrained joint；否则不进 R056。

### R056: automatic full-model method

- 从 0–31 层用固定 `w=2` 执行 NC-PTQ，不使用测试结果挑层。
- 基线：PT² matched budget、CAT-Q/sliding reconstruction、unconstrained joint hard-T。
- 主表：W2/C4 PPL + zero-shot average + actual bpw + PTQ time/VRAM。
- 通过后才做第二模型家族和 3 seeds。

## Compute and Failure Policy

- R054: 2–4 GPU-hours。
- R055: 4–8 GPU-hours。
- R056: 8–16 GPU-hours。
- 若约束导致 accepted flip ratio 近似 0，记录为“safe but vacuous”，不宣称成功，不用事后 epsilon 放宽拯救。
- 若 R054 机理不成立，回到“离散优化的分布泛化”问题，而不否定 hard-T 本身。
