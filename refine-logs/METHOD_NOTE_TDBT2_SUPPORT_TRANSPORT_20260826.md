# TDBT2 Method Note: Support-Transport PTQ

日期：2026-08-26

状态：clean-room method note；不启动实验。

## 1. Problem Anchor

固定三值码本下，PTQ 一次性选择的离散 codeword 与预训练模型函数之间存在 compatibility gap；QAT 能通过量化前向下的状态调整缓解该 gap，但成本高。目标是在不训练 latent FP weights、不依赖目标模型 QAT teacher 的条件下，设计一个 PTQ 级机制，利用三值 `{-1,0,+1}` 的支撑/极性结构改善量化后函数保持。

## 2. Legacy Evidence Used

旧结果只用于以下四点：

- 采用修正后的 explicit next-token CE，避免 HuggingFace labels 二次 shift；
- 4090 24GB 下优先使用 OPT-125M sanity、OPT-350M primary；
- 旧 QAT trajectory 提醒：早期状态变化主要是 `0 <-> nonzero` support change，而不是 sign flip；
- 旧 path-barrier 结果只说明“路径 barrier 可测”，不作为新方法成功证据。

不继承旧结果中的 layer gate、threshold 成功标准、zero-mediated sign-flip 主张或 QAT checkpoint/logits/latent weights。

## 3. Method Thesis

TDBT2 的最小机制不是符号翻转路径，而是三值支撑运输：

> 把三值权重拆成 support mask `M` 和 polarity `S`，在固定 nonzero budget 下做成对 support transport，用路径 barrier 约束每一步的函数扰动，并用 composed-operator distortion 选择能够改善 Q/K 与 V/O 耦合函数的离散终点。

三值特异性来自 `0` 状态。二值没有 `support off` 状态，只能改 sign；普通多 bit 有多个幅值，support 不再是最自然的主自由度。三值的 `{-alpha,0,+alpha}` 刚好把“是否参与表达”和“参与后的方向”分开：

```text
W_hat_g = alpha_g * M_g * S_g
M_g in {0,1}
S_g in {-1,+1} when M_g=1
```

TDBT2 首轮只更新 `M`，默认 `S = sign(W_fp)`，冻结或最后一次 refit `alpha`。这样先验证支撑运输是否是主要可迁移机制，避免把 support、sign、scale 和 path 全部混成一个调参球。

## 4. Strict PTQ Boundary

Strict TDBT2 不使用：

- QAT checkpoint；
- QAT logits；
- QAT latent weights；
- QAT state prior；
- 目标模型短 QAT 生成的 transition margin。

允许使用：

- 原始 FP checkpoint 作为 frozen reference；
- calibration activations；
- 一次 quantized-point gradient 用于 candidate ranking 的 `TDBT2-G` 变体；
- forward-only salience 用于 `TDBT2-F` 变体。

如果未来引入短 QAT state prior，该分支必须改名为 `QAT-assisted`，不得作为主方法。

## 5. Candidate Operations

首轮只考虑同层同矩阵或 Q/K、V/O 成对矩阵内的 support transport：

```text
donor:    M_i = 1 -> 0
receiver: M_j = 0 -> 1, S_j = sign(W_fp,j)
budget:   sum(M) unchanged
```

路径约束不是为了证明真实 QAT 一定这么走，而是为了避免离散搜索一步跨出局部低损失区域：

```text
T0 -> deactivate donor -> activate receiver -> T1
```

候选只有在每个中间状态都满足 local trust region 时才允许进入 beam。该 trust region 只作为安全约束，不作为论文核心创新。

## 6. Objective

首轮不做完整全模型 teacher KL。只使用缓存 FP reference activations 和 composed operators。

Q/K:

```text
D_QK = NMSE( (X W_Q^T) (X W_K^T)^T, (X W_Qfp^T) (X W_Kfp^T)^T )
```

V/O:

```text
D_VO = NMSE( (X W_V^T) W_O^T, (X W_Vfp^T) W_Ofp^T )
```

Local trust:

```text
D_local = NMSE( X W_hat^T, X W_ptq^T )
accept only if D_local <= tau * D_local_baseline
```

推荐初值：

- `tau = 1.05`，仅在 plan 中预注册，不能事后调；
- beam width `4`；
- max accepted swaps per layer/operator `64`；
- final alpha refit `1` 次。

## 7. Minimal Falsification

若在相同候选预算下：

- support-transport 不优于 endpoint-only beam；
- 或只在 fit split 改善，validation/untouched 不改善；
- 或 one-shot quantized-gradient 已达到同等效果；
- 或 wall-clock 超过 direct PTQ 3x；

则关闭“support transport 是必要机制”的主张。后续可以保留“QAT/PTQ gap 诊断”作为论文方向，但不继续推进 TDBT2 主方法。

