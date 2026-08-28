# CEGSP P5-0 Strong PTQ Protocol Audit

日期：2026-08-28  
审计对象：PT² official commit `9e943e6` 及此前 OPT-350M baseline reproduction 结果  
结论类型：protocol audit，不是性能比较，不启动 GPU。

## 1. 审计目标

P5 需要回答的不是“PT² 数字是否最好”，而是：

> PT² / strong ternary PTQ 生成的 ternary state 是否仍属于当前 CEGSP 定义的合法 feasible space？

当前 CEGSP 的核心 feasible space 是 centered group-wise ternary：

```text
Q_i = α_g s_i z_i,    s_i ∈ {−1,+1}, z_i ∈ {0,1}
```

其中 zero state 是真实 0；support relocation 是在相同 group/budget 下交换 zero/nonzero state。

## 2. PT² codebook 审计

PT² `quantizer.py` 显示：

- `ternary_init` 先计算 row-wise mean：`mean = x.mean(dim=1, keepdim=True)`，见 `quantizer.py:16`；
- 再对 centered weight 做阈值分配，得到 `ternary_matrix ∈ {-1,0,+1}`，见 `quantizer.py:21-23`；
- 最终 reconstructed ternary weight 是：

```python
w_ternary = ternary_matrix * scale[:, None] + mean[:, None]
```

见 `quantizer.py:41-43`。

因此 PT² 的实际 codebook 是 row-wise affine ternary：

```text
{ μ_i − α_i, μ_i, μ_i + α_i }
```

而不是当前 CEGSP 的：

```text
{ −α_g, 0, +α_g }
```

`update_ternary` 也明确写到，它 snap 到 `{mean-alpha, mean, mean+alpha}`，见 `quantizer.py:292-307`。

## 3. Scale 与 group size 审计

PT² `TernaryQuantizer` 有 `groupsize` 参数，见 `quantizer.py:387-394`。  
但 ATQ 的 `ternary_init` / `solve_closed_form_alpha_mu` 是按 row 计算 mean/scale：

- `mean` shape 是 `[out_channels]`；
- `scale` shape 是 `[out_channels]`；
- 每个 row 有一个 affine center μ 和 scale α。

在 `gptq.py` 中，量化是在 column block 上进行：`W1 = W[:, col_st:col_ed]`，再调用 `braq_quantizer.quantize(W1, S=S, ...)`，见 `gptq.py:300-315`。  
这意味着 PT² 的 effective scale/mean 与 column block、row、activation-aware matrix S 耦合，而不是当前 CEGSP 简单的 group-wise α。

## 4. Zero state 审计

当前 CEGSP：

```text
z_i = 0  ⇒  Q_i = 0
```

PT²：

```text
T_i = 0  ⇒  Q_i = μ_i
```

所以 PT² 的 “zero state” 不是数学零值，而是 affine center μ。  
这会直接影响 CEGSP 的三值 support relocation：

- 当前 CEGSP 的 support exchange 是 `0 ↔ ±α`；
- PT² affine ternary 的对应 exchange 应是 `μ ↔ μ±α`；
- 如果原封不动使用 centered CEGSP，会把 PT² 的 affine center 错当真实 0，破坏 feasible space。

## 5. Cardinality 与 sign 审计

PT² 的 ternary matrix 确实有 `{-1,0,+1}` 离散 state。  
但是：

- 其 cardinality 是 ATQ/AGA 根据 row-wise affine fitting 与 activation-aware S 更新出来；
- relocation 后是否保持原 row/block 的 support cardinality，需要重新定义；
- receiver sign 不能简单使用 FP sign，因为实际选择应比较 `μ+α` 与 `μ−α` 的函数损失或 centered residual sign。

因此，当前 CEGSP 的 receiver sign rule 不能无修改接入 PT²。

## 6. Scale update 审计

PT² ATQ 会更新 α 和 μ：

- `solve_closed_form_alpha_mu(new_ternary, new_matrix)`，见 `quantizer.py:83`；
- AGA 分支还用 activation-aware matrix S 解 α/μ，见 `quantizer.py:107-160`。

当前 CEGSP 则冻结 α，不重估。  
若要接入 PT²，必须预注册一种规则：

1. freeze μ/α，只做 affine-state support relocation；
2. 或 relocation 后重新 closed-form refit μ/α；
3. 或只把 CEGSP 作为 score/action selector，再调用 PT² AGA refit。

这三种是不同方法，不能混为一谈。

## 7. Calibration / reconstruction / activation info 审计

PT² strong PTQ 已经使用 activation-aware information：

- `S = inpᵀ inp`，见 `gptq.py:300-315`；
- `gptaq=True`，见 `quantize.py:217-239`；
- SSR 分支会 reorder columns，见 `quantize.py:209-230` 与 `gptq_ssr.py` 的 reorder 逻辑。

因此若 CEGSP 接在 PT² 后，真正问题是：

> activation-aware reconstruction 后是否仍残留 task CE-loss-relevant affine support error？

而不是简单比较 Direct+CEGSP 与 PT²。

## 8. Eval 协议审计

已有 `CEGSP_12A_OFFICIAL_BASELINE_AUDIT_20260827.md` 显示：

- clean FP16 reference：W2 PPL 22.0046，C4 PPL 22.5898；
- official `quantize.py ... fp16` 入口不是干净 FP16 reference；
- PT² ATQ/ATQ+SSR 在 OPT-350M 上 native protocol finite 但严重退化；
- 当前 PT² strong baseline reproduction 未通过。

因此 PT² 不能马上作为 strong baseline 进入 P5-1。

## 9. P5-0 Protocol Audit 表

| 项目 | 审计结论 | 对 P5-1 的影响 |
|---|---|---|
| Codebook | PT² 是 `{μ−α, μ, μ+α}`，不是 `{−α,0,+α}` | 当前 CEGSP 不能原封不动接入 |
| Scale | row/block affine α，与 activation-aware S 耦合 | 需要 μ/α-aware rule |
| Group size | 存在 groupsize/block，但 effective fitting 不是当前 CEGSP group α | 需重新定义 exchange group |
| Zero state | PT² zero state 对应 `Q=μ`，非真实 0 | 当前 zero/nonzero support 语义不兼容 |
| Cardinality | ternary state 有 support，但由 ATQ/AGA 更新 | exchange 合法性需重定义 |
| Sign | receiver sign 不能直接用 FP sign | 需 centered residual 或 loss-based sign |
| Scale update | PT² 会解 α/μ；CEGSP 当前冻结 α | 必须预注册 freeze/refit |
| Calibration | PT² 使用 128×2048 calibration | 与 P3/P4 compact protocol 不同 |
| Reconstruction | GPTQ/GPTAQ layer/block reconstruction | CEGSP 若接入应是 residual CE repair |
| Activation info | 使用 S / Hessian-like activation info | CEGSP 需证明 complementary |
| Teacher/QAT | 无 QAT teacher/optimizer | PTQ 范式兼容 |
| Eval | 官方 2048 PPL 与 CEGSP compact NLL 不同 | 主表需统一协议 |

## 10. P5-0 最终判定

```text
P5-0 verdict: BLOCK_P5-1_AS_ORIGINAL_CENTERED_CEGSP
```

含义：

当前不能直接跑：

```text
PT² → current CEGSP
```

因为这不是合法同一 feasible space 的 refinement，会把 affine ternary center μ 错当真实 zero。

但可以继续两个方向：

### 方向 A：Affine-ternary CEGSP adapter

定义：

```text
Q = μ + αT,    T ∈ {−1,0,+1}
```

在 `T` 空间做 support relocation：

```text
T_i = 0 ↔ T_j = ±1
```

并预注册：

- μ/α freeze；
- 或 μ/α closed-form refit；
- receiver sign 用 centered residual / CE score 决定。

这才是合法的 `PT² + CEGSP`。

### 方向 B：Strong baseline 暂不接入

如果不想引入 affine-CEGSP 新方法，则 PT² 只能作为外部 strong baseline；当前 CEGSP claim 保持为：

```text
task-aware support relocation for centered/direct ternary PTQ
```

## 11. 建议下一步

不要直接启动 P5-1。下一步应先做一个极小的 adapter feasibility 实验：

```text
P5-AFFINE-0:
For one OPT-350M layer/block,
recover PT²-style μ, α, T representation,
test whether μ/α-frozen affine support relocation is numerically well-defined
and whether it preserves codebook/cardinality.
```

只有这个 feasibility 通过后，才考虑：

```text
PT²-affine state + affine CEGSP
```

否则 P5 与 PT² 的组合会变成概念混搭，论文风险很高。

