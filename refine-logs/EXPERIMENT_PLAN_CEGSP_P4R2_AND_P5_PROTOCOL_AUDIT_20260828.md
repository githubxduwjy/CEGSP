# CEGSP P4R2 Fairness Check 与 P5-0 Strong PTQ Protocol Audit 计划

日期：2026-08-28

## 1. 为什么这不是横向发散

P4-R 已经给出强证据：

- CEGSP 在 W2 上优于 properly swept one-step QAT；
- CEGSP 明显优于 edit-matched one-step QAT；
- CEGSP 使用 768 个 targeted ternary changes，而 one-step QAT validation-best 需要 16,799 个 changes；
- multi-step QAT 是更强但成本更高的 reference。

本轮不继续扩大 P4-R，只补两个公平性缺口：

1. **更严格的 768 附近 edit matching**：P4-R 中 edit-matched one-step QAT 的 changed coords=528，离 CEGSP 的 768 仍有差距。本轮密扫 eta 使 one-step QAT 更接近 768。
2. **QAT update scope**：P4-R 的 one-step QAT 更新 all Q/K layers，而 CEGSP 最终只部署 selected top-6 layers。本轮比较：
   - all Q/K layers one-step QAT；
   - CEGSP-selected layers only one-step QAT。

CEGSP 本身不变。

## 2. P4R2 运行配置

Run ID：

`CEGSP-V2-P4R2-OPT350M-STRICT-EDITMATCH-SCOPE-OFFSET2`

固定设置：

| 项目 | 设置 |
|---|---:|
| 模型 | `facebook/opt-350m` |
| 数据 | Wikitext-2 + C4 validation |
| fit / val / W2 untouched / C4 untouched | 8 / 8 / 64 / 32 |
| offsets | fit=8192, val=8192, C4=16384 |
| CEGSP | canonical support relocation top-6 |
| eta sweep | 3e-5 到 1e-3，重点覆盖 768 changes 附近 |
| QAT scopes | all Q/K layers；CEGSP-selected layers only |

判据：

- 若 CEGSP 仍优于 strict edit-matched one-step QAT，则 P4-R 机制证据更稳；
- 若 selected-layer QAT 接近或超过 CEGSP，则说明 CEGSP 的优势可能部分来自 layer/action selection，需要在论文中更谨慎表述；
- 若 all-layer QAT 明显更强但 changes 远大于 CEGSP，则仍支持“CEGSP 是小预算离散修复”。

## 3. P5-0 Protocol Audit 范围

P5-0 不跑 GPU，先审计 strong ternary PTQ 是否能合法接入 CEGSP。

审计对象：

- PT² official commit `9e943e6`；
- 本地参考代码：
  - `reference-code/pt2_official_9e943e6/quantizer.py`
  - `reference-code/pt2_official_9e943e6/quantize.py`
  - `reference-code/pt2_official_9e943e6/gptq.py`
  - `reference-code/pt2_official_9e943e6/gptq_ssr.py`
- 既有审计报告：
  - `refine-logs/CEGSP_11A_11A2_STRONG_BASELINE_AUDIT_20260827.md`
  - `refine-logs/CEGSP_12A_OFFICIAL_BASELINE_AUDIT_20260827.md`

## 4. P5-0 必须回答的问题

| 项目 | 必须确认的问题 |
|---|---|
| Codebook | 是否真正是 `{−α,0,+α}`，还是 `{μ−α, μ, μ+α}` |
| Scale | α 是 per-group / per-channel / per-row / per-weight |
| Group size | CEGSP relocation 能否在相同 group 内定义 |
| Zero state | 是否有真实数学零态，还是 offset center μ |
| Cardinality | support 数是否固定；exchange 后是否合法 |
| Sign | receiver sign 的定义能否继续使用 |
| Scale update | relocation 后 α/μ 应冻结还是重估 |
| Calibration | strong PTQ 是否用了额外 calibration information |
| Reconstruction | 是否用了 layer/block reconstruction |
| Activation info | 是否用了 activation-aware objective |
| Teacher/QAT | 是否包含训练式 refinement |
| Eval | W2/C4 tokenizer、seq length、batch 协议是否一致 |

## 5. P5-0 初步风险

PT² 的 `quantizer.py` 初读已经显示：

```text
w_ternary = ternary_matrix * scale[:, None] + mean[:, None]
```

这意味着 PT² ATQ 的三值 codebook 是 row-wise affine ternary：

```text
{μ_i - α_i, μ_i, μ_i + α_i}
```

而当前 CEGSP 的 feasible space 是 group-wise centered ternary：

```text
{−α_g, 0, +α_g}
```

因此，当前不能预设 PT² 与 CEGSP 在同一 feasible space。P5-0 的核心就是判定：

1. 是否能定义 affine-ternary CEGSP；
2. 是否必须先实现 `μ-aware support relocation`；
3. 或者 PT² 暂时只能作为不可直接组合的 strong baseline，而不能做 StrongPTQ+CEGSP。

## 6. P5-1 的启动条件

只有当 P5-0 判定以下条件成立时，才启动 GPU 实验：

1. strong PTQ 权重可被解释为合法三值 state；
2. relocation 后仍留在 strong PTQ 的 feasible space；
3. α/μ freeze/refit 规则明确；
4. evaluation protocol 可比；
5. strong PTQ 本身不是异常退化 baseline。

否则不启动 `StrongPTQ+CEGSP`，避免产生不可解释或不可投稿的结果。
