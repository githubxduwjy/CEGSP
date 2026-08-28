# CEGSP-P5-C：真实 PT² → affine CEGSP 兼容性实验报告

日期：2026-08-28  
模型：`facebook/opt-350m`；GPU：RTX 4090 24GB  
云端入口：`xj-member.bitahub.com:42055`（与本轮已完成任务所在容器一致）  
原始结果：`results/remote-runs/cegsp_p5c_pt2_affine_opt350m_20260828_screen_p5c_pt2_affine_result.json`

## 1. 实验问题

本轮只检验一个论文关键 gate：

> 在官方 PT² ATQ 产生的真实 affine ternary state 上，冻结 `mu/alpha`，只用一次 fit-split quantized-point CE gradient，按照预先冻结的全模型 layer-ranking 和小预算做 Q/K support relocation，是否仍能改善 PT² 的 held-out language-model loss？

这不是 Direct+CEGSP 与 PT² 的最终性能竞赛，而是 `PT² → PT²+affine CEGSP` 的 compatibility test。

## 2. 固定协议

- PT² official commit：`9e943e6`，`method=atq`，`ssr=False`；全模型 decoder Linear 均按官方流程量化。
- CEGSP 候选：全部 24 层 Q/K；group/block size 128；每个 row-group 的 affine codebook 为 `q=mu+alpha*T`。
- 校准：Wikitext-2 fit，8 batches × batch size 2，即 16 条、128-token 输入；PT² 与 CEGSP fingerprint 一致。
- layer rule：每层 Q/K 候选 top-8 score sum 排名，固定选 top-6；每层 64 对 relocation，共 384 对、768 个 changed coordinates。
- `mu/alpha` 冻结；不使用 QAT teacher、latent FP training、optimizer、validation/untouched 选层或调预算。
- 评估：同一 compact evaluator，Wikitext-2 validation 8 batches、Wikitext-2 untouched 8 batches、C4 untouched 8 batches，seq 128、batch 2。

## 3. State parity 结果

首轮 harness 曾以 `1e-5` 绝对阈值比较 FP32 capture 与 FP16 deployed weight，误把首层大数的精度转换当成漂移。诊断确认：

- 误差只集中在 layer 0 的 Q/K/V；`fasterquant` 返回瞬间已与 capture 一致，说明不是记录错位；
- layer 0 的官方 ATQ 输出最大绝对值约 `1.0e3`，FP32 capture residual 为 `1.220703125e-4`，符合该数值量级下的 FP32 roundoff；
- 修复后的同一协议以 capture-FP32 residual `<1e-3`，并以最终模型与 `captured_q.to(FP16)` residual `<1e-3` 为 gate；最终 deployed residual 为 `0.0`。

正式结果为 `status=parity_passed`：

| 审计项 | 结果 |
|---|---:|
| Q/K module count | 48/48 |
| illegal T | 0 |
| nonfinite T | 0 |
| max capture codebook residual | 0.0001220703125 |
| max final-vs-deployed-capture residual | 0.0 |
| group / granularity | 128 / per-row per-group |
| permutation | disabled (`ssr=False`) |

因此可以确认：本轮不是因为 state mapping、group boundary、placeholder-T 或 evaluator mismatch 而无法比较。需要注意的是，官方 PT² 在本设置中确实产生了异常大的 layer-0 Q/K 数值；本轮没有事后修正它。

## 4. 预注册性能结果

| Variant | Val NLL | Δ vs PT² | W2 untouched NLL | Δ vs PT² | C4 untouched NLL | Δ vs PT² | edits |
|---|---:|---:|---:|---:|---:|---:|---:|
| PT² ATQ | 9.850781 | 0 | 10.170247 | 0 | 9.642318 | 0 | 0 |
| PT² + affine CEGSP top-6 | 9.900243 | +0.049462 | 10.303432 | +0.133185 | 9.645273 | +0.002955 | 384 pairs / 768 coords |
| PT² + matched random top-6 | 9.853100 | +0.002318 | 10.165869 | −0.004378 | 9.649151 | +0.006833 | 384 pairs / 768 coords |

所有 variant 均 finite，CEGSP 与 random 的 support cardinality、合法性审计均通过。

预注册 gate：

- `legality_pass = true`
- `finite_pass = true`
- CEGSP improves PT² validation = `false`
- CEGSP improves PT² W2 untouched = `false`
- CEGSP beats matched random on W2 = `false`
- `strong_compatibility_pass = false`

## 5. 结果解释

### 支持的结论

1. **真实 PT² 状态接口是可恢复的。** 官方 quantizer 返回的 placeholder `T` 不能直接使用，但在 quantizer 内部捕获的 `T`、`mu/alpha` 和 `q` 可以在不改 PT² 算法的前提下恢复；group、Q/K layout 和最终 FP16 weight 也能对齐。
2. **本轮没有观察到 strong PT² 后的正交 residual repair。** 在固定全模型 top-6 / 384-pair 规则下，CEGSP 没有修复 PT² 的 held-out NLL，且 W2 明显恶化。
3. **matched random 没有复现 CEGSP 的 W2 恶化幅度。** 这说明本轮的 CEGSP 选择并不是一个已验证的“有益方向”；同时也不能把它解释成随机扰动带来的收益。

### 不支持的结论

- 不能声称 `PT²+CEGSP` 优于 PT²。
- 不能把 P5-B affine baseline 的正结果外推到真实 PT²；P5-B 是非 PT² 的 diagnostic reference。
- 不能由这一轮否定 CEGSP 在 direct/较弱 affine initialization 上的机制证据，也不能据此宣称 CEGSP 对所有 strong ternary PTQ 都无效。

### 需要保留的风险说明

官方 PT² baseline 在本轮 compact 配置下的 PPL 极高（W2 PPL 约 26114），且 layer 0 Q/K 出现约 `1e3` 量级值。它是本轮真实执行的官方行为，但也意味着该 baseline 的 numerical health 仍是论文主表前必须单列的风险。P5-C 证明了接口可接入，不等于证明该 PT² 配置已经是具有竞争力的 strong baseline。

## 6. 对研究路线的影响

本轮应记录为：

> `STATE_PARITY_PASS; PT2_COMPATIBILITY_FAIL; DO_NOT_GENERALIZE`

它把论文主张边界从“CEGSP 是 strong PTQ 的通用后处理增强”收窄为：CEGSP 在已有 direct/affine ternary initialization 上有机制和整体优化证据，但在当前官方 PT² ATQ state、固定整体 hard relocation 规则和本数值协议下没有 complementary gain。

这不是因为单个负结果而放弃 CEGSP；相反，本轮完成了最重要的强基线兼容性判别，并关闭了“只要把 P5-B 接到 PT² 就会继续提升”的未经验证分支。后续若继续，应优先审计 PT² layer-0 数值健康和官方评测配置的可比性，再设计一个有明确预注册依据的安全残差修复实验；不应围绕本轮结果盲目扫描预算、sign rule 或层子集。
