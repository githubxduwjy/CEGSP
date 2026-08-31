# CEGSP 论文级实验 Tracker

日期：2026-08-27

原则：本 tracker 只承接 CEGSP 论文级闭环；不再按单个小模块无限开实验。所有 run 在启动前固定 split、预算、gate 和解释规则。

| Run group | Block | 目的 | 主要系统 | 模型/数据 | 关键指标 | 优先级 | 状态 |
|---|---|---|---|---|---|---|---|
| CEGSP-10A | B1 | 非 OPT 跨架构迁移 | direct / support / signflip / joint | Pythia-1B; W2+C4 | untouched NLL, adapter, cost | MUST | PASS_CROSS_ARCH |
| CEGSP-11A-AUDIT | B0 | direct/strong PTQ 可复现性与组合接口审计 | direct ternary + PT² ATQ/ATQ+SSR official adapter | OPT-350M | parity, finite, flags, timing | MUST | BASELINE_PROTOCOL_MISMATCH |
| CEGSP-11B-MAIN-MATRIX | B1 | 论文主表：规模×架构×数据与 2×2 组合 | direct / strong PTQ / CEGSP / strong+CEGSP | OPT-1.3B, OPT-2.7B, Pythia-1B | W2/C4 NLL, paired CI, PPL, cost | MUST | TODO |
| CEGSP-11C-GAP-COST | B2 | 实测 PTQ–QAT gap 与 closure | direct / QAT reference / CEGSP | OPT-350M + Pythia-410M/1B | gap closure, wall-clock, VRAM | MUST | TODO |
| CEGSP-11D-MECHANISM-COMPACT | B3 | 一次性完成三值特异性和简洁性 | support / signflip / joint / random / deletion | OPT-350M + Pythia-1B | matched holdout NLL, CI | MUST | TODO |
| CEGSP-11E-GENERALIZATION | B4 | split robustness 与连续分数下游 | frozen canonical CEGSP | Pythia 新 offset + OPT 新 offset | W2/C4 + 2 downstream tasks | MUST | TODO |
| CEGSP-11F-APPENDIX | B5 | 第二非 OPT family或更多结构约束 | 只在主闭环通过后考虑 | 资源允许时 | appendix metrics | NICE | BLOCKED_ON_MAIN |
| CEGSP-V2-P0 | V2-P0 | 量化点 CE 梯度机制边界：CEGSP vs One-Step/4-Step QAT + score-validity | direct / CEGSP / latent QAT controls | OPT-125M; W2 | NLL, score-validity, gap ratio, cost | MUST | COMPLETE_POSITIVE_BUT_QAT_STRONGER |
| CEGSP-V2-P1A | V2-P1 | move-space 对照：support/signflip/mixed/matched/random，验证 top-k 与混合动作 | support / signflip / mixed / random | OPT-125M; W2 | NLL, matched controls, random controls | MUST | COMPLETE_MIXED_TOP6_BEST |
| CEGSP-V2-P1B | V2-P1 | top-k saturation：检查 Q/K 编辑是否在 top-6 饱和 | support / signflip / mixed | OPT-125M; W2 | top-k NLL curve | MUST | COMPLETE_NO_SATURATION_TOP12_BEST |
| CEGSP-V2-P1C | V2-P1 | OPT-350M top-k saturation：验证 top-k 规律是否随模型扩大保持 | support / signflip / mixed | OPT-350M; W2 | top-k NLL curve | MUST | COMPLETE_TOP6_BEST_OVEREDIT_AFTER_12 |
| CEGSP-V2-P2A | V2-P2 | OPT-350M 固定 top-6 大 holdout + C4 transfer | support / signflip / mixed / random | OPT-350M; W2+C4 | large untouched NLL, C4 transfer | MUST | COMPLETE_TOP6_TRANSFER_PASS |
| CEGSP-V2-P2B | V2-P2 | OPT-350M top-6 offset robustness + C4 transfer | support / signflip / mixed / random | OPT-350M; W2+C4 | shifted-offset holdout NLL | MUST | COMPLETE_OFFSET_ROBUSTNESS_PASS |
| CEGSP-V2-P2C2 | V2-P2 | 非 OPT 跨架构 adapter 验证 | support / signflip / joint / random | Pythia-1B; W2 | GPT-NeoX adapter, untouched NLL, random controls | MUST | COMPLETE_CROSS_ARCH_SUPPORT_DOMINANCE |
| CEGSP-V2-P3A | V2-P3 | canonical fixed-rule 新 offset 验证 | support primary / fixed top6 / random controls | OPT-350M; W2+C4 | W2/C4 untouched NLL, fixed rule gate | MUST | STRONG_PASS_W2_C4 |
| CEGSP-V2-P3B | V2-P3 | canonical fixed-rule 新 offset 跨架构验证 | support primary / fixed top4 / random controls | Pythia-1B; W2 | W2 untouched NLL, adapter, random controls | MUST | STRONG_PASS_W2 |
| CEGSP-V2-P4 | V2-P4 | PTQ-QAT gap/cost 闭环 | direct / CEGSP / one-step QAT / 10,50-step QAT | OPT-350M; W2+C4 | gap closure, PPL/NLL, wall-clock, score-validity | MUST | PASS_GAP_COST_WEAK_ONE_STEP_QAT |
| CEGSP-V2-P4R | V2-P4R | QAT baseline 修复：transition audit + edit-matched one-step | CEGSP / one-step dense eta / edit-matched / 5,10,20,50-step QAT | OPT-350M; W2+C4 | transition counts, edit-matched QAT, gap/cost | MUST | STRONG_PASS_QAT_TRANSITION_EDIT_MATCHED |
| CEGSP-V2-P4R2 | V2-P4R2 | 公平性补洞：严格 edit match + QAT update scope | CEGSP / one-step all-QK / one-step selected-QK | OPT-350M; W2+C4 | strict edit match, scope fairness | MUST | STRONG_PASS_STRICT_EDITMATCH_SCOPE |
| CEGSP-P7-R | P7-R | 大模型扩大留出与跨域稳健性 | affine baseline / frozen affine CEGSP top-6 / matched random | Llama-2-7B + Qwen3-8B; W2+C4; A100 | held-out NLL, legality, finite, cross-domain transfer | MUST | STRONG_PASS_LARGE_MODEL_CROSS_DOMAIN |
| CEGSP-P8-A | P8-A | 下游连续分数筛查 | BF16 / affine ternary / affine+CEGSP | Llama-2-7B + Qwen3-8B; PIQA + ARC-Easy | normalized choice LL, accuracy, margin | MUST | PREPARED_BLOCKED_REMOTE_ENDPOINT |
| CEGSP-P9-S0 | P9-S0 | 官方 PT² Llama-2-7B 数值健康与 checkpoint 审计 | PT² ATQ+SSR only, no CEGSP | Llama-2-7B; W2+C4; A100 | official PPL, finite saved state, checkpoint existence | MUST | PT2_LLAMA7B_HEALTH_PASS_CANDIDATE |
| CEGSP-P5-0 | P5-0 | Strong PTQ protocol audit | PT² codebook/scale/support/eval audit | PT² official 9e943e6 | feasible-space compatibility | MUST | BLOCK_CURRENT_CENTERED_CEGSP_ON_PT2 |
| CEGSP-P5-A | P5-A | 仿射三值 index-space adapter 合法性与最小机制信号 | affine ternary baseline / affine CEGSP / random affine relocation | OPT-350M L13 Q/K; W2+C4 | legality, score identity, val/W2/C4 NLL | MUST | PASS_AFFINE_ADAPTER_FEASIBILITY |
| CEGSP-P5-B | P5-B | 全候选 Q/K 空间上的整体 affine CEGSP compatibility | affine baseline / overall affine CEGSP top-4/top-6 / matched random | OPT-350M all 24 layers; W2+C4; RTX 4090 | legality, finite, val/W2/C4 NLL, fixed global ranking | MUST | PASS_OVERALL_AFFINE_COMPATIBILITY |
| CEGSP-P5-C | P5-C | 真实 PT² official state/export parity 与 affine CEGSP compatibility | PT² ATQ / PT²+affine CEGSP / matched random；P5-B 仅 diagnostic | OPT-350M all 24 layers; W2+C4; RTX 4090 | state parity, legality, finite, val/W2/C4 NLL, matched edits | MUST | PARITY_PASS_COMPATIBILITY_FAIL |
| CEGSP-P5-C0 | P5-C0 | 官方 PT² 数值健康、官方/compact evaluator parity 与 layerwise audit | clean FP16 / PT² ATQ / PT² ATQ+SSR | OPT-350M; W2+C4; RTX 4090 | official/compact PPL-NLL, 144 modules, 1728 blocks, q outliers, output reconstruction | MUST | NUMERICAL_HEALTH_FAIL_PT2_NOT_MAINLINE |
| CEGSP-P6-A | P6-A | 全 24 层 centered/affine ternary 的量化点 CE score-validity | centered / affine gradient-ranked relocation / matched random | OPT-350M; W2 fit/val/untouched + C4 baseline; RTX 4090 | Spearman score-vs-held-out Delta NLL, fixed bins, top-20% vs random, finite audit | MUST | PASS_CROSS_REPRESENTATION_SCORE_VALIDITY |
| CEGSP-P6-B | P6-B | 固定协议下的 seed/offset replication，检验 score-validity 与 affine split bias 稳健性 | centered / affine gradient-ranked relocation / matched random | OPT-350M; 3 fixed seed/offset replicates; RTX 4090 | rho, Delta_rank, top-20% vs random, score-bin trend, finite audit | MUST | STABLE_SUPPORT_SCORE_VALIDITY |

## 统一 gate

- 强基线 gate：至少在一个共同 OPT 模型上复现 PT² ATQ（或明确记录 `baseline-reproduction-failed`）；在此之前不能声称 CEGSP 优于最新三值 PTQ。
- B1：至少 4/6 model×offset cells 在 Wikitext 和 C4 同时改善，且无单模型双 holdout 明显恶化。
- 组合 gate：若 CEGSP standalone 不超过强基线，但 strong+CEGSP 相对 strong 有稳定增益，则保留“quantized-point function repair layer/complement”论文定位；若组合也无增益，则不得把 direct baseline 的改善写成强竞争性结论。
- B2：QAT reference 确实优于 direct；CEGSP closure 为正，目标至少 25%；成本仍为 PTQ 级别。
- B3：CE joint 优于 random；same-layer support 至少在多数核心 cell 不弱于 signflip；deletion 不证明需要更复杂迭代。
- B4：多数新 offset 不恶化；至少一个连续分数下游任务改善或不劣。

## 运行纪律

- 不使用 untouched test 选择 k、offset、threshold 或报告指标。
- 强基线必须使用相同模型/tokenizer、calibration token、split、W1.58A16 weight-only 口径，并报告 asymmetric codebook、outlier/residual、mixed precision 与有效 bpw；不可比配置单列，不混入主表。
- 不做 projection mask 枚举，不事后调 epsilon，不增加 QAT teacher。
- 单个负结果只能标为 `module-negative` 或 `split-sensitive`，不能触发方向切换。
- 只有第 13 节止损条件满足时，才重新评估主方法。
