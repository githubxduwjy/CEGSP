# CEGSP 论文级实验 Tracker

日期：2026-08-27

原则：本 tracker 只承接 CEGSP 论文级闭环；不再按单个小模块无限开实验。所有 run 在启动前固定 split、预算、gate 和解释规则。

| Run group | Block | 目的 | 主要系统 | 模型/数据 | 关键指标 | 优先级 | 状态 |
|---|---|---|---|---|---|---|---|
| CEGSP-10A | B1 | 非 OPT 跨架构迁移 | direct / support / signflip / joint | Pythia-1B; W2+C4 | untouched NLL, adapter, cost | MUST | PASS_CROSS_ARCH |
| CEGSP-11A-AUDIT | B0 | direct/strong PTQ 可复现性与组合接口审计 | direct ternary + PT² ATQ + PT-BitNet interface | OPT-350M, Pythia-1B | parity, finite, flags, timing | MUST | TODO |
| CEGSP-11B-MAIN-MATRIX | B1 | 论文主表：规模×架构×数据与 2×2 组合 | direct / strong PTQ / CEGSP / strong+CEGSP | OPT-1.3B, OPT-2.7B, Pythia-1B | W2/C4 NLL, paired CI, PPL, cost | MUST | TODO |
| CEGSP-11C-GAP-COST | B2 | 实测 PTQ–QAT gap 与 closure | direct / QAT reference / CEGSP | OPT-350M + Pythia-410M/1B | gap closure, wall-clock, VRAM | MUST | TODO |
| CEGSP-11D-MECHANISM-COMPACT | B3 | 一次性完成三值特异性和简洁性 | support / signflip / joint / random / deletion | OPT-350M + Pythia-1B | matched holdout NLL, CI | MUST | TODO |
| CEGSP-11E-GENERALIZATION | B4 | split robustness 与连续分数下游 | frozen canonical CEGSP | Pythia 新 offset + OPT 新 offset | W2/C4 + 2 downstream tasks | MUST | TODO |
| CEGSP-11F-APPENDIX | B5 | 第二非 OPT family或更多结构约束 | 只在主闭环通过后考虑 | 资源允许时 | appendix metrics | NICE | BLOCKED_ON_MAIN |

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
