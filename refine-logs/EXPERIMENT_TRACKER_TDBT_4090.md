# Experiment Tracker：Ternary Discrete Basin Transport / RTX 4090

| Run ID | Milestone | Purpose | Model / Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| G4090-TDBT-00 | A0 | CUDA/BF16/ternary export/quantized-gradient harness | OPT-125M | finite/invariant/parity/VRAM | MUST | PASS | 42137 / screen `tdbt_G4090_TDBT_00`; W2 train/validation, 32/16 batches, seq 128; FP NLL 9.2227, direct ternary PTQ NLL 9.8320; no nonfinite. |
| G4090-TDBT-01 | A0 | ternary state patch 与 path metric | OPT-125M/350M | candidate score/barrier/operator metric | MUST | PASS_DIAGNOSTIC | BF16 125M was resolution-limited. FP32 125M: 3/8 reductions, mean direct 0.001276 vs bridge 0.000669. FP32 350M layer-0 q_proj: 7/8 reductions, mean direct 2.051e-5 vs bridge 1.039e-5; all finite. Mechanism replicated, endpoint/task benefit not yet shown. |
| G4090-TDBT-02 | A1 | 固定网格 QAT–PTQ gap + FP-FT control | OPT-350M; fit-A/val-B/untouched-C/W | D_QK/D_VO/NLL/trajectory/cost | MUST | DIAGNOSTIC_PASS | 125M: PTQ 9.8320; QAT-256 7.3027; FP-FT 6.4785 then PTQ 9.6406. Clean 350M report (`G4090-TDBT-02-OPT350M-FIXEDREPORT2`): FP 9.4180; PTQ 9.3477; QAT 8.4082; `gap_closed_fraction=null`. FP-FT 350M: 7.3848 then PTQ 9.0664. QAT beats FP-FT→PTQ by 0.6582 NLL, but no gap-closure claim; operator metrics/untouched C4 remain pending. |
| G4090-TDBT-02-TRAJ-OPT125M | A1b | Li et al. mechanism audit: deployed-vs-latent QAT trajectory | OPT-125M; W2 train/validation | f(Q(w_k)), f(w_k), distance, state transitions | MUST | PASS_DIAGNOSTIC | Correct next-token CE: FP holdout 4.0347; direct PTQ 9.6975; QAT deployed 9.6389→5.4503 while latent 4.0347→4.3639. Steps 1–64 are dominated by 0↔nonzero support changes; sign flips only 21 at 128 and 402 at 256. Strongly matches quantization-compatibility correction, not a zero-mediated sign-flip requirement. |
| G4090-TDBT-02-ALIGNED-OPT125M | A1c | Corrected next-token QAT/PTQ gap | OPT-125M; W2 train/validation | FP/PTQ/QAT NLL | MUST | PASS | Explicit next-token CE: FP 4.0347; PTQ 9.6975; QAT-256 5.4503; gap closure 0.7500. |
| G4090-TDBT-02-ALIGNED-FPFT-OPT125M | A1c | Corrected equal-budget FP-FT→PTQ control | OPT-125M; W2 train/validation | FP-FT/PTQ NLL | MUST | PASS | Explicit next-token CE: FP-FT 3.8926; after PTQ 9.6225. QAT is 4.1722 NLL better than FP-FT→PTQ under identical 256-step budget/grid. |
| G4090-TDBT-03 | B1 | endpoint/one-shot/path feasibility | OPT-350M; layers 0/7/15/23 | endpoint loss/barrier/closure | CONDITIONAL | BLOCKED | 仅 A1 通过 |
| G4090-TDBT-04 | B2 | barrier、M/S、zero-mediated 消融 | OPT-350M; Q/K,V/O | path/operator/NLL/transition stats | CONDITIONAL | BLOCKED | matched candidate budget |
| G4090-TDBT-05 | C1 | TDBT-F/TDBT-G held-out | OPT-350M; C4/W2 untouched | gap closure/NLL/CVaR/VRAM/time | CONDITIONAL | BLOCKED | 仅 B1/B2 通过 |
| G4090-TDBT-06 | C2 | seed/second-family replication | TinyLlama-1.1B or 300–500M | same decisive metrics | NICE | BLOCKED | 仅 C1 paper-worthy |

## 预注册 gate

- A1：QAT-256 相对 direct PTQ 在至少 3/4 层 operator distortion 改善 >=5%，且 bootstrap CI 不跨 0；FP-FT + PTQ 不能完全解释收益。
- B1：TDBT 相对 endpoint-only 在至少 3/4 层 barrier 或 held-out operator distortion 改善 >=5%。
- B2：若 zero-mediated 与 direct sign-flip 无差异，关闭 zero-channel claim；若 TDBT 与 QG-one-shot 无差异，关闭 path-planning claim。
- C1：gap closure >=0.30 为 diagnostic success；>=0.50、C4/W2 NLL 不退化且耗时 <= direct PTQ 3 倍，才是 paper-worthy。
- 任何 OOM、nonfinite、split leak、export mismatch 标记 `INVALID_HARNESS`，不修改研究 gate。
