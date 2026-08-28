# Experiment Tracker：Ternary Discrete Basin Transport / RTX 4090

| Run ID | Milestone | Purpose | Model / Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| G4090-TDBT-00 | A0 | CUDA/BF16/ternary export/quantized-gradient harness | OPT-125M | finite/invariant/parity/VRAM | MUST | TODO | peak <18 GiB |
| G4090-TDBT-01 | A0 | ternary state patch 与 path metric | OPT-125M | candidate score/barrier/operator metric | MUST | TODO | 只验证实现 |
| G4090-TDBT-02 | A1 | 固定网格 QAT–PTQ gap + FP-FT control | OPT-350M; fit-A/val-B/untouched-C/W | D_QK/D_VO/NLL/trajectory/cost | MUST | TODO | QAT-64/256 |
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
