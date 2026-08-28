# Experiment Tracker：Ternary QAT–PTQ Gap / RTX 4090

| Run ID | Milestone | Purpose | Model / Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| G4090-GAP-00 | A0 | CUDA/BF16/三值导出/显存 preflight | OPT-125M | finite/invariant/export parity/VRAM | MUST | TODO | peak <18 GiB |
| G4090-GAP-01 | A0 | direct PTQ 与 16-step QAT harness | OPT-125M | loss/finite/ternary invariant/VRAM | MUST | TODO | 只验证链路 |
| G4090-GAP-02 | A1 | 固定预算证明 QAT–PTQ gap | OPT-350M; fit-A/val-B/untouched-C/W | D_QK/D_VO/NLL/zero-rate/wall-clock | MUST | TODO | PTQ vs QAT-64/256 |
| G4090-GAP-03 | A2 | binary/ternary/4-level specificity control | OPT-350M; 2 layers | NormGap_b/operator distortion | CONDITIONAL | BLOCKED | 仅 A1 通过后 |
| G4090-GAP-04 | B1 | M/S/alpha oracle attribution | OPT-350M; layers 0/7/15/23 | closure_M/S/alpha/joint | CONDITIONAL | BLOCKED | 诊断 oracle，不是方法 |
| G4090-GAP-05 | B2 | salience-preserving path null | OPT-350M; Q/K,V/O | A/C path stats/D_QK/D_VO | CONDITIONAL | BLOCKED | 200 permutations first |
| G4090-GAP-06 | C1 | 无 teacher Ternary Path Projection PTQ | OPT-350M; 4 layers | gap closure/NLL/cost/VRAM | CONDITIONAL | BLOCKED | 仅 B 通过后 |
| G4090-GAP-07 | C2 | 32-step teacher-assisted cost control | OPT-350M; 4 layers | closure vs wall-clock | CONDITIONAL | BLOCKED | 仅无 teacher bridge失败且有 state-margin证据 |
| G4090-GAP-08 | D0 | second-family replication | TinyLlama-1.1B | same decisive metrics | NICE | BLOCKED | 仅 C2 paper-worthy |

## 预注册总门槛

- A1：至少 3/4 层 operator gap >=5%，bootstrap 95% CI 不跨 0，zero-rate mismatch <=0.2 pp。
- A2：若 ternary `NormGap` 不明显高于 binary/4-level，关闭“三值专属”主张。
- B：M/S/path null 在 salience matching 后仍能解释 gap，才允许运行 C1。
- C1：无 teacher bridge 至少关闭 30% gap 才算 diagnostic success，关闭 50% 且 C4/W2 NLL 不退化才算 paper-worthy。
- 任何 OOM、nonfinite、split leak、导出不一致只标为 `INVALID_HARNESS`，不得修改研究 gate。
