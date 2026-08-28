# Experiment Tracker：QAT–PTQ Discrete Gauge / RTX 4090

| Run ID | Milestone | Purpose | Model / Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| G4090-00 | M0 | CUDA/BF16/显存/磁盘 preflight | RTX 4090 | system inventory | MUST | TODO | 不下载或运行模型前先审计 |
| G4090-01 | M0 | 16-step ternary QAT harness 与导出 parity | OPT-125M | loss/finite/ternary invariant/VRAM | MUST | TODO | peak <18 GiB |
| G4090-02 | M1 | direct PTQ 与 zero-rate matched PTQ | OPT-350M; C4 calib 128×512 | zero-rate/sign/scale/NMSE | MUST | TODO | mismatch <=0.2 pp |
| G4090-03 | M1 | 固定 256-step short QAT | OPT-350M; C4 fit 1,048,576 tokens | loss/NLL/VRAM/wall-clock | MUST | TODO | BF16, seq512, mb1, ga8 |
| G4090-04 | M2 | 四层 residual support + null + gauge | layers 0/7/15/23; C4/W2 untouched | CMI/z/p/operator NMSE/incremental R2 | MUST | TODO | 200 permutations first |
| G4090-05 | M3 | causal support intervention | selected heads; matched controls | worst-domain operator distortion | MUST-CONDITIONAL | BLOCKED | Only if G4090-04 C1 gate passes |
| G4090-06 | M4 | second-family replication | TinyLlama-1.1B or frozen alternative | same decisive metrics | NICE-CONDITIONAL | BLOCKED | Only if SUPPORT_DIAGNOSTIC |
