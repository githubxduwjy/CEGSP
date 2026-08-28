# Experiment Plan: NC-PTQ

**Problem**: 局部有效的 hard ternary updates 在全量化轨迹中不稳定泛化。  
**Method thesis**: 在 per-layer no-regression feasible set 内优化跨层边界，防止 calibration-specific error cancellation。  
**Date**: 2026-08-24

## Claim Map

| Claim | Minimum convincing evidence | Block |
|---|---|---|
| C1: 无约束 joint ternarization 会利用有害误差相消 | 预定 cancellation index 与 local regression 共同预测 untouched/OOD NLL harm | R054 |
| C2: local trust region 比无约束 joint 更稳定 | 同候选/同步数下 local violation=0，worst-domain NLL 更好，接受率非零 | R055 |
| C3: 方法可以不挑层地量化全模型 | 自动 0–31 层后 W2/C4 PPL 均不退化且至少一者改善 | R056 |

## R054: Cancellation Mechanism Audit

- **Status**: next; R053 has selected the official safe fallback at `(30,31)`.
- **Windows**: all four previously frozen controls `(0,1)`, `(10,11)`, `(20,21)`, `(30,31)`; this reuses the complete depth map and is not further layer search.
- **Systems**: PT² initializer, first-layer hard-T, second-layer hard-T, and their composition. The composition is not described as a joint solver.
- **Data**: unseen windows 72–87 with an 8/8 split; WikiText2 and C4 are analyzed separately and jointly. No final PPL set is used for selection.
- **Metrics**: per-layer normalized local error, window boundary error, `C_S`, mean/CVaR token NLL, nonfinite count.
- **Gate**: the preregistered cancellation-risk score must beat boundary error alone on both splits under the exact thresholds in `EXPERIMENT_PLAN_R054_20260824.md`. Null or low-prevalence results stop NC-PTQ without tuning.
- **Cost**: 2–4 RTX 4090 GPU-hours.

## R055: Constrained Solver Isolation

- **Systems**: initializer / independent hard-T / unconstrained joint / NC-PTQ.
- **Fairness**: same initializer, blocksize, calibration tokens, hard-flip proposal set, coordinate passes, max steps and final grid refit. Report extra constraint-check wall time separately.
- **Data split**: fit generates flips; validation early-stops/rolls back; untouched test evaluates. No reuse across roles.
- **Primary metrics**: worst-domain mean NLL, worst-domain CVaR10, local violation rate, accepted flip ratio.
- **Success**: local violation=0; accepted flip ratio materially above zero; worst-domain untouched NLL lower than unconstrained joint on both frozen controls.
- **Failure**: if accepted ratio is near zero, label safe-but-vacuous; do not tune epsilon. If unconstrained joint is equally robust, the constraint has no method value.
- **Cost**: 4–8 GPU-hours.

## R056: Full-Model Automatic Quantization

- **Entry condition**: R055 passes.
- **Run**: fixed non-overlapping `w=2` windows from layer 0 to 31 in quantized-prefix context. No layer ranking or mask.
- **Baselines**: PT² matched budget; CAT-Q-style unconstrained sliding reconstruction; unconstrained joint hard-T.
- **Metrics**: WikiText2/C4 PPL, zero-shot average, accepted flip/window ratio, actual bpw including metadata, PTQ time, peak VRAM.
- **Gate**: W2 and C4 PPL both non-regressing vs matched PT² and at least one improves. Only then expand to three seeds and another model family.
- **Cost**: 8–16 GPU-hours.

## Intentionally Cut

- remaining-layer enumeration after R053;
- Fisher/K-FAC ranker unless exact path checks later prove the bottleneck is computational;
- projection masks, rotation competition, epsilon sweep;
- W1.58A4, kernel and mixed precision before the weight-only method passes.

## Main Risks

1. Error cancellation may not explain the observed distribution split; R054 directly falsifies it.
2. Zero local regression may eliminate useful compensation; R055 reports coverage and compares the unconstrained frontier.
3. PTQ cost may approach QAT; R056 must report time and VRAM against PT²/CAT-Q.
