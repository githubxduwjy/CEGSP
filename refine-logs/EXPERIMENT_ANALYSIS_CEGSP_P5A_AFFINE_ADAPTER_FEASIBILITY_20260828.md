# CEGSP P5-A — Affine-Ternary Adapter Feasibility Report

Date: 2026-08-28

Run id: `cegsp_p5a_affine_adapter_opt350m_20260828`

Raw artifacts:

- Result JSON: `/home/x1shan/文档/ChatGPT/PTQ_paper/results/remote-runs/cegsp_p5a_affine_adapter_opt350m_20260828/p5a_affine_adapter_result.json`
- Log: `/home/x1shan/文档/ChatGPT/PTQ_paper/results/remote-runs/cegsp_p5a_affine_adapter_opt350m_20260828/cegsp_p5a_affine_adapter_opt350m_20260828.log`

## 1. What this experiment tested

P5-A tested whether CEGSP can be lifted from a centered ternary codebook

```math
Q_i = \alpha_g s_i z_i
```

to a PT²-style affine ternary codebook

```math
Q_i = \mu_g + \alpha_g T_i,\qquad T_i\in\{-1,0,+1\},
```

while freezing `μ_g` and `α_g`.

This was a protocol-feasibility experiment. It does not claim that CEGSP beats PT², because this run constructed a PT²-style affine ternary initialization from FP weights rather than using a fully reproduced PT² checkpoint.

## 2. Fixed protocol

- Model: `facebook/opt-350m`
- Layer: 13 only
- Modules: Q/K only
- Sequence length: 128
- Batch size: 2
- Fit / validation / untouched W2 / untouched C4 batches: 8 / 8 / 8 / 8
- Group size: 128
- Threshold factor: 0.75
- Budget: 64 relocations = 128 changed coordinates
- Gradient batches: 1
- No QAT teacher
- No latent FP training
- No post-relocation `μ/α` refit

## 3. Main results

| Method | Val NLL | W2 untouched NLL | C4 untouched NLL | Δ Val | Δ W2 | Δ C4 |
|---|---:|---:|---:|---:|---:|---:|
| FP reference | 3.803895 | 3.987563 | 3.438045 | — | — | — |
| Affine ternary baseline | 3.812594 | 3.995015 | 3.448459 | 0 | 0 | 0 |
| Affine CEGSP, affine-relative FP sign | 3.797204 | 3.985037 | 3.445810 | -0.015390 | -0.009978 | -0.002649 |
| Affine CEGSP, gradient-best sign | 3.799891 | 3.987532 | 3.445091 | -0.012703 | -0.007482 | -0.003367 |
| Random affine relocation | 3.813878 | 3.994545 | 3.448281 | +0.001284 | -0.000470 | -0.000178 |

The canonical adapter, `affine_fp`, improves validation, W2 untouched, and C4 untouched NLL relative to the affine ternary baseline. It also clearly improves over the random affine relocation control on validation and W2, while C4 improvement is small but still positive relative to baseline.

## 4. Protocol gates

| Gate | Result | Evidence |
|---|---|---|
| Gate 1: legality | PASS | illegal states = 0; max codebook residual = 0; cardinality violations = 0 |
| Gate 2: gradient signal | PASS | W2 NLL: affine_fp 3.985037 < random 3.994545 |
| Gate 3: strong-initialization improvement | PASS | affine_fp improves both val and W2 over affine baseline |

Additional audit values:

- Baseline illegal states: 0
- Baseline max codebook residual: 0.0
- Affine CEGSP illegal states: 0
- Affine CEGSP cardinality violations: 0
- First-order score identity max error: 0.0
- Changed coordinates: 128
- Elapsed time: 71.41 seconds
- Peak memory: 0.88 GB reported by PyTorch allocator

## 5. Interpretation

P5-A is a meaningful positive result.

It shows that the P5-0 incompatibility finding does not kill CEGSP. Instead, the method can be redefined more generally in ternary index space:

```math
T_i\in\{-1,0,+1\},\qquad Q_i=\mu_g+\alpha_gT_i.
```

Under frozen affine codebook parameters, support relocation remains legal and its CE-gradient score remains exactly consistent with the actual affine weight displacement. The `μ_g` term cancels from the relocation displacement, so the same first-order logic can operate on the side/center index state.

The receiver sign result is also useful: the canonical affine-relative FP sign slightly outperforms gradient-best on validation and W2, while gradient-best is slightly better on C4. Since the canonical rule already passes the main gates, there is no need to redefine the method around gradient-best sign at this stage.

## 6. What this does not prove

This experiment does not prove:

- CEGSP improves a real PT² checkpoint.
- CEGSP beats PT² or any latest ternary PTQ baseline.
- The effect is stable across layers, offsets, seeds, or model families.
- The affine adapter solves the unresolved PT² reproduction issue.

The correct claim after P5-A is narrower:

> CEGSP can be legally and beneficially applied inside a frozen affine ternary codebook on a small OPT-350M feasibility setting.

## 7. Recommended next experiment

The next experiment should be P5-B0, not full P5-B.

P5-B0 should keep the same affine adapter, but expand from one layer to a small fixed layer set, such as layers `6,13,20`, still on OPT-350M and still without PT² checkpoint dependence. Its purpose is to test whether the affine-index formulation is a one-layer artifact.

Suggested fixed design:

- Model: `facebook/opt-350m`
- Layers: `6,13,20`
- Modules: Q/K
- Budget: 64 relocations per layer, frozen before untouched evaluation
- Sign rule: canonical `affine_fp`
- Controls: affine baseline and random affine relocation only
- Metrics: validation, W2 untouched, C4 untouched NLL/PPL
- Gate: legality must pass; at least 2/3 layers or the joint three-layer patch should improve W2 over affine baseline; random should not match the improvement.

Only after P5-B0 passes should we spend effort on true PT² checkpoint compatibility. Before that, running PT²+CEGSP would be premature because the PT² baseline/state export protocol is still unresolved.
