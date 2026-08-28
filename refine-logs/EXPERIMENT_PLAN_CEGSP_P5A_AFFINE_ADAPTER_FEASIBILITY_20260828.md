# CEGSP P5-A — Affine-Ternary Adapter Feasibility

Date: 2026-08-28

## Purpose

P5-A is a protocol-feasibility experiment, not a strong-baseline performance claim.

The previous P5-0 audit found that the current centered CEGSP formulation cannot be attached directly to PT² because PT² uses an affine ternary codebook:

```math
Q_i = \mu_g + \alpha_g T_i,\qquad T_i\in\{-1,0,+1\}.
```

Therefore P5-A tests whether CEGSP can be redefined in ternary index space while freezing the affine codebook parameters.

## Frozen decisions before running

- Model: `facebook/opt-350m`.
- Hardware target: RTX 4090 24GB.
- Layer: OPT layer 13 only.
- Modules: Q/K only.
- Group size: 128.
- Threshold factor: 0.75, matching PT²-style initialization.
- Budget: 64 relocations, i.e. 128 changed coordinates.
- No QAT teacher.
- No latent FP training.
- No μ/α refit after relocation.
- No budget sweep.

## Variants

1. `affine_fp`: receiver side sign is selected by the affine-relative FP direction:

```math
s_r = \operatorname{sign}(W^{FP}_r-\mu_g).
```

This is the canonical adapter candidate.

2. `grad_best`: receiver side sign is selected by the best legal first-order CE score:

```math
s_r = \arg\max_{s\in\{-1,+1\}} -G_r\alpha_g s.
```

This is a feasibility ablation, not the main method unless later evidence forces a definition change.

3. `random_relocation`: same cardinality-preserving affine relocation, but random donor/receiver/sign.

## Gates

### Gate 1 — Legality

Required:

```math
Q'_i\in\{\mu_g-\alpha_g,\mu_g,\mu_g+\alpha_g\}
```

with frozen `μ, α`, zero illegal states, and zero per-group cardinality violations.

### Gate 2 — Gradient signal

Canonical affine CEGSP should beat random relocation on Wikitext-2 untouched NLL:

```math
L_{\text{affine\_fp}} < L_{\text{random}}.
```

### Gate 3 — Strong-initialization improvement

Canonical affine CEGSP should improve the affine ternary initialization on validation and Wikitext-2 untouched:

```math
L_{\text{affine\_fp}} < L_{\text{affine baseline}}.
```

C4 is recorded but is not a hard gate in P5-A.

## Interpretation

Passing P5-A does not mean CEGSP beats PT². It only means the method can be legally and meaningfully lifted from centered ternary codebooks to affine ternary codebooks. A true PT² compatibility experiment is P5-B and should be run only after P5-A passes and a reliable PT² baseline/state export protocol is fixed.
