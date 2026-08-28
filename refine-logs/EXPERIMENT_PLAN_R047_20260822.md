# Experiment Plan R047

**Problem**: R046 shows that a layer-0 hard-T update can improve isolated/local metrics yet split downstream behavior across WikiText2 and C4 in a fully quantized context.

**Method Thesis**: Keep the single-layer validation-gated hard-T constraint, but add a short adjacent-layer trajectory screen before claiming that cross-layer joint constraints are useful.

**Date**: 2026-08-22

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Run |
|---|---|---|---|
| C1: local hard-T is not enough to predict downstream behavior | R045 and R046 already disagree; R047 tests whether adjacent layers explain the mismatch | `hard_l0`, `hard_l1`, and `hard_l0_l1` have different local/joint/NLL signatures under the same full quantized context | R047 |
| C2: cross-layer constraints should be short-window and gated, not full-model optimization | Full-model joint optimization is too heavy and confounds local attribution | The two-layer checkpoint drift and token NLL expose accept/reject signals beyond layer-local NMSE | R047 |

## Compared Systems

| Variant | Meaning |
|---|---|
| `official` | Official ATQ/GPTQ no-SSR baseline, fixed T everywhere |
| `hard_l0` | Validation-gated hard-T only in layer 0, official elsewhere |
| `hard_l1` | Validation-gated hard-T only in layer 1, official elsewhere |
| `hard_l0_l1` | Validation-gated hard-T in layers 0 and 1, official elsewhere |

## Metrics

- Final sequence metrics: `mean_token_nll`, `mean_nll_increase`, `cvar10_nll_increase`, `nonfinite_count`
- Local/joint trajectory metrics: `layer0_nmse`, `layer0_cosine_drift`, `layer1_nmse`, `layer1_cosine_drift`
- Datasets: WikiText2 and C4, first 4 evaluation windows, matched to R046

## Success Criterion

R047 passes as a diagnostic if it distinguishes at least one of the following:

- `hard_l0_l1` is better than both single-layer variants on joint layer-1 drift and downstream NLL on the same dataset.
- `hard_l0_l1` worsens joint drift or downstream NLL despite local improvements, proving that a future R048 needs a hard joint reject gate.
- WikiText2/C4 disagreement aligns with joint drift rather than local layer-0 drift, supporting a multi-distribution gate.

R047 does not by itself claim a final method improvement.

## Next Gate

If R047 shows interpretable adjacent-layer interaction, run R048:

```text
accept hard-T block update iff:
  local layer loss does not exceed epsilon_l
  and two-layer window loss does not exceed epsilon_joint
  and both WikiText2/C4 held-out NLL gates pass
```

If R047 is flat or noisy, do not escalate to full joint optimization; instead test layer windows `(0,1)`, `(10,11)`, `(20,21)`, `(30,31)` with the same screen.
