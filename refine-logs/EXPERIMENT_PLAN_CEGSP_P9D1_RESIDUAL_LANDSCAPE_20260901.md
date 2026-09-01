# P9-D1: Ordinary-Affine vs PT² Residual Discrete Landscape

**Date:** 2026-09-01
**Status:** pre-registered for one A100 diagnostic run
**Branch:** P9 strong-initializer compatibility; independent of the frozen R014–R058 branch and the G4090 QAT–PTQ gauge branch

## Scientific question

Does the same frozen TernRefine/CEGSP local scoring rule see a different
single-relocation landscape after ordinary affine ternarization and after the
healthy official PT² ATQ+SSR initializer?

The experiment measures single legal moves only. It is deliberately not a
full-patch performance test and is not a budget, threshold, sign-rule, or
epsilon search.

## Frozen comparison

| Item | Fixed choice |
|---|---|
| Model / device | Llama-2-7B; one NVIDIA A100 |
| Initializer A | ordinary affine ternary, group size 128, threshold factor 0.75, original column order |
| Initializer B | the existing detached P9-S2 official PT² ATQ+SSR sidecar |
| Scope | all 32 decoder layers; Q/K projections only (64 modules) |
| Calibration gradient | one Wikitext train batch; exactly one backward; only Q/K require gradients |
| Candidate signal | `S_m = -<G, ΔQ_m>` using the unchanged frozen CEGSP legal relocation rule |
| Ranked sample | ranks `{1, 2, 4, 8, 16, 32, 64, 128}` from each layer's combined Q/K ranking |
| Random control | 8 deterministic random legal moves per layer, same count, separate fixed seed |
| Evaluation | each candidate is applied alone, then restored; fixed validation slice and untouched Wikitext-2 test slice; 2 batches each, sequence length 128, batch size 1 |
| Selection | no validation or untouched-test selection; all candidates are predeclared by fit-gradient rank or deterministic random draw |
| Scale / state | `mu`, `alpha`, T, and SSR are frozen; one support exchange changes exactly two ternary coordinates and preserves per-group cardinality |
| C4 | omitted from candidate-level evaluation; this is a compact landscape diagnosis, not a paper PPL table |

## Measured quantities

For each initializer and each candidate, save its complete identity, sampling
label, predicted score, validation NLL delta, and untouched W2 NLL delta. The
primary summary is:

\[
\rho_D=\operatorname{Spearman}\bigl(S_m,-\Delta L_m^D\bigr),
\qquad D\in\{\mathrm{val},\mathrm{W2}\}.
\]

Also report positive-move density for all candidates, rank-1 candidates, the
eight ranked samples, and random controls; mean delta for rank-1, ranked
samples, and random controls; and the fixed initializer baseline NLLs.

Expected sample count is `32 × (8 ranked + 8 random) = 512` per initializer,
`1024` total. The run is valid only if both initializers produce the complete
sample count, all metrics are finite, all moves are legal, and no test split is
used for candidate generation or selection.

## Interpretation gate (no post-hoc protocol change)

* **QGP boundary:** PT² substantially lowers the pre-registered Spearman
  correlations relative to ordinary affine. This supports a claim that the
  first-order quantized-point ranking becomes less predictive after optimized
  PT².
* **Residual depletion:** PT² retains usable correlation, but has much lower
  positive-move density and much smaller beneficial ranked effects. This
  supports a claim that strong PTQ leaves less actionable residual in the
  current move space.
* **Composition boundary:** only if PT² single moves remain clearly beneficial
  and predictive while the earlier 384-relocation full patch is mixed, a single
  conditional P9-D2 frozen-prefix composition diagnostic may be considered.
* **No clear branch:** if neither contrast is separable, record the negative
  result and close P9 without tuning the canonical CEGSP rule.

The diagnostic cannot change the canonical 64-relocation rule, add new
initializers, search projection masks, or tune epsilon.

## Integrity checklist

- [ ] P9-S2 sidecar is loaded from disk; PT² is not rerun.
- [ ] Ordinary affine and PT² differ only in initializer.
- [ ] Ranked candidates use the fit gradient only.
- [ ] Validation and Wikitext-2 test are evaluation-only.
- [ ] Every candidate identity and score is saved.
- [ ] No nonfinite values, illegal ternary states, or cardinality changes.
- [ ] Raw JSON and screen log are retained for analysis.
