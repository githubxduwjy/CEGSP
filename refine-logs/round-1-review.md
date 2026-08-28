# Round 1 Review

## Scores

| Dimension | Score |
|---|---:|
| Problem Fidelity | 7 |
| Method Specificity | 6 |
| Contribution Quality | 5 |
| Frontier Leverage | 5 |
| Feasibility | 5 |
| Validation Focus | 6 |
| Venue Readiness | 4 |

**Overall**: 5.5/10  
**Verdict**: REVISE

## Main critique

The proposal correctly targets the experimentally observed local-to-global mismatch, but the Fisher ranker plus CVaR commit rule reads as a safety wrapper around existing ternary proposal generation. CAT-Q already provides sliding-layer reconstruction; cross-layer error compensation already jointly optimizes discrete codes; KronQ already introduces output-gradient covariance. The proposal needs a sharper mechanism than combining these ingredients.

## Required changes

1. Remove Fisher ranking as the dominant contribution unless an approximation guarantee and clear scalability advantage can be established.
2. Separate data used for mechanism diagnosis, commit/early stopping, and final evaluation.
3. Explain why the method is not simply automated layer selection.
4. Compare against unconstrained cross-layer reconstruction and global error compensation, not only layer-local PT².
5. Report acceptance/coverage and PTQ overhead; an algorithm that rolls everything back is safe but not useful.

## Resolution direction

Reframe the method around a **no-cancellation constraint**: cross-layer optimization may improve a window boundary by making errors from adjacent layers cancel on calibration samples, while damaging an individual layer map and failing off-domain. Joint optimization should therefore minimize the downstream/window objective subject to every layer-local error remaining no worse than its strong ternary initializer. This directly implements the user's requested cross-layer-plus-local structure and distinguishes the method from unconstrained sliding/global reconstruction.
