# R044a Plan: Local-to-Global Accumulation Diagnosis

R043a changed only 4/32 layers and improved C4 PPL from 66.7370 to 56.0225, but WikiText2 became NaN. All 1,112 refined blocks had finite grid solutions and zero fallback rows, so the failure is not explained by degenerate closed-form fitting. The central unresolved question is whether a single early refined layer is already unsafe, or individually useful updates become unsafe only through cross-layer accumulation.

R044a applies the identical validation-gated `T` update only to layer 0; all other layers use official ATQ. Model, WikiText2 calibration samples, 6/2 fit-validation split, seed, block size, search budget, GPTQ propagation, and W2/C4 evaluation remain unchanged.

Interpretation is pre-registered:

- If layer-0-only is finite and improves both official no-SSR PPL values (25.8104/66.7370), the block mechanism transfers locally and R043a is evidence of an accumulation/error-budget failure. The next algorithmic step is a layer-level validation trust region, not projection enumeration.
- If layer-0-only is non-finite or worsens either dataset, block-local validation NMSE is not a sufficient transfer criterion even for one layer; stop the current hard-`T` update and redesign the objective before further PPL runs.

This is a single causal attribution test, not a layer sweep.

