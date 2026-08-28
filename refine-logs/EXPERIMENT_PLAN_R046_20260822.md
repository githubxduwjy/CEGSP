# R046 Experiment Plan: Quantized-Context Sequence Gate

## Objective

Test whether the R044 distribution split is visible when the layer-0 hard-`T`
candidate is scored inside the matched full-model quantized context, rather than
inside an otherwise FP16 model.

## Candidates

- `official_full_context`: official no-SSR ATQ/GPTQ on all layers.
- `hard_t_layer0_full_context`: validation-gated hard-`T` only on layer 0; all
  later layers use the same official no-SSR ATQ/GPTQ path.

Both use LLaMA-2-7B, WikiText2 calibration with 8 samples, seed 0, group size
128, and the same layerwise evaluator. Scoring uses the first 4 full 2048-token
windows from WikiText2 test and C4 validation.

## Gate

This is a contextual diagnostic, not a new method claim. It passes only if a
predeclared sequence metric rejects the hard-`T` candidate on every scored
sequence. If it does not pass but reproduces the WikiText2/C4 direction split,
the next method should treat calibration distribution and cross-distribution
robustness as part of the acceptance rule.
