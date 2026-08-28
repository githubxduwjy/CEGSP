# CEGSP-05B：random-control NLL baseline

## Motivation

CEGSP-05A confirms that CE-gradient top-k edits improve larger untouched W2/C4 samples on OPT-350M and OPT-125M. The remaining control risk is that any small ternary edit plus validation selection might improve direct ternary PTQ. CEGSP-05B adds random support/signflip controls with the same edit budget and the same validation-based layer selection.

## Fixed settings

- Models: `facebook/opt-350m`, `facebook/opt-125m`
- Offset: O0
- Fit batches: 8
- Validation batches: 8
- Untouched WikiText batches: 32
- Untouched C4 batches: 32
- Max edits: 64
- Random repeats: 3
- QAT checkpoint/logits/latent weights/optimizer steps: forbidden

## Fairness rule

Random controls are allowed to use the same validation split for layer ranking as CE candidates. Therefore this is a strong control:

```text
CE candidate generation + validation top-k
vs
random candidate generation + validation top-k
```

Both use the same layer counts and edit budgets.

## Gate

Primary:

- CE joint top-k should beat the mean of random joint top-k controls on WikiText untouched-32 and C4 untouched-32 for both models.

Secondary:

- If random controls also improve but CE is stronger, the claim becomes “CE gradient improves edit quality over random search,” not merely “edits help.”
- If random matches CE, the mechanism claim weakens and the next step must focus on validation-selection bias and candidate-space priors.
