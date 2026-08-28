# CEGSP-05A：larger untouched holdout validation

## Motivation

CEGSP-03B and CEGSP-04B used 8 untouched batches for WikiText and C4. The next risk is small-sample variance. CEGSP-05A keeps the method fixed and increases untouched evaluation to 32 batches on one representative offset for both validated models.

## Fixed settings

- Method: CE gradient at deployed ternary weights
- QAT checkpoint/logits/latent weights/optimizer steps: forbidden
- Fit batches: 8
- Validation batches for layer ranking: 8
- Untouched WikiText batches: 32
- Untouched C4 batches: 32
- Offset: O0
- C4: report-only transfer, not used for selection
- Max edits: 64

## Runs

| run | model | layers | k sweep |
|---|---|---:|---|
| `CEGSP-05A-OPT350M-O0-U32` | `facebook/opt-350m` | 24 | `{4,6}` |
| `CEGSP-05A-OPT125M-O0-U32` | `facebook/opt-125m` | 12 | `{2,3}` |

## Gate

Primary:

- For each model, at least one pre-registered top-k family improves val, WikiText untouched-32, and C4 untouched-32.

Stronger:

- The previously favored joint top-k improves all three splits on both models.

Interpretation:

- If gains persist, current CEGSP evidence graduates from tiny-window diagnostic to larger-heldout diagnostic.
- If only 8-batch runs passed, but 32-batch runs fail, the method is not invalidated but claims must be narrowed and selection must use larger validation/holdout.
