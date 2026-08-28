# CEGSP-04A：edit-budget / cost sensitivity

## Motivation

CEGSP-01B/02A/03A/03B used `max-edits=64`. The result is now robust across WikiText offsets and transfers to C4, but we still need to know whether `64` is a lucky setting or a broad budget plateau.

## Fixed settings

- Model: `facebook/opt-350m`
- Offset: O0 only for this first budget curve
- Data: WikiText fit/val/untouched plus report-only C4 validation
- Method: CE gradient at deployed ternary weights; support/signflip/joint top-k
- k: `{4, 6}` only
- Gradient batches: `1`
- QAT teacher/checkpoint/logits/latent weights: forbidden

## Sweep

`max-edits ∈ {16, 32, 64, 128}`

## Gate

Primary:

- There exists a non-singleton stable region among `max-edits` where joint top4 or joint top6 improves val, WikiText untouched, and C4 untouched.

Interpretation:

- If 16/32/64/128 all improve with diminishing returns, the method is budget-robust.
- If only 64 improves, the current evidence is more hyperparameter-sensitive than desired; next step must regularize candidate count or define an adaptive edit budget.
- If 128 degrades while 16/32/64 improve, the stop rule is “small bounded edits only,” consistent with all-layer failure.
