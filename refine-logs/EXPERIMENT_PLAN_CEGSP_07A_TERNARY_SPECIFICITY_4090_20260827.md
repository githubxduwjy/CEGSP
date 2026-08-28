# CEGSP-07A Ternary Specificity Plan

日期：2026-08-27

## Claim Tested

CEGSP uses ternary-specific structure, especially zero-support relocation, rather than only a generic nonzero sign correction.

## Setup

- Run id: `CEGSP-07A-OPT350M-O0-U32-TERNARYSPEC`
- Model: `facebook/opt-350m`
- Device target: RTX 4090 24GB
- Quantization: direct ternary PTQ, group size 128, threshold factor 0.7
- Edited weights: Q/K projection matrices only
- Fit/selection: WikiText-2 O0, 8 fit batches and 8 validation batches
- Untouched evaluation: WikiText-2 32 batches and C4 32 batches
- k sweep: `4,6`
- Edit budget: 64 per layer/family
- Random control repeats: 3

## Compared Systems

- CE joint: support relocation or signflip per layer, selected by validation NLL.
- Support-only: CE support relocation with support-selected layers.
- Signflip-only: CE nonzero-only signflip with signflip-selected layers.
- Matched support/signflip on joint-selected layers.
- Cross-layer matched support/signflip on each other's selected layers.
- Random joint control.

## Success Criteria

Main method success:

- CE joint improves both untouched WikiText-2 and C4 versus direct ternary.
- CE joint beats random joint mean.

Ternary specificity:

- Strong: support relocation beats signflip in same-layer comparisons on both W32 and C4.
- Partial: support wins on one holdout or one k; keep support as a useful ternary module but maintain joint method.
- Negative: signflip matches or beats support consistently; do not claim zero-support relocation is dominant.

## Expected Runtime

Approximately 90-120 seconds on RTX 4090, slightly above CEGSP-06A because of added same-layer patch sets.
