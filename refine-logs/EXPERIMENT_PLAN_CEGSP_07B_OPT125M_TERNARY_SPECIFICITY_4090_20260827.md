# CEGSP-07B OPT-125M Ternary Specificity Plan

日期：2026-08-27

## Purpose

CEGSP-07A showed on OPT-350M that zero-support relocation is a real ternary-specific module beyond nonzero-only signflip. CEGSP-07B repeats the same matched-control logic on OPT-125M to check whether this mechanism transfers to a second model size.

## Setup

- Run id: `CEGSP-07B-OPT125M-O0-U32-TERNARYSPEC`
- Model: `facebook/opt-125m`
- Device target: RTX 4090 24GB
- Quantization: direct ternary PTQ, group size 128, threshold factor 0.7
- Edited weights: Q/K projection matrices only
- Layers: 0-11
- Fit/selection: WikiText-2 O0, 8 fit batches and 8 validation batches
- Untouched evaluation: WikiText-2 32 batches and C4 32 batches
- k sweep: `2,3`
- Edit budget: 64 per layer/family
- Random control repeats: 3

## Gates

Main method gate:

- CE joint improves both untouched WikiText-2 and C4 versus direct ternary.
- CE joint beats random joint mean.

Ternary-specificity gate:

- Same-layer support relocation should beat or remain competitive with nonzero-only signflip on W32/C4.
- If support is weaker on 125M, keep the final method as joint support/sign editing and state that the support/sign balance is model-size dependent.

## Direction Rule

This run may refine the support-vs-sign mechanism claim. It must not change the locked CEGSP method family.
