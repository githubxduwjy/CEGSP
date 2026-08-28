# CEGSP Two-Day Synthesis and Next Run

日期：2026-08-27

## 1. What the Last Two Days Established

The reliable evidence chain is now:

1. Direct CE gradients at deployed ternary weights improve NLL after direct ternary PTQ.
2. The effect is not a single-split accident: O0/O1/O2 robustness passed on OPT-350M.
3. The effect transfers from WikiText-based selection to untouched C4.
4. The effect is not only small holdout noise: 32-batch WikiText and C4 holdouts passed.
5. The effect is not arbitrary local editing: CE joint edits beat random joint edits by large margins.
6. CEGSP-06A shows the main signal is intra-layer CE candidate quality, while layer/type selection amplifies it.

Current supported claim:

> At deployed ternary PTQ weights, CE gradients contain useful local information for discrete ternary edits. With small top-k layer budgets, these edits improve validation and untouched NLL without QAT teachers, latent weight updates, or optimizer steps.

## 2. What Is Still Missing

The biggest reviewer risk is ternary specificity:

> Is CEGSP genuinely using the three-state structure, especially the zero-support relocation channel, or is it just a generic low-bit gradient trick that would work similarly as nonzero-only sign correction?

Existing results include support-only and signflip-only comparisons, but they do not fully match layer choices. A fairer test must compare support relocation and signflip on the same selected layers.

## 3. Next Run: CEGSP-07A

Purpose:

> Test ternary specificity by comparing zero-support relocation against nonzero-only signflip under matched layer budgets.

Configuration:

- Model: `facebook/opt-350m`
- Quantization: direct ternary PTQ
- Edited modules: Q/K only
- Calibration: WikiText-2 fit/validation O0
- Holdouts: WikiText-2 32 batches, C4 32 batches
- k: `4,6`
- max edits: 64
- random repeats: 3
- Hardware: RTX 4090 24GB

Patch sets of interest:

- `ksweep-joint-top{k}-qk`: final CE joint method.
- `ksweep-support-top{k}-qk`: support relocation with its own top-k layers.
- `ksweep-signflip-top{k}-qk`: nonzero-only signflip with its own top-k layers.
- `matched-support-on-joint-layers-top{k}-qk`: support relocation on CE joint-selected layers.
- `matched-signflip-on-joint-layers-top{k}-qk`: signflip on the same CE joint-selected layers.
- `matched-signflip-on-support-layers-top{k}-qk`: signflip on support-selected layers.
- `matched-support-on-signflip-layers-top{k}-qk`: support relocation on signflip-selected layers.

Primary gate:

- CE joint top-k improves both WikiText-2 32-batch and C4 32-batch holdouts.
- CE joint remains better than random joint mean.

Ternary-specificity gate:

- Support relocation should outperform nonzero-only signflip on at least one same-layer comparison on untouched metrics, or the claim must be narrowed to "quantized-point CE editing" rather than "zero-support relocation is the dominant mechanism."
- If signflip consistently matches or beats support in same-layer comparisons, the method should remain joint support/sign editing, and the paper should not overclaim zero-state transport.

Direction rule:

- CEGSP-07A may refine the mechanism claim.
- It must not trigger a method-family pivot by itself.
