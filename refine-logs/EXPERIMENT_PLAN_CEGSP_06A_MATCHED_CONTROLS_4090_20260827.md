# CEGSP-06A Matched Controls Plan

日期：2026-08-27

## Purpose

CEGSP-05B showed that CE-gradient joint edits beat random joint edits on larger untouched WikiText-2 and C4 holdouts. CEGSP-06A asks a narrower mechanism question:

> Is the gain mainly from CE choosing good layers/edit types, from CE choosing better intra-layer ternary candidates, or both?

This remains inside the locked method family: quantized-point CE-gradient guided ternary editing. It does not use QAT teachers, QAT checkpoints, QAT logits, latent full-precision updates, optimizer steps, or TDBT/path-barrier transport.

## Experiment

- Model: `facebook/opt-350m`
- Hardware target: RTX 4090 24GB
- Quantization: direct ternary PTQ, group size 128, threshold factor 0.7
- Edited weights: Q/K projection weights only
- Calibration/selection: WikiText-2 fit/val split, offset O0
- Untouched holdouts: WikiText-2 32 batches and C4 validation 32 batches
- Budget: max edits 64 per layer/family
- k sweep: `4,6`
- Random repeats: 3

## Patch Sets

- `ksweep-joint-top{k}-qk`: CE-selected layer/type and CE-selected intra-layer candidates.
- `random-r*-joint-top{k}-qk`: random candidates with their own validation-selected layer/type.
- `matched-r*-random-candidate-on-ce-joint-top{k}-qk`: CE-selected layer/type, but random intra-layer candidates.
- `matched-r*-ce-candidate-on-random-joint-top{k}-qk`: random-selected layer/type, but CE intra-layer candidates.

## Gates

Primary gate:

- CE joint top-k must improve both WikiText-2 32-batch and C4 32-batch untouched NLL versus direct ternary.
- CE joint top-k must beat random joint mean on both untouched holdouts.

Mechanism gate:

- If CE joint beats `random-candidate-on-ce` by a large margin, CE intra-layer candidate quality is supported.
- If `ce-candidate-on-random` beats random joint but is weaker than CE joint, both candidate quality and layer/type selection matter.
- If matched controls equal CE joint, the claim must be narrowed to the surviving component rather than changing research direction.

## Direction Rule

This experiment may refine the mechanism attribution of CEGSP. It must not trigger a new research direction by itself.
