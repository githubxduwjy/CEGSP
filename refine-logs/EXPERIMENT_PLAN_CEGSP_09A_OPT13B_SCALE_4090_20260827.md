# CEGSP-09A OPT-1.3B Scale Validation Plan

日期：2026-08-27

## Purpose

Recent CEGSP runs established mechanism evidence on OPT-125M/350M. The user correctly noted that continued small ablations risk becoming too local. CEGSP-09A therefore moves from mechanism dissection to scale validation.

## Claim Tested

> CEGSP is not only a small-model/local-mechanism artifact; the same PTQ-only quantized-point CE editing should improve direct ternary PTQ on a larger 4090-feasible model.

## Setup

- Run id: `CEGSP-09A-OPT13B-O0-U32-SCALE`
- Model: `facebook/opt-1.3b`
- Device target: RTX 4090 24GB
- Quantization: direct ternary PTQ, group size 128, threshold factor 0.7
- Edited weights: Q/K projection matrices only
- Layers: 0-23
- Fit/selection: WikiText-2 O0, 8 fit batches and 8 validation batches
- Untouched evaluation: WikiText-2 32 batches and C4 32 batches
- k sweep: `8,12`
- Edit budget: 64 per layer/family
- Random repeats: 0
- Cloze/downstream: disabled

## Gates

Primary scale gate:

- At least one CE joint top-k improves both W32 and C4 NLL versus direct ternary.
- Runtime and memory remain feasible on RTX 4090, preserving PTQ-level practicality.

Failure interpretation:

- If OOM occurs, retry should reduce holdout batches or evaluate fewer patch sets, not change the method.
- If CEGSP improves W32 but not C4, scale claim remains partial and needs split/model follow-up.
- If CEGSP fails both W32 and C4, this is evidence that small-model mechanism does not directly scale and the next step should examine layer budget or calibration size, not pivot immediately.

## Direction Rule

This run tests scale. It must not start a new method family or add new algorithmic modules.
