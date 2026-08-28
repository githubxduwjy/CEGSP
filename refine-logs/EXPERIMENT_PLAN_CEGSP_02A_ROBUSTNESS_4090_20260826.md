# Experiment Plan: CEGSP-02A Split/Offset Robustness

日期：2026-08-26

## Purpose

`CEGSP-01B` showed a strong single-split result, especially `joint top-6` with untouched NLL delta `-0.2807`. But this is still one calibration split. `CEGSP-02A` tests stability before any further method inflation.

## Frozen Claim

This run tests only:

> CE-gradient ternary editing has stable benefit over direct ternary PTQ across calibration offsets.

It does not claim final SOTA and does not change the research direction.

## Setup

- Model: `facebook/opt-350m`
- Data: Wikitext-2
- Strict PTQ: no QAT checkpoint/logits/teacher, no optimizer steps
- Layers: all decoder layers `0..23`
- Matrices: Q/K
- Fit / val / untouched batches: `8 / 8 / 8`
- Max edits per layer: `64`
- CE gradient batches: `1`
- k sweep: `4,6,8,12`
- Offsets:
  - `CEGSP-02A-O0`: fit offset 0, val offset 0
  - `CEGSP-02A-O1`: fit offset 4096, val offset 4096
  - `CEGSP-02A-O2`: fit offset 8192, val offset 8192

## Compared Families

- support-only top-k
- signflip-only top-k
- joint per-layer best top-k

## Gate

Robustness gate:

- At least one family/k should improve untouched NLL in all 3 offsets.
- Joint top-k should be competitive with the best single-family result.

Failure interpretation:

- If improvements only appear at offset 0, CEGSP is likely calibration-specific.
- If support/signflip alternates heavily, the method should be framed as joint edit selection, not support-only.
- If all offsets fail except one, stop scaling and add regularization/holdout selection.

