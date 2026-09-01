# P9-I0: Patch-Level Instrumentation Validation

**Status:** pre-registered and approved for one A100 instrumentation run  
**Date:** 2026-09-01  
**Purpose:** make the next PT²/CEGSP diagnostic answer patch-level identity
questions without launching another performance replication

## Question

Can a detached CEGSP run save and reload the complete candidate table and
selected patch needed to distinguish layer-level stability from within-layer
patch stability?

## Frozen protocol

- Input: the existing P9-S2 Llama-2-7B detached sidecar; do not rerun PT².
- Device: one A100; BF16 model; `use_cache=False`.
- State: reload real `T`, `mu`, `alpha`, validity masks, SSR permutations, and
  the deployed PT² Q/K checkpoint from disk.
- Scope: the six P9-S2 selected layers `[4, 10, 11, 9, 14, 5]`, Q/K only.
- Gradient: exactly one calibration batch and one backward pass; all model
  parameters frozen except the scoped Q/K projections.
- Candidate generation: the existing frozen affine CEGSP rule, 128 ranked
  candidates per Q/K module (256 combined candidates per layer), with the
  existing non-overlapping selection rule and 64 selected relocations per
  selected layer.
- Deployment comparison: FP32 sidecar reconstruction against the BF16 model
  uses the fixed `5e-3` maximum-absolute residual tolerance; this is a dtype
  representation tolerance, not a searched experiment parameter.
- No PPL/NLL evaluation, no untouched data, no threshold/budget/epsilon sweep,
  and no post-hoc candidate selection.

## Required artifacts and gates

The run must save:

1. every candidate identity `(layer, projection, row, group, donor, receiver,
   donor_sign, receiver_sign)`;
2. every candidate score and selected flag;
3. top-60/61, top-64/65, and top-70/71 score margins where available;
4. canonical patch SHA-256;
5. calibration-token, gradient, state, and baseline-Q fingerprints;
6. a reload/reconstruction check and a legality/cardinality check.

The instrumentation gate passes only if all candidate identities are present,
the selected patch has 384 moves / 768 changed coordinates, all fingerprints
are finite and deterministic, reloaded patch hash equals the original hash,
SSR/state reconstruction succeeds, and no ternary legality or cardinality
violation is observed.

This run cannot establish P9-S1/P9-S2 patch overlap because P9-S1 did not save
its per-move records. It establishes that all future runs will save the records
needed for that comparison.
