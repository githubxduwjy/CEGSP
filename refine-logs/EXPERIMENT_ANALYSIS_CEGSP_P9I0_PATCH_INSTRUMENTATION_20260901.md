# P9-I0: Patch-Level Instrumentation Validation

**Run:** `cegsp_p9i0_patch_instrumentation_llama2_7b_a100_20260901_42028_retry2`  
**Device:** NVIDIA A100-SXM4-80GB  
**Status:** `PASS_INSTRUMENTATION_GATE`  
**Scope:** one detached diagnostic run; no PT² rerun and no PPL/NLL evaluation

## Purpose

P9-A0 showed stable top-6 layer selection but could not determine whether the
384 within-layer relocations were identical. P9-I0 validates the instrumentation
needed for that question in future runs.

## Frozen protocol

- Input: existing P9-S2 real detached sidecar and deployed Q/K checkpoint.
- Model: Llama-2-7B, BF16, `use_cache=False`.
- Scope: P9-S2 layers `[4, 10, 11, 9, 14, 5]`, Q/K only.
- Gradient: exactly one Wikitext train calibration batch and one backward pass.
- Candidate table: 128 candidates per Q/K module, 256 combined per layer.
- Patch: existing non-overlapping selection rule, 64 relocations per layer.
- Untouched data, PPL/NLL, parameter search, budget search, and epsilon search:
  not used.

## Gate results

| Check | Result |
|---|---:|
| Candidate records | 1536 / 1536 |
| Selected relocations | 384 / 384 |
| Changed coordinates | 768 / 768 |
| Candidate identity fields | PASS |
| Scores finite | PASS |
| Boundary margins saved | PASS |
| Patch reload hash match | PASS |
| Patch reconstruction hash match | PASS |
| Ternary legality | PASS; 0 violations |
| Cardinality preservation | PASS; 0 violations |
| Sidecar reload | PASS; two loads consistent |
| Q/K modules in checkpoint | 64 |
| BF16 deployment residual | `1.953125e-3 < 5e-3` |
| Exit code | 0 |

The complete candidate table and selected patch are saved in:

`results/remote-runs/cegsp_p9i0_patch_instrumentation_llama2_7b_a100_20260901_42028_retry2/candidate_records.json`

The compact result is:

`results/remote-runs/cegsp_p9i0_patch_instrumentation_llama2_7b_a100_20260901_42028_retry2/p9i0_result.json`

## Fingerprints

| Object | SHA-256 |
|---|---|
| Calibration token stream | `877e89228bb6561ae6be3a61aee27bb02e8d12076416d6082a5a77db1a73b91d` |
| Quantized-point gradient | `e81d959220c0caad546a21d405b4a4bafc6ec78e770eb01486d59b8eed0edcad` |
| Baseline Q/K snapshot | `540ee4561772f608bf70503b66a2e6dda172257bcfada0ff184c584337bf2085` |
| State before patch | `722713751b775e4f844d6ae674d3cc5f1eefd4126f1c6716ce88dac2d868b6d0` |
| State after patch | `2c391f200f828b345df6fd44844cf58f348d83280679da9d65eb67052f8333ec` |
| Canonical selected patch | `381ac014681f68dfda710460d80827c77002c3605eef74329a4c1bddeebefc9f` |

## Boundary margins

Raw candidate-score margins were recorded for each selected layer:

| Layer | `S60-S61` | `S64-S65` | `S70-S71` |
|---:|---:|---:|---:|
| 4 | 8.121e-6 | 4.117e-6 | 1.166e-5 |
| 10 | 3.297e-6 | 9.037e-7 | 6.085e-7 |
| 11 | 5.376e-6 | 2.613e-6 | 1.597e-7 |
| 9 | 4.374e-6 | 1.346e-6 | 8.721e-7 |
| 14 | 3.077e-6 | 1.581e-6 | 2.277e-6 |
| 5 | 5.362e-6 | 2.725e-6 | 4.520e-7 |

Several boundaries are in the `1e-7`–`1e-6` range. This supports the
plausibility of within-layer top-K sensitivity, but P9-I0 itself does not
compare two independent gradients and therefore does not establish a
reproducibility rate.

## Interpretation

P9-I0 passes its intended infrastructure claim:

> A detached CEGSP run can now save a complete, auditable candidate table and
> selected patch, together with the hashes and score margins needed for a
> future P9-D1 comparison.

This is not a performance result and does not alter the P9-S2 conclusion:
strong-PTQ robust gain remains unestablished. The next diagnostic, if approved,
can compare candidate-level validity and patch overlap without guessing from
aggregate JSON summaries.

