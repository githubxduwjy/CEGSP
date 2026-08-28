# Experiment Plan: TQGSP-01B Budget-Matched Validation

日期：2026-08-26

## Why 01B Exists

`TQGSP-01A` completed, but the mechanism control was unfair: `NZ-signflip-G` accepted 64 edits while `TQGSP-support-G` accepted only 11–14 support swaps because the pairwise candidate list reused donors/receivers. Therefore `01A` is treated as a harness diagnostic, not as the final mechanism verdict.

`01B` keeps the same model, data, layers, operator, metrics, and clean-room constraints. The only change is budget matching:

- `TQGSP-support-G` uses unique donor/receiver support exchanges and can consume the full edit budget.
- `support-forward` uses the same unique donor/receiver exchange construction.
- `support-random` and `NZ-signflip-G` keep the same `max_edits=64`.

## Frozen Setup

- Run ID: `TQGSP-01B`
- Model: `facebook/opt-350m`
- Layers: `0,7,15,23`
- Operator: `qk`
- Fit / val / untouched batches: 16 / 8 / 8
- Max edits: 64
- Gradient batches: 1
- No QAT artifacts.
- No TDBT path/barrier.

## Gate

Mechanism gate:

- `TQGSP-support-G` should beat `support-random`, `support-forward`, and `NZ-signflip-G` on `untouched_w` operator NMSE in at least 3/4 layers.

Transfer gate:

- `TQGSP-support-G` patched model should not degrade untouched NLL by more than `+0.02` versus direct PTQ.

Interpretation:

- If `NZ-signflip-G` still wins at matched budget, the ternary-zero support projection claim is weak; next idea should shift from support projection to CE-aware quantized-gradient editing or sign/support joint optimization.
- If `TQGSP-support-G` wins operator but not NLL, change objective before scaling.

