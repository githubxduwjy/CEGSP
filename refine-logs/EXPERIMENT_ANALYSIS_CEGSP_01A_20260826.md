# Experiment Analysis: CEGSP-01A

日期：2026-08-26

## 1. Run Integrity

- Run ID: `CEGSP-01A`
- Remote: `root@xj-member.bitahub.com:42126`
- GPU: RTX 4090 24GB
- Model: `facebook/opt-350m`
- Layers: all decoder layers `0..23`
- Matrices: Q/K
- Data source: `wikitext-2-raw-v1-arrow-cache-after-ImportError`
- Fit / val / untouched batches: `8 / 8 / 8`
- Elapsed: `19.88 s`
- Result path: `results/remote-runs/CEGSP-01A/result.json`
- Console path: `results/remote-runs/CEGSP-01A/console.log`

Clean-room invariants:

- `uses_qat_checkpoint = false`
- `uses_qat_logits = false`
- `uses_qat_latent_weights = false`
- `uses_qat_state_prior = false`
- `uses_path_barrier_or_tdbt_transport = false`
- `uses_optimizer_steps = false`
- `uses_ce_gradient_at_quantized_weights = true`

This is strict PTQ post-processing: one small CE gradient at deployed ternary weights plus discrete edits.

## 2. Main NLL Results

Baseline:

| Model | Val NLL | Untouched NLL |
|---|---:|---:|
| FP | 3.8039 | 3.9876 |
| direct ternary PTQ | 8.6946 | 8.9900 |

Patch sets:

| Patch set | Layers | Val delta | Untouched delta |
|---|---:|---:|---:|
| cegsp-support-all-qk | 24 | +0.1216 | +0.1332 |
| cegsp-support-selected-qk | 24 | +0.1216 | +0.1332 |
| cegsp-support-topk-qk | 6 | -0.2975 | -0.2720 |
| ce-signflip-all-qk | 24 | +0.0386 | +0.0527 |
| ce-signflip-selected-qk | 23 | +0.0384 | +0.0530 |
| ce-signflip-topk-qk | 6 | -0.3021 | -0.2701 |

## 3. Key Findings

1. Direct CE-gradient editing is much stronger than the operator-proxy version.

   `CEGSP-support-topk-qk` improves untouched NLL by `-0.2720`, compared with the earlier operator-proxy CE-selected improvement of only `-0.0157`.

2. The current best result is top-k CE-gradient editing, not all-layer editing.

   Single-layer deltas are nearly all favorable, but applying all 24 Q/K layer edits simultaneously degrades NLL. This shows strong cross-layer interference.

3. Support projection and nonzero signflip are both strong.

   At top-k 6, support improves untouched NLL by `-0.2720`; signflip improves by `-0.2701`. Support is slightly better on untouched, signflip slightly better on val. The evidence does not yet justify a support-only claim.

4. The next method should become CE-gradient ternary editing with layer-budget selection.

   The contribution should likely be framed as ternary discrete edit selection at quantized weights, with support-swap and signflip as two edit families. The support-only story is too narrow.

## 4. Decision

```text
Primary gate: PASS.
Ternary support-only specificity: NOT YET.
Next experiment: top-k sweep and joint support/signflip selection.
```

## 5. Next Experiment

Run:

```text
CEGSP-01B: k-sweep for support, signflip, and per-layer joint best edits
```

Purpose:

- Determine the stable layer-budget range.
- Check whether top-k 6 is a lucky point.
- Compare support-only, signflip-only, and joint per-layer best.

Gate:

- If a broad k range improves untouched NLL, the CE-gradient edit direction is robust.
- If joint selection beats both support-only and signflip-only, the method should become joint ternary edit selection rather than support projection only.

