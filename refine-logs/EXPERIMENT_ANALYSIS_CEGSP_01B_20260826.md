# Experiment Analysis: CEGSP-01B k-Sweep

日期：2026-08-26

## 1. Run Integrity

- Run ID: `CEGSP-01B`
- Remote: `root@xj-member.bitahub.com:42126`
- GPU: RTX 4090 24GB
- Model: `facebook/opt-350m`
- Layers: all decoder layers `0..23`
- Matrices: Q/K
- Fit / val / untouched batches: `8 / 8 / 8`
- k sweep: `1,2,4,6,8,12,16,24`
- Elapsed: `22.55 s`
- Result path: `results/remote-runs/CEGSP-01B/result.json`
- Console path: `results/remote-runs/CEGSP-01B/console.log`

Clean-room invariants:

- `uses_qat_checkpoint = false`
- `uses_qat_logits = false`
- `uses_qat_latent_weights = false`
- `uses_qat_state_prior = false`
- `uses_path_barrier_or_tdbt_transport = false`
- `uses_optimizer_steps = false`
- `uses_ce_gradient_at_quantized_weights = true`

## 2. k-Sweep Results

Direct ternary baseline:

- Val NLL: `8.6946`
- Untouched NLL: `8.9900`

| k | Support val Δ | Support untouched Δ | Signflip val Δ | Signflip untouched Δ | Joint val Δ | Joint untouched Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -0.0816 | -0.0650 | -0.0881 | -0.0660 | -0.0881 | -0.0660 |
| 2 | -0.1523 | -0.1203 | -0.1552 | -0.1292 | -0.1582 | -0.1188 |
| 4 | -0.2651 | -0.2263 | -0.2552 | -0.2184 | -0.2718 | -0.2250 |
| 6 | -0.2975 | -0.2720 | -0.3021 | -0.2701 | -0.3226 | -0.2807 |
| 8 | -0.2461 | -0.2134 | -0.2536 | -0.2447 | -0.3106 | -0.2621 |
| 12 | -0.2970 | -0.2596 | -0.1476 | -0.1261 | -0.1426 | -0.1130 |
| 16 | -0.2911 | -0.2415 | -0.1033 | -0.1068 | -0.1264 | -0.1115 |
| 24 | +0.1216 | +0.1332 | +0.0386 | +0.0527 | +0.0755 | +0.0940 |

Best untouched result:

```text
joint top-6: val Δ = -0.3226, untouched Δ = -0.2807
layers = [22, 16, 13, 19, 12, 17]
edits = {22: signflip, 16: support, 13: support, 19: support, 12: support, 17: signflip}
```

## 3. Findings

1. CE-gradient ternary editing is robust over a useful k range.

   k values from `1` through `16` improve untouched NLL for support-only, signflip-only, and joint variants. The improvement peaks around k = 6.

2. All-layer editing is consistently harmful.

   At k = 24, all three families degrade NLL. This confirms the CEGSP-01A observation: single-layer positive deltas do not add linearly; layer-budget control is essential.

3. Joint support/signflip is the best current method.

   The best untouched result is `joint top-6` with `-0.2807` NLL. This slightly beats support-only top-6 (`-0.2720`) and signflip-only top-6 (`-0.2701`).

4. The innovation should not be framed as support-only.

   Since signflip is competitive and joint is best, the clean claim is:

   > CE gradients at deployed ternary weights reveal a small set of beneficial discrete ternary edits; both zero-support relocation and nonzero polarity correction are needed.

5. The method remains PTQ-like.

   CE gradient collection took `0.18 s`; no optimizer step or QAT artifact was used. Most time is model/data loading and patch-set evaluation.

## 4. Decision

```text
CEGSP direction: STRONG PASS.
Support-only claim: WEAK.
Joint ternary edit selection: CURRENT BEST.
Recommended next run: reproduce joint top-k on another calibration split / seed, then test C4 transfer.
```

## 5. Next Experiment

Recommended:

```text
CEGSP-02A: split/seed robustness
```

Minimal design:

- Same OPT-350M setup.
- Run 3 seeds or 3 calibration offsets.
- Evaluate k values around the stable region: `4,6,8,12`.
- Compare support-only, signflip-only, joint-best.
- Primary metric: untouched Wikitext NLL.

Gate:

- joint top-k should improve untouched NLL in all or most seeds.
- k=6 or nearby should remain near-optimal.

Only after this passes should we scale to larger models or add C4.

