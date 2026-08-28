# CEGSP-V2-P1A Move-Space Analysis

Date: 2026-08-27

## Purpose

This run continues from `CEGSP-V2-P0-OPT125M`.  Because direct file sync to the remote machine was blocked by transient DNS failures, this P1A run reused the already-deployed stable `cegsp_ce_gradient_4090.py` harness.

It tests the core move-space question:

> Is CEGSP's gap to One-Step QAT mainly caused by using support relocation alone, or do mixed support/signflip edits recover substantially more?

This run does not include same-run One-Step QAT or alpha-refit.  The One-Step QAT reference remains the immediately preceding same-model/same-split P0 result.

## Configuration

- Remote: `root@xj-member.bitahub.com:42188`
- GPU: RTX 4090 24GB
- Run id: `CEGSP-V2-P1A-MOVE-SPACE-EXISTING-OPT125M`
- Remote result: `/root/tqgsp-runs/CEGSP-V2-P1A-MOVE-SPACE-EXISTING-OPT125M/result.json`
- Model: `facebook/opt-125m`
- Dataset: Wikitext-2 raw
- Fit / val / untouched batches: 8 / 8 / 16
- Quantization: direct ternary, group size 128, rho 0.7
- Search space: Q/K only
- Edits: 64 per layer
- k sweep: top-3 and top-6 layers
- Controls: support-only, signflip-only, mixed joint, matched-layer controls, random support/signflip/joint controls

## Main Results

Direct ternary baseline:

| Metric | NLL |
|---|---:|
| val | 9.7031 |
| untouched W2 | 9.6952 |

Top-3:

| Patch set | Val NLL | Untouched NLL | Delta Val | Delta Untouched |
|---|---:|---:|---:|---:|
| support top-3 | 9.4254 | 9.4326 | -0.2777 | -0.2626 |
| signflip top-3 | 9.4396 | 9.4420 | -0.2635 | -0.2533 |
| mixed joint top-3 | 9.4174 | 9.4246 | -0.2857 | -0.2706 |
| random joint r0 top-3 | 9.7019 | 9.6956 | -0.0012 | +0.0003 |
| random joint r1 top-3 | 9.7015 | 9.6942 | -0.0016 | -0.0011 |

Top-6:

| Patch set | Val NLL | Untouched NLL | Delta Val | Delta Untouched |
|---|---:|---:|---:|---:|
| support top-6 | 9.3370 | 9.3356 | -0.3661 | -0.3597 |
| signflip top-6 | 9.3814 | 9.3730 | -0.3217 | -0.3222 |
| mixed joint top-6 | 9.3368 | 9.3309 | -0.3662 | -0.3644 |
| random joint r0 top-6 | 9.7022 | 9.6955 | -0.0009 | +0.0003 |
| random joint r1 top-6 | 9.6998 | 9.6934 | -0.0033 | -0.0018 |

## Interpretation

Positive evidence:

1. Increasing the layer budget from top-3 to top-6 gives a real gain without over-edit collapse.
2. Mixed support/signflip is the best patch set on both top-3 and top-6.
3. CE-guided edits dominate random edits by a very large margin, so the gain is not arbitrary perturbation.
4. Support relocation remains at least as important as signflip; signflip helps in some layers but does not replace support movement.

Limiting evidence:

1. Mixed joint only slightly improves over support-only:
   - top-3 untouched: `-0.2706` vs `-0.2626`
   - top-6 untouched: `-0.3644` vs `-0.3597`
2. Therefore the P0 gap to One-Step QAT is not solved by simply adding signflip.
3. The next missing piece is likely either alpha/scale adaptation, broader module scope, or stronger layer/module selection, not a wholesale direction change.

## Relation to P0 One-Step QAT

From `CEGSP-V2-P0-OPT125M`:

- One-Step QAT untouched improvement: `9.6952 -> 9.0134`, gain `0.6818`
- P1A best mixed top-6 untouched improvement: `9.6952 -> 9.3309`, gain `0.3644`

So P1A mixed top-6 recovers about:

```text
0.3644 / 0.6818 = 53.4%
```

of the One-Step QAT improvement on untouched Wikitext-2, using one CE gradient and discrete post-training edits.

This is a stronger result than P0 support-only top-3, but still not enough to claim CEGSP matches QAT.

## Claim Update

Supported more strongly after P1A:

> Under a fixed ternary PTQ initialization, quantized-point CE gradients can select useful support and sign discrete edits. A small mixed move space recovers a meaningful fraction of One-Step QAT's gain while remaining optimizer-free for the CEGSP path.

Still not supported:

> Mixed CEGSP fully explains or matches QAT recovery.

## Next Step

Run a fuller P1B once file sync is available, or via a compact remote script:

- Add alpha/scale refit control explicitly.
- Add top-k `{3, 6, 9}` to test whether gains saturate before all layers.
- Keep same One-Step QAT in the same result JSON.
- Optionally add V/O as a second module scope only after Q/K top-k saturation is measured.

