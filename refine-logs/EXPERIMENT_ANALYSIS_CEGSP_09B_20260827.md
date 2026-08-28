# CEGSP-09B analysis: OPT-2.7B larger scale validation on RTX 4090

日期：2026-08-27

## 1. Run identity

- Run ID: `CEGSP-09B-OPT27B-O0-U32-SCALE`
- Remote raw result: `/root/tqgsp-runs/CEGSP-09B-OPT27B-O0-U32-SCALE/result.json`
- Remote log: `/root/tqgsp-runs/CEGSP-09B-OPT27B-O0-U32-SCALE.log`
- Model: `facebook/opt-2.7b`
- Layers: 0--31, Q/K only
- Quantization: direct ternary PTQ, group size 128, threshold factor 0.7
- Selection: CE gradient at deployed ternary weights
- Clean-room invariants:
  - QAT checkpoint: false
  - QAT logits: false
  - QAT latent weights: false
  - QAT state prior: false
  - optimizer steps: false
  - CE gradient at quantized weights: true
- Runtime: 199.14 s
- Peak CUDA memory allocated: 7.96 GB on RTX 4090

## 2. Pre-registered gate

Gate: at least one CE joint top-k patch set improves both untouched Wikitext and untouched C4 NLL over direct ternary PTQ, with acceptable RTX 4090 cost.

Result: **PASS_LARGER_SCALE**.

Both joint top-k patch sets pass by a large margin:

| Patch set | Val NLL | Δ val | Wikitext-24 NLL | Δ Wikitext-24 | C4-24 NLL | Δ C4-24 |
|---|---:|---:|---:|---:|---:|---:|
| direct ternary | 9.854505 | — | 9.967919 | — | 9.516477 | — |
| `ksweep-joint-top12-qk` | 9.186573 | -0.667932 | 9.293325 | -0.674594 | 8.891018 | -0.625459 |
| `ksweep-joint-top16-qk` | 9.145458 | -0.709047 | 9.236267 | -0.731652 | 8.867140 | -0.649337 |

The best joint variant is `ksweep-joint-top16-qk`.

## 3. Support/sign comparison

At OPT-2.7B scale, all three CE-guided edit families improve strongly:

| Patch set | Δ val | Δ Wikitext-24 | Δ C4-24 |
|---|---:|---:|---:|
| `ksweep-support-top12-qk` | -0.651334 | -0.666567 | -0.612084 |
| `ksweep-signflip-top12-qk` | -0.626442 | -0.657883 | -0.596303 |
| `ksweep-joint-top12-qk` | -0.667932 | -0.674594 | -0.625459 |
| `ksweep-support-top16-qk` | -0.694597 | -0.724081 | -0.637067 |
| `ksweep-signflip-top16-qk` | -0.665407 | -0.702378 | -0.636116 |
| `ksweep-joint-top16-qk` | -0.709047 | -0.731652 | -0.649337 |

Interpretation:

- Joint support/sign selection is strongest on all three measured metrics.
- Support relocation remains competitive and is usually stronger than signflip at matched k.
- The result supports the current method shape: keep joint editing, but present the zero-support relocation path as the ternary-specific primitive that gives the method its 1.58-bit identity.

## 4. Cost

The larger run remains practical on a single RTX 4090:

- Total runtime: 199.14 s
- Model loading: 87.12 s
- Tokenizer/data loading: 44.08 s
- CE gradient collection: 1.31 s
- Direct PTQ apply: 1.43 s
- Edit generation and single-layer eval: 41.39 s
- Patch-set NLL eval: 20.30 s
- Peak memory: 7.96 GB

This is not QAT-like cost. The actual CE-gradient collection is around one second; the expensive parts are model/data loading and patch evaluation.

## 5. Claim update

The earlier concern that experiments were too small is now materially addressed:

- OPT-350M: passed multiple offsets, C4 transfer, budget, random controls, matched controls, ternary specificity.
- OPT-125M: passed second-model sanity and ternary specificity.
- OPT-1.3B: passed scale validation.
- OPT-2.7B: passed larger scale validation with stronger deltas than OPT-1.3B.

Current robust claim:

> In OPT-family decoder LMs up to 2.7B on RTX 4090, a strict-PTQ CE-gradient edit pass at deployed ternary weights can substantially improve direct ternary PTQ NLL on held-out Wikitext and transfer C4, without QAT teachers, QAT logits, latent weights, or optimizer steps.

Still missing before a paper-level claim:

1. One non-OPT architecture or at least a different model family.
2. A stronger downstream evaluation than the current LAMBADA floor-effect sanity check.
3. A larger holdout or repeated offset on 1.3B/2.7B to rule out a single split artifact.
4. Comparison against a real ternary/PTQ baseline implementation, if available in the repo.

## 6. Decision

Do not pivot. The scale result strengthens the fixed CEGSP direction.

Recommended next stage:

1. Run one cross-family validation if a 4090-feasible model is locally/cache-feasible.
2. Otherwise run a repeated-offset scale validation on OPT-1.3B or OPT-2.7B.
3. Avoid returning to tiny 350M ablations unless they test a specific paper-critical claim.
