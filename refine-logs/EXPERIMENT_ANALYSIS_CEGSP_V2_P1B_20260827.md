# CEGSP-V2-P1B Top-k Saturation Analysis

Date: 2026-08-27

## Purpose

P1B continues the move-space investigation after P1A.  It asks whether the gain saturates around top-6 layers or keeps improving as more Q/K layers are edited.

## Configuration

- Remote: `root@xj-member.bitahub.com:42188`
- GPU: RTX 4090 24GB
- Run id: `CEGSP-V2-P1B-TOPK-SATURATION-OPT125M`
- Remote result: `/root/tqgsp-runs/CEGSP-V2-P1B-TOPK-SATURATION-OPT125M/result.json`
- Model: `facebook/opt-125m`
- Dataset: Wikitext-2 raw
- Fit / val / untouched batches: 8 / 8 / 16
- Quantization: direct ternary, group size 128, rho 0.7
- Search space: Q/K only
- Edits: 64 per layer
- k sweep: `{3, 6, 9, 12}`

## Results

Direct ternary:

| Metric | NLL |
|---|---:|
| val | 9.7031 |
| untouched W2 | 9.6952 |

Top-k curve:

| k | support Δval | support Δuntouched | signflip Δval | signflip Δuntouched | mixed Δval | mixed Δuntouched |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | -0.2777 | -0.2626 | -0.2635 | -0.2533 | -0.2857 | -0.2706 |
| 6 | -0.3661 | -0.3597 | -0.3217 | -0.3222 | -0.3662 | -0.3644 |
| 9 | -0.3982 | -0.3939 | -0.3465 | -0.3424 | -0.3962 | -0.3959 |
| 12 | -0.4164 | -0.4151 | -0.3568 | -0.3637 | -0.4199 | -0.4201 |

Best patch by untouched NLL:

- `ksweep-joint-top12-qk`
- untouched W2 NLL: `9.2751`
- delta vs direct: `-0.4201`

## Interpretation

1. The gain does not saturate at top-6.  Editing all Q/K layers continues to improve validation and untouched NLL.
2. Support relocation is consistently stronger than signflip-only.
3. Mixed support/signflip is usually best, but the gain over support-only is small.
4. Therefore the P0 gap to One-Step QAT is not mainly a missing signflip issue.

Relative to P0 One-Step QAT:

- P0 One-Step QAT untouched gain: `9.6952 - 9.0134 = 0.6818`
- P1B mixed top-12 untouched gain: `9.6952 - 9.2751 = 0.4201`
- Approximate closure vs One-Step QAT: `61.6%`

This is now a stronger partial-gap-recovery result than P0/P1A.

## Next Step

The next useful check is scale, not another tiny local ablation:

- Run the same top-k saturation idea on `facebook/opt-350m`.
- Use Q/K layers 0-23 and k `{6, 12, 18, 24}`.
- Keep random controls minimal.
- Use the result to decide whether all-layer Q/K CEGSP is a stable rule or an OPT-125M artifact.

