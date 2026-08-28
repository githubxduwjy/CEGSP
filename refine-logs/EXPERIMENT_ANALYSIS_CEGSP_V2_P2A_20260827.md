# CEGSP-V2-P2A OPT-350M Top-6 Holdout Analysis

Date: 2026-08-27

## Purpose

P2A freezes the rule suggested by P1C:

> Use a small validation-selected top-6 Q/K layer budget on OPT-350M; do not expand to all layers.

It tests whether this rule survives larger untouched Wikitext and C4 transfer.

## Configuration

- Remote: `root@xj-member.bitahub.com:42188`
- GPU: RTX 4090 24GB
- Run id: `CEGSP-V2-P2A-OPT350M-TOP6-HOLDOUT`
- Remote result: `/root/tqgsp-runs/CEGSP-V2-P2A-OPT350M-TOP6-HOLDOUT/result.json`
- Model: `facebook/opt-350m`
- Dataset: Wikitext-2 raw + C4 validation transfer
- Fit / val / untouched W2 / untouched C4 batches: 8 / 8 / 64 / 32
- Quantization: direct ternary, group size 128, rho 0.7
- Search space: Q/K only
- Edits: 64 per layer
- Fixed rule: validation-selected top-6 layer budget

## Results

Direct ternary:

| Metric | NLL |
|---|---:|
| val | 8.6946 |
| untouched W2 | 8.6514 |
| untouched C4 | 8.1248 |

Patch results:

| Patch set | Δval | Δuntouched W2 | Δuntouched C4 |
|---|---:|---:|---:|
| support top-6 | -0.2975 | -0.2013 | -0.2276 |
| signflip top-6 | -0.3021 | -0.2191 | -0.2120 |
| mixed top-6 | -0.3226 | -0.2362 | -0.2103 |
| matched support on mixed layers | -0.3119 | -0.2378 | -0.2127 |
| matched signflip on mixed layers | -0.2971 | -0.2308 | -0.1808 |
| random joint top-6 | -0.0016 | -0.0002 | +0.0002 |
| random candidate on CE joint layers | +0.0014 | +0.0006 | +0.0007 |

## Interpretation

P2A is a strong positive robustness result:

1. The top-6 rule transfers from the small P1C holdout to a larger 64-batch Wikitext holdout.
2. It also transfers to C4 without using C4 for selection.
3. Random controls remain near zero, so the improvement is not caused by arbitrary perturbations on lucky layers.
4. Mixed top-6 is best on Wikitext, while support-only is slightly best on C4.  This suggests support relocation is the more stable core, and signflip is a useful but not dominant companion.

## Claim Update

Supported:

> On OPT-350M, a validation-selected small top-k CEGSP rule improves both Wikitext and C4 holdouts after direct ternary PTQ, while random matched edits do not.

Still open:

> Whether the same top-6 rule is robust to different calibration/validation offsets.

## Next Step

Run P2B with shifted fit/validation/C4 token offsets while keeping the same method family and top-6 rule.  Do not change k based on P2A.

