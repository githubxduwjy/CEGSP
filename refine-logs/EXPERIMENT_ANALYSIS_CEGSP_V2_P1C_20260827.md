# CEGSP-V2-P1C OPT-350M Top-k Saturation Analysis

Date: 2026-08-27

## Purpose

P1C scales the P1B top-k saturation test from OPT-125M to OPT-350M.  The goal is to check whether the "more Q/K layers is better" observation from OPT-125M generalizes.

## Configuration

- Remote: `root@xj-member.bitahub.com:42188`
- GPU: RTX 4090 24GB
- Run id: `CEGSP-V2-P1C-TOPK-SATURATION-OPT350M`
- Remote result: `/root/tqgsp-runs/CEGSP-V2-P1C-TOPK-SATURATION-OPT350M/result.json`
- Model: `facebook/opt-350m`
- Dataset: Wikitext-2 raw
- Fit / val / untouched batches: 8 / 8 / 16
- Quantization: direct ternary, group size 128, rho 0.7
- Search space: Q/K only
- Edits: 64 per layer
- k sweep: `{6, 12, 18, 24}`

## Results

Direct ternary:

| Metric | NLL |
|---|---:|
| val | 8.6946 |
| untouched W2 | 8.8477 |

Top-k curve:

| k | support Δval | support Δuntouched | signflip Δval | signflip Δuntouched | mixed Δval | mixed Δuntouched |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | -0.2975 | -0.2392 | -0.3021 | -0.2437 | -0.3226 | -0.2612 |
| 12 | -0.2970 | -0.2210 | -0.1476 | -0.0673 | -0.1426 | -0.0515 |
| 18 | -0.1855 | -0.1262 | -0.0419 | +0.0168 | -0.1002 | -0.0247 |
| 24 | +0.1216 | +0.1976 | +0.0386 | +0.1043 | +0.0755 | +0.1561 |

Best patch:

- `ksweep-joint-top6-qk`
- untouched W2 NLL: `8.5865`
- delta vs direct: `-0.2612`

## Interpretation

This is the most important correction from today's continuation:

1. OPT-350M does not follow OPT-125M's "edit all Q/K layers" behavior.
2. Top-6 is clearly best; top-12 already loses much of the gain, and top-24 is harmful.
3. Mixed support/signflip is best at top-6, but the advantage over support/signflip alone is moderate.
4. CEGSP needs a layer-budget/trust-region rule.  "Apply to all Q/K layers" is not a safe default.

## Research Implication

The idea is not in a dead end, but the method should not be framed as unrestricted all-layer editing.  The stronger and safer claim is:

> Quantized-point CE gradients identify a small subset of high-leverage ternary Q/K layers whose discrete support/sign edits reduce task loss; over-extending edits can leave the local basin and degrade performance.

This aligns well with the QAT-basin interpretation: CEGSP is a local discrete correction, so the edit budget and layer budget are part of the method, not incidental hyperparameters.

## Next Step

Before expanding model size or writing claims, the next necessary experiment is validation robustness for the selected top-6 rule on OPT-350M:

- Same OPT-350M setup.
- Freeze rule: mixed top-6 selected on Wikitext validation.
- Evaluate larger untouched Wikitext and C4 if available.
- Do not retune k on the larger holdout.

If top-6 transfers, the next paper-level rule becomes:

> Use validation-selected small top-k layer budgets, not all-layer editing.

