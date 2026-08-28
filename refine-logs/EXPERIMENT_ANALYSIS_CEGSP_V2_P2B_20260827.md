# CEGSP-V2-P2B OPT-350M Offset Robustness Analysis

Date: 2026-08-27

## Purpose

P2B repeats P2A with shifted token offsets to test whether the OPT-350M top-6 rule is split-sensitive.

## Configuration

- Remote: `root@xj-member.bitahub.com:42188`
- GPU: RTX 4090 24GB
- Run id: `CEGSP-V2-P2B-OPT350M-TOP6-OFFSET`
- Remote result: `/root/tqgsp-runs/CEGSP-V2-P2B-OPT350M-TOP6-OFFSET/result.json`
- Model: `facebook/opt-350m`
- Dataset: Wikitext-2 raw + C4 validation transfer
- Fit / val / untouched W2 / untouched C4 batches: 8 / 8 / 64 / 32
- Offsets: fit `4096`, val `4096`, C4 `8192`
- Quantization: direct ternary, group size 128, rho 0.7
- Search space: Q/K only
- Edits: 64 per layer
- Fixed rule: validation-selected top-6 layer budget

## Results

Direct ternary:

| Metric | NLL |
|---|---:|
| val | 8.7235 |
| untouched W2 | 8.5921 |
| untouched C4 | 8.4237 |

Patch results:

| Patch set | Δval | Δuntouched W2 | Δuntouched C4 |
|---|---:|---:|---:|
| support top-6 | -0.2528 | -0.2026 | -0.2037 |
| signflip top-6 | -0.2757 | -0.2331 | -0.1599 |
| mixed top-6 | -0.2679 | -0.2112 | -0.2342 |
| matched support on mixed layers | -0.2700 | -0.2121 | -0.2276 |
| matched signflip on mixed layers | -0.2452 | -0.2020 | -0.2027 |
| random joint top-6 | -0.0007 | -0.0007 | -0.0010 |
| random candidate on CE joint layers | +0.0011 | +0.0004 | -0.0012 |

## Interpretation

P2B confirms the P2A result under shifted token offsets:

1. Top-6 CE-guided edits improve Wikitext and C4 again.
2. Random and matched-random controls remain essentially zero.
3. The relative ranking varies by split:
   - signflip is strongest on untouched W2,
   - mixed is strongest on C4,
   - support remains competitive and stable.

The core claim is now stronger:

> On OPT-350M, validation-selected small-budget CEGSP edits improve held-out Wikitext and C4 across at least two token-offset settings.

## Method Implication

The paper should avoid saying support-only is always dominant.  A more accurate method is a mixed move space with support relocation as the ternary-native backbone and signflip as a same-gradient companion action.

## Next Step

Move to cross-architecture validation.  Run the same family of experiments on a Pythia/GPT-NeoX model to ensure the result is not an OPT projection-layout artifact.

