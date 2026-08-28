# CEGSP-07B OPT-125M Ternary Specificity Analysis

日期：2026-08-27

## 1. Question

CEGSP-07B repeats the CEGSP-07A ternary-specificity matched controls on OPT-125M. The goal is to check whether the zero-support relocation signal transfers to a second model size.

This run is strict PTQ-only: no QAT teacher, QAT logits, QAT checkpoint, latent full-precision update, optimizer step, or TDBT/path-barrier transport.

## 2. Integrity

| Item | Value |
|---|---:|
| Run id | `CEGSP-07B-OPT125M-O0-U32-TERNARYSPEC` |
| Remote result path | `/root/tqgsp-runs/CEGSP-07B-OPT125M-O0-U32-TERNARYSPEC/result.json` |
| Status | complete |
| Elapsed | 64.02 s |
| Model | OPT-125M |
| Layers | 12 |
| Patch sets | 50 |
| Untouched WikiText batches | 32 |
| Untouched C4 batches | 32 |

This report uses a direct remote JSON summary. Raw result remains on the remote path above.

## 3. Main Results

Direct ternary baseline:

| Metric | NLL |
|---|---:|
| validation | 9.703077 |
| WikiText-2 untouched 32 | 9.698358 |
| C4 untouched 32 | 9.190435 |

Deltas are versus direct ternary; lower is better.

| Patch set | k | val delta | W32 delta | C4-32 delta |
|---|---:|---:|---:|---:|
| CE joint | 2 | -0.231834 | -0.256313 | -0.372047 |
| support top-k | 2 | -0.214485 | -0.241514 | -0.356439 |
| signflip top-k | 2 | -0.207950 | -0.228517 | -0.296202 |
| support on joint layers | 2 | -0.214485 | -0.241514 | -0.356439 |
| signflip on joint layers | 2 | -0.207950 | -0.228517 | -0.296202 |
| random joint mean | 2 | -- | -0.000081 | +0.000204 |
| CE joint | 3 | -0.285708 | -0.303213 | -0.410006 |
| support top-k | 3 | -0.277671 | -0.297697 | -0.405734 |
| signflip top-k | 3 | -0.263477 | -0.277491 | -0.341346 |
| support on joint layers | 3 | -0.277671 | -0.297697 | -0.405734 |
| signflip on joint layers | 3 | -0.263477 | -0.277491 | -0.341346 |
| random joint mean | 3 | -- | -0.000318 | -0.000010 |

## 4. Interpretation

Main method gate passes:

- CE joint top-2/top-3 improves both untouched WikiText-2 and C4.
- CE joint is far stronger than random joint, whose average deltas remain near zero.

Ternary-specificity gate also passes on OPT-125M:

- On the same joint-selected layers, support relocation beats nonzero-only signflip for both k=2 and k=3 on W32 and C4.
- The C4 gap is especially clear: at k=3, support on joint layers gives -0.405734 while signflip on the same layers gives -0.341346.
- CE joint remains best overall, so the preferred method is still joint support/sign editing, not support-only.

## 5. Claim Update

Supported after CEGSP-07A/07B:

> Zero-support relocation is a real ternary-specific component of CEGSP. It is not fully replaceable by nonzero-only signflip under matched layer budgets, and this holds on both OPT-350M and OPT-125M.

The paper claim should remain calibrated:

> The final method is not support-only. It is a small-budget joint ternary editing method where support relocation provides a ternary-specific channel and signflip supplies complementary polarity correction.

## 6. Next Step

The next paper-critical gap is no longer basic ternary specificity. The next smallest useful experiment should be one of:

1. Compare against a stronger published PTQ baseline or a locally implemented GPTQ/AdaRound-like reconstruction baseline under the same OPT-125M/350M setting.
2. Add a small downstream task evaluation for direct ternary versus CEGSP top-k to check whether NLL gains translate.
3. Run a larger OPT model only if the implementation/cost remains 4090-safe.

Recommended next run: a small downstream sanity evaluation, because it directly tests whether the current NLL mechanism matters beyond perplexity tables.
