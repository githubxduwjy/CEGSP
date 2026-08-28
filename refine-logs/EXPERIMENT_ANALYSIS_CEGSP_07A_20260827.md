# CEGSP-07A Ternary Specificity Analysis

日期：2026-08-27

## 1. Question

CEGSP-07A tests whether the current CEGSP gain is tied to ternary-specific zero-support relocation, or whether it can be explained by a generic nonzero-only signflip control.

The key matched comparison is:

- support relocation on a fixed set of layers;
- nonzero-only signflip on the same set of layers.

This is still strict PTQ-only: no QAT teacher, QAT logits, latent weights, optimizer steps, or path-barrier/TDBT transport.

## 2. Integrity

| Item | Value |
|---|---:|
| Run id | `CEGSP-07A-OPT350M-O0-U32-TERNARYSPEC` |
| Remote result path | `/root/tqgsp-runs/CEGSP-07A-OPT350M-O0-U32-TERNARYSPEC/result.json` |
| Status | complete |
| Elapsed | 92.05 s |
| Model | OPT-350M |
| Layers | 24 |
| Patch sets | 50 |
| Untouched WikiText batches | 32 |
| Untouched C4 batches | 32 |

Local `scp/rsync` artifact pull was blocked by temporary DNS resolution failure for `xj-member.bitahub.com`. The complete raw result remains on the remote path above; this report uses a direct remote JSON summary.

## 3. Main Results

Direct ternary baseline:

| Metric | NLL |
|---|---:|
| validation | 8.694630 |
| WikiText-2 untouched 32 | 8.790496 |
| C4 untouched 32 | 8.124830 |

Deltas are versus direct ternary; lower is better.

| Patch set | k | val delta | W32 delta | C4-32 delta |
|---|---:|---:|---:|---:|
| CE joint | 4 | -0.271754 | -0.208597 | -0.156213 |
| support top-k | 4 | -0.265143 | -0.208540 | -0.161899 |
| signflip top-k | 4 | -0.255191 | -0.202086 | -0.156673 |
| support on joint layers | 4 | -0.265143 | -0.208540 | -0.161899 |
| signflip on joint layers | 4 | -0.242857 | -0.189116 | -0.133153 |
| support on signflip layers | 4 | -0.245137 | -0.204278 | -0.148231 |
| random joint mean | 4 | -- | -0.000443 | -0.000236 |
| CE joint | 6 | -0.322584 | -0.255838 | -0.210322 |
| support top-k | 6 | -0.297465 | -0.222278 | -0.227607 |
| signflip top-k | 6 | -0.302090 | -0.238024 | -0.212017 |
| support on joint layers | 6 | -0.311942 | -0.257343 | -0.212732 |
| signflip on joint layers | 6 | -0.297142 | -0.247238 | -0.180830 |
| signflip on support layers | 6 | -0.278321 | -0.214413 | -0.191746 |
| support on signflip layers | 6 | -0.293882 | -0.244388 | -0.210299 |
| random joint mean | 6 | -- | -0.000803 | -0.000465 |

## 4. Interpretation

Main method gate passes again: CE joint top-4/top-6 improves both untouched WikiText-2 and C4 by large margins and remains far stronger than random joint.

The ternary-specificity gate is partially-to-strongly positive:

- On the same CE joint-selected layers, support relocation beats signflip for both k=4 and k=6 on W32 and C4.
- For k=4, support on joint layers improves W32 by -0.208540 versus -0.189116 for signflip, and C4 by -0.161899 versus -0.133153.
- For k=6, support on joint layers improves W32 by -0.257343 versus -0.247238 for signflip, and C4 by -0.212732 versus -0.180830.
- Cross-layer controls are more mixed: signflip top-k is competitive, and CE joint still chooses some signflip layers. Therefore the correct method claim is not "support-only dominates"; it is "zero-support relocation is a real ternary-specific module inside a joint support/sign editing method."

This is exactly the claim shape we want: not overclaiming support relocation as always best, but showing it is not replaceable by a generic nonzero-only edit.

## 5. Claim Update

Supported after CEGSP-07A:

> Quantized-point CE gradients expose useful ternary edit directions. A zero-support relocation channel contributes real same-layer gains beyond nonzero-only signflip, while the best method remains a small-budget joint support/sign edit selected by validation.

Still missing:

- OPT-125M repetition of the ternary-specificity matched controls.
- Downstream task metrics beyond NLL.
- A comparison to a stronger published PTQ baseline is still needed for final paper positioning.

## 6. Next Minimal Experiment

Run `CEGSP-07B` on OPT-125M with the same matched controls and proportional k values (`2,3`). Gate:

- same-layer support relocation should beat or at least remain competitive with signflip on W32/C4;
- CE joint should remain stronger than random joint;
- if 125M shows a different sign/support balance, keep joint method and state model-size dependence honestly.
