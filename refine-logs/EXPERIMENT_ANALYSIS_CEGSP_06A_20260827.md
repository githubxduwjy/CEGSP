# CEGSP-06A Matched Controls Analysis

日期：2026-08-27

## 1. Question

CEGSP-05B proved that CE-gradient joint ternary edits beat random joint edits. CEGSP-06A further asks whether the advantage comes from:

- CE choosing better layers/edit types;
- CE choosing better intra-layer ternary candidates;
- or both.

The run is strict PTQ-only: no QAT teacher, QAT checkpoint/logits, latent weight update, optimizer step, or TDBT/path-barrier transport.

## 2. Integrity

| Item | Value |
|---|---:|
| Run id | `CEGSP-06A-OPT350M-O0-U32-MATCHED` |
| Remote path | `/root/tqgsp-runs/CEGSP-06A-OPT350M-O0-U32-MATCHED/result.json` |
| Status | complete |
| Elapsed | 88.26 s |
| Model | OPT-350M |
| Layers | 24 |
| Patch sets | 42 |
| Untouched WikiText batches | 32 |
| Untouched C4 batches | 32 |
| Random repeats | 3 |

Local raw artifact pull was blocked by temporary local DNS failure for `xj-member.bitahub.com`; the complete raw `result.json` and log remain on the remote path above.

## 3. Results

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
| random joint mean | 4 | -0.000655 | -0.000443 | -0.000236 |
| random candidates on CE-selected layers mean | 4 | -0.000021 | -0.000513 | -0.000400 |
| CE candidates on random-selected layers mean | 4 | -0.104450 | -0.096204 | -0.145772 |
| CE joint | 6 | -0.322584 | -0.255838 | -0.210322 |
| random joint mean | 6 | -0.001302 | -0.000803 | -0.000465 |
| random candidates on CE-selected layers mean | 6 | +0.000459 | +0.000204 | -0.000166 |
| CE candidates on random-selected layers mean | 6 | -0.093721 | -0.069778 | -0.204841 |

## 4. Interpretation

Primary gate passes again: CE joint top-4/top-6 improves both untouched WikiText-2 and C4, and is far stronger than random joint.

The matched controls separate the mechanism:

- Random candidates placed on CE-selected layers give almost zero gain. Therefore the improvement is not mainly because CE found generally good layers and any edit works there.
- CE candidates placed on random-selected layers still improve clearly, especially on C4. Therefore quantized-point CE gradients produce useful intra-layer ternary edits even when layer/type selection is imperfect.
- Full CE joint is best on WikiText-2 and competitive/best on C4, so CE layer/type selection still matters as an enhancer.

Supported claim after 06A:

> At deployed ternary weights, CE gradients provide useful local information for selecting discrete ternary edits. The main signal is intra-layer candidate quality; validation-based layer/type selection amplifies it.

Not yet supported:

- This does not prove full all-layer editing is safe.
- This does not prove downstream task accuracy gains.
- This does not yet isolate support relocation versus signflip as the unique source of the effect.
- This is still OPT-350M for the matched-control split; OPT-125M should be checked if this becomes a paper-level mechanism claim.

## 5. Next Step

Keep the CEGSP direction fixed. The next minimal useful experiment should test ternary specificity more directly:

- compare CE support relocation against CE signflip under matched layer/type conditions;
- add a binary-like nonzero-only control where the zero support is not allowed to act as a relocation channel;
- repeat only the strongest k values on OPT-350M first, then run OPT-125M if the gate is clean.

Do not start a new method family based on this result.
