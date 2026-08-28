# CEGSP-08A Cloze Sanity Analysis

日期：2026-08-27

## 1. Question

CEGSP-08A tests whether the robust NLL improvements from CEGSP have any small downstream-style signal on a LAMBADA-style last-token cloze task.

This is a sanity check, not a full downstream benchmark.

## 2. Integrity

| Item | OPT-125M | OPT-350M |
|---|---:|---:|
| Run id | `CEGSP-08A-OPT125M-O0-U32-CLOZE` | `CEGSP-08A-OPT350M-O0-U32-CLOZE` |
| Status | complete | complete |
| Elapsed | 63.1 s | 79.4 s |
| Cloze source | `lambada:validation` | `lambada:validation` |
| Cloze examples | 128 | 128 |
| QAT teacher/logits/checkpoint/optimizer | none | none |

Raw remote paths:

- `/root/tqgsp-runs/CEGSP-08A-OPT125M-O0-U32-CLOZE/result.json`
- `/root/tqgsp-runs/CEGSP-08A-OPT350M-O0-U32-CLOZE/result.json`

## 3. Results

### OPT-125M

| System | W32 delta | C4 delta | Cloze NLL | Top1 | Top5 |
|---|---:|---:|---:|---:|---:|
| FP | -- | -- | 2.090691 | 0.625000 | 0.726563 |
| direct ternary | 0 | 0 | 10.816895 | 0.000000 | 0.000000 |
| CE joint top2 | -0.256313 | -0.372047 | 10.887460 | 0.000000 | 0.000000 |
| CE joint top3 | -0.303213 | -0.410006 | 10.898307 | 0.000000 | 0.000000 |

### OPT-350M

| System | W32 delta | C4 delta | Cloze NLL | Top1 | Top5 |
|---|---:|---:|---:|---:|---:|
| FP | -- | -- | 1.979932 | 0.656250 | 0.734375 |
| direct ternary | 0 | 0 | 11.529533 | 0.000000 | 0.000000 |
| CE joint top4 | -0.208597 | -0.156213 | 11.177016 | 0.000000 | 0.000000 |
| CE joint top6 | -0.255838 | -0.210322 | 11.222092 | 0.000000 | 0.000000 |

## 4. Interpretation

Primary NLL gate passes again for both models: CE joint improves W32 and C4 NLL.

The hard cloze accuracy gate is not informative:

- Direct ternary collapses LAMBADA last-token top1/top5 to zero on both OPT-125M and OPT-350M.
- Since the baseline is already zero, CE joint cannot show accuracy improvement under this hard metric.
- CE joint does not further reduce top1/top5 because both remain at zero, but this is a floor effect rather than positive task evidence.

Cloze NLL is more informative:

- OPT-350M improves cloze NLL from 11.529533 to 11.177016 at top4 and 11.222092 at top6.
- OPT-125M worsens cloze NLL from 10.816895 to 10.887460/top2 and 10.898307/top3 despite improving W32/C4 NLL.

Therefore, CEGSP-08A should be read as diagnostic:

> NLL improvements on Wikitext/C4 are robust, but LAMBADA hard accuracy is too brittle under direct ternary PTQ. For downstream evidence, the next evaluation should use smoother ranking/log-likelihood metrics or easier multiple-choice tasks before claiming task-level gains.

## 5. Claim Impact

Supported:

- CEGSP remains robust on NLL.
- On OPT-350M, CEGSP also improves LAMBADA-style cloze NLL.
- CEGSP does not create a visible hard-accuracy recovery from a zero-accuracy direct ternary baseline.

Not supported:

- We cannot claim downstream accuracy improvement from this run.
- We should not use LAMBADA top1/top5 as the main downstream table for 1.58-bit PTQ in this setup.

## 6. Next Experiment

The next downstream experiment should avoid the zero-accuracy floor. Recommended:

- Use multiple-choice log-likelihood ranking on a small subset of HellaSwag/PIQA/ARC-Easy, where each option is scored by conditional NLL.
- Report normalized choice NLL and accuracy.
- Evaluate only FP, direct ternary, and best CE joint top-k.

This keeps the task metric smoother and closer to standard LM evaluation without introducing QAT or training.
