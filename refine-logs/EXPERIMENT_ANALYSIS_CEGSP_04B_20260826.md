# Experiment Analysis: CEGSP-04B OPT-125M Second-Model Check

## Summary

- Model: `facebook/opt-125m`
- Runs: `CEGSP-04B-OPT125M-O0/O1/O2`
- Layers: `0..11`
- Proportional top-k: `k ∈ {2,3}`
- Max edits: `64`
- Data: WikiText fit/val/untouched plus report-only C4 validation
- Status: all complete
- Nonfinite: none
- Runtime: `44.81s`, `44.34s`, `50.49s`

CEGSP-04B passes both the primary and stronger second-model gate. Every pre-registered top-k family improves validation, WikiText untouched, and C4 untouched in all three offsets.

## Raw per-offset deltas

Negative is better. Deltas are vs direct ternary.

### O0

| patch set | val Δ | WikiText untouched Δ | C4 untouched Δ |
|---|---:|---:|---:|
| support top2 | -0.214485 | -0.190984 | -0.291913 |
| signflip top2 | -0.207950 | -0.186749 | -0.248861 |
| joint top2 | -0.231834 | -0.203377 | -0.308967 |
| support top3 | -0.277671 | -0.248794 | -0.337107 |
| signflip top3 | -0.263477 | -0.239662 | -0.288907 |
| joint top3 | -0.285708 | -0.256155 | -0.343269 |
| support all | -0.416382 | -0.391074 | -0.471854 |
| signflip all | -0.356786 | -0.351040 | -0.407138 |

### O1

| patch set | val Δ | WikiText untouched Δ | C4 untouched Δ |
|---|---:|---:|---:|
| support top2 | -0.131565 | -0.107056 | -0.130348 |
| signflip top2 | -0.160019 | -0.136777 | -0.139470 |
| joint top2 | -0.160019 | -0.136777 | -0.139470 |
| support top3 | -0.191100 | -0.168393 | -0.188008 |
| signflip top3 | -0.204401 | -0.180070 | -0.205689 |
| joint top3 | -0.217875 | -0.191106 | -0.206334 |
| support all | -0.333879 | -0.354603 | -0.345195 |
| signflip all | -0.323436 | -0.330798 | -0.350971 |

### O2

| patch set | val Δ | WikiText untouched Δ | C4 untouched Δ |
|---|---:|---:|---:|
| support top2 | -0.257019 | -0.222897 | -0.279343 |
| signflip top2 | -0.258469 | -0.233403 | -0.252880 |
| joint top2 | -0.292501 | -0.252668 | -0.310591 |
| support top3 | -0.314707 | -0.275575 | -0.317941 |
| signflip top3 | -0.318883 | -0.294844 | -0.305572 |
| joint top3 | -0.352203 | -0.312365 | -0.356115 |
| support all | -0.554451 | -0.506862 | -0.511382 |
| signflip all | -0.520498 | -0.486236 | -0.476672 |

## Aggregate over three offsets

| patch set | val mean Δ | WikiText untouched mean Δ | C4 untouched mean Δ | triple wins |
|---|---:|---:|---:|---:|
| support top2 | -0.201023 | -0.173646 | -0.233868 | 3/3 |
| signflip top2 | -0.208812 | -0.185643 | -0.213737 | 3/3 |
| joint top2 | -0.228118 | -0.197607 | -0.253009 | 3/3 |
| support top3 | -0.261159 | -0.230921 | -0.281019 | 3/3 |
| signflip top3 | -0.262254 | -0.238192 | -0.266723 | 3/3 |
| joint top3 | -0.285262 | -0.253209 | -0.301906 | 3/3 |
| support all | -0.434904 | -0.417513 | -0.442810 | 3/3 |
| signflip all | -0.400240 | -0.389358 | -0.411594 | 3/3 |

## Gate judgement

Primary gate: pass.

Stronger gate: pass. A single family/k, `joint top3`, improves validation, WikiText untouched, and C4 untouched in 3/3 offsets. In fact every reported top-k family does.

## Interpretation

1. **The signal is not OPT-350M-only.**  
   The same clean-room PTQ-only mechanism works on a second cached model with a proportional layer budget.

2. **The all-layer control diverges from OPT-350M.**  
   On OPT-350M, all-layer editing degraded at `max-edits=64/128`. On OPT-125M, all-layer editing improves all splits. This should not be treated as a direction change; it is a model-scale interaction. A safe paper method should still keep top-k selection because it works on both models, while all-layer editing is not scale-robust.

3. **Joint remains the most consistent top-k story.**  
   `joint top3` has the strongest aggregate top-k deltas: val `-0.285262`, WikiText untouched `-0.253209`, C4 untouched `-0.301906`.

4. **Cost remains PTQ-like in these diagnostics.**  
   Each OPT-125M run completes in about 45–50 seconds on 4090. There are no optimizer steps and no QAT teacher artifacts.

## Updated claim strength

The evidence now supports a two-model diagnostic claim:

> CE-gradient-guided ternary edits at deployed PTQ weights improve held-out NLL across multiple offsets and transfer from WikiText to C4 on OPT-350M and OPT-125M, without QAT teacher/checkpoint/logits or optimizer training.

Still not proven:

- large-model scalability;
- stronger PTQ baseline integration;
- full-length perplexity rather than small-window NLL;
- downstream zero-shot accuracy.

## Next minimal experiment

`CEGSP-05A`: larger untouched evaluation. Keep selection budgets fixed, but increase WikiText/C4 untouched batches from 8 to 32 for one representative offset on both OPT-350M and OPT-125M. This tests whether the observed NLL gains survive larger held-out sample size.
