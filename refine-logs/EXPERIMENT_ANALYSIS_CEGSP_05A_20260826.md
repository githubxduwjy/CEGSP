# Experiment Analysis: CEGSP-05A Larger Untouched Holdout

## Summary

CEGSP-05A increased untouched evaluation from 8 batches to 32 batches for both validated models while keeping selection unchanged.

- `CEGSP-05A-OPT350M-O0-U32`
- `CEGSP-05A-OPT125M-O0-U32`

Both runs completed with finite metrics and clean-room invariants intact: no QAT checkpoint, logits, latent weights, state prior, path-barrier transport, or optimizer steps.

## OPT-350M raw deltas

Negative is better. Deltas are vs direct ternary.

| patch set | val Δ | WikiText untouched-32 Δ | C4 untouched-32 Δ |
|---|---:|---:|---:|
| support top4 | -0.265143 | -0.208540 | -0.161899 |
| signflip top4 | -0.255191 | -0.202086 | -0.156673 |
| joint top4 | -0.271754 | -0.208597 | -0.156213 |
| support top6 | -0.297465 | -0.222278 | -0.227607 |
| signflip top6 | -0.302090 | -0.238024 | -0.212017 |
| joint top6 | -0.322584 | -0.255838 | -0.210322 |
| support all | +0.121583 | +0.193845 | -0.051798 |
| signflip all | +0.038583 | +0.103298 | -0.030301 |

## OPT-125M raw deltas

| patch set | val Δ | WikiText untouched-32 Δ | C4 untouched-32 Δ |
|---|---:|---:|---:|
| support top2 | -0.214485 | -0.241514 | -0.356439 |
| signflip top2 | -0.207950 | -0.228517 | -0.296202 |
| joint top2 | -0.231834 | -0.256313 | -0.372047 |
| support top3 | -0.277671 | -0.297697 | -0.405734 |
| signflip top3 | -0.263477 | -0.277491 | -0.341346 |
| joint top3 | -0.285708 | -0.303213 | -0.410006 |
| support all | -0.416382 | -0.462616 | -0.559504 |
| signflip all | -0.356786 | -0.395652 | -0.475813 |

## Gate judgement

Primary gate: pass.

- OPT-350M: all pre-registered top-k variants improve validation, WikiText untouched-32, and C4 untouched-32.
- OPT-125M: all pre-registered top-k variants improve validation, WikiText untouched-32, and C4 untouched-32.

Stronger joint gate: pass.

- OPT-350M `joint top6`: val `-0.322584`, W32 `-0.255838`, C4-32 `-0.210322`.
- OPT-125M `joint top3`: val `-0.285708`, W32 `-0.303213`, C4-32 `-0.410006`.

## Interpretation

1. **The improvements survive larger held-out sample size.**  
   The 8-batch results were not a tiny-window artifact in the tested O0 setting.

2. **Top-k is scale-robust; all-layer is not.**  
   OPT-350M all-layer editing still hurts WikiText even when C4 improves. OPT-125M all-layer improves. Therefore the paper method should retain top-k as the default robust design and discuss all-layer as model-scale dependent.

3. **Joint remains the cleanest headline variant.**  
   Joint top-k passes on both models and both larger held-out distributions. It is not always the single best C4 number, but it is the most coherent method story because it allows both ternary support relocation and polarity correction under one selector.

4. **Next missing evidence is a random/edit-control NLL baseline.**  
   We now know CE-gradient top-k improves held-out NLL. We still need to prove that CE-gradient direction matters, rather than merely showing that low-budget edits to a bad direct-ternary model often help.

## Next minimal experiment

`CEGSP-05B`: random-control NLL baseline. For the frozen O0/U32 setting, compare CE-gradient top-k edits against same-budget random support/signflip edits on the same selected layer counts. This should be done before larger model integration.
