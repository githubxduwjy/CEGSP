# Experiment Analysis: CEGSP-03B C4 Transfer Offset Replication

## Summary

CEGSP-03B repeated the CEGSP-03A WikiText-to-C4 transfer test on the two remaining offsets from CEGSP-02A:

- `CEGSP-03B-O1`: fit/val/C4 token offset `4096`
- `CEGSP-03B-O2`: fit/val/C4 token offset `8192`

Together with CEGSP-03A/O0, the cross-data transfer evidence now covers three offsets.

All runs completed, used RTX 4090, produced finite metrics, and retained the clean-room PTQ invariants: no QAT checkpoint, no QAT logits, no QAT latent weights, no optimizer steps.

## Per-offset raw deltas

Negative is better; all deltas are vs direct ternary.

### O0

| patch set | val Δ | WikiText untouched Δ | C4 untouched Δ |
|---|---:|---:|---:|
| support top4 | -0.265143 | -0.226320 | -0.149488 |
| signflip top4 | -0.255191 | -0.218416 | -0.145943 |
| joint top4 | -0.271754 | -0.225032 | -0.145396 |
| support top6 | -0.297465 | -0.272000 | -0.192771 |
| signflip top6 | -0.302090 | -0.270071 | -0.198171 |
| joint top6 | -0.322584 | -0.280708 | -0.174876 |

### O1

| patch set | val Δ | WikiText untouched Δ | C4 untouched Δ |
|---|---:|---:|---:|
| support top4 | -0.238042 | -0.239104 | -0.169848 |
| signflip top4 | -0.222209 | -0.218432 | -0.148054 |
| joint top4 | -0.238042 | -0.239104 | -0.169848 |
| support top6 | -0.252782 | -0.295736 | -0.234147 |
| signflip top6 | -0.275745 | -0.274459 | -0.162270 |
| joint top6 | -0.267890 | -0.268749 | -0.295749 |

### O2

| patch set | val Δ | WikiText untouched Δ | C4 untouched Δ |
|---|---:|---:|---:|
| support top4 | -0.136461 | -0.121191 | -0.196211 |
| signflip top4 | -0.201654 | -0.193715 | -0.128883 |
| joint top4 | -0.208867 | -0.194375 | -0.149725 |
| support top6 | -0.149819 | -0.115122 | -0.201254 |
| signflip top6 | -0.188918 | -0.167224 | -0.185233 |
| joint top6 | -0.181360 | -0.156086 | -0.208237 |

## Aggregate over O0/O1/O2

| patch set | val mean Δ | WikiText untouched mean Δ | C4 untouched mean Δ | val wins | W wins | C4 wins |
|---|---:|---:|---:|---:|---:|---:|
| support top4 | -0.213216 | -0.195538 | -0.171849 | 3/3 | 3/3 | 3/3 |
| signflip top4 | -0.226352 | -0.210188 | -0.140960 | 3/3 | 3/3 | 3/3 |
| joint top4 | -0.239554 | -0.219504 | -0.154990 | 3/3 | 3/3 | 3/3 |
| support top6 | -0.233356 | -0.227619 | -0.209390 | 3/3 | 3/3 | 3/3 |
| signflip top6 | -0.255584 | -0.237251 | -0.181891 | 3/3 | 3/3 | 3/3 |
| joint top6 | -0.257278 | -0.235181 | -0.226287 | 3/3 | 3/3 | 3/3 |
| support all layers | +0.061261 | +0.076356 | -0.105682 | 1/3 | 1/3 | 2/3 |
| signflip all layers | +0.006135 | +0.007533 | -0.066253 | 1/3 | 1/3 | 2/3 |

## Gate judgement

Primary CEGSP-03B gate: pass.

Across O0/O1/O2, every pre-registered top-k family at `k ∈ {4, 6}` improves both WikiText untouched and C4 untouched. The strongest aggregate C4 transfer is `joint top6` with mean C4 delta `-0.226287`; the strongest aggregate WikiText untouched delta is `signflip top6` with mean `-0.237251`, close to `joint top6` at `-0.235181`.

## Interpretation

1. **The current positive result is no longer a one-offset artifact.**  
   CEGSP-02A established WikiText robustness; CEGSP-03A/B extend it to report-only C4 transfer. This directly addresses the earlier R043/R046 failure pattern where improvements split between W2 and C4.

2. **Small budget is the stabilizer.**  
   Top4/top6 editing is consistently positive; all-layer editing is not. This should be frozen as a method constraint, not treated as an implementation detail.

3. **The method should remain “ternary edit selection”, not support-only.**  
   Support relocation is ternary-specific because it uses the zero state as an active support variable. Signflip/polarity correction is also important. Joint top6 has the best C4 aggregate, while signflip top6 slightly wins WikiText untouched aggregate. The paper claim should present both as two ternary-native edit channels under one CE-gradient selector.

4. **Cost remains far below QAT in this diagnostic setting.**  
   Each run takes about 50–51 seconds on 4090. The actual gradient/edit work is a small fraction of runtime; C4 streaming/tokenization dominates. This weakens the concern that using quantized-point CE gradients automatically collapses into QAT-like cost.

## Updated method constraint

Freeze the next-stage method shape as:

```text
deployed ternary PTQ
→ one/few CE-gradient batches at quantized weights
→ generate support-relocation and signflip candidate edits
→ rank layers by validation CE/NLL
→ apply only a small top-k layer budget
→ never all-layer edit by default
```

## Next experiments

1. **CEGSP-04A: edit-budget/cost sensitivity.**  
   Keep model/data fixed; sweep `max-edits ∈ {16, 32, 64, 128}` with `k ∈ {4, 6}`. This checks whether the result requires a lucky 64-edit setting and gives a cost-quality curve.

2. **CEGSP-04B: second cached model check.**  
   If `facebook/opt-125m` is cached, run a proportional layer budget test to verify that the signal is not OPT-350M-specific.

3. **Later, not yet:** integrate into a stronger PTQ baseline.  
   Do this only after budget/cost and second-model sanity pass; otherwise integration noise may obscure the mechanism.
