# Experiment Analysis: CEGSP-02A Split/Offset Robustness

日期：2026-08-26

## 1. Run Integrity

- Run group: `CEGSP-02A`
- Remote: `root@xj-member.bitahub.com:42126`
- GPU: RTX 4090 24GB
- Model: `facebook/opt-350m`
- Layers: all decoder layers `0..23`
- Matrices: Q/K
- Data source: Wikitext-2 Arrow cache
- Fit / val / untouched batches: `8 / 8 / 8`
- k sweep: `4,6,8,12`
- Offsets:
  - `CEGSP-02A-O0`: fit offset `0`, val offset `0`
  - `CEGSP-02A-O1`: fit offset `4096`, val offset `4096`
  - `CEGSP-02A-O2`: fit offset `8192`, val offset `8192`

All three runs completed and remained strict PTQ:

- no QAT checkpoint/logits/teacher;
- no optimizer steps;
- no TDBT path/barrier;
- CE gradient is computed at deployed ternary weights only.

## 2. Aggregate k-Sweep Results

Untouched NLL delta versus direct ternary PTQ:

| Family | k | Mean Δ | Std | Wins |
|---|---:|---:|---:|---:|
| joint | 4 | -0.2195 | 0.0187 | 3/3 |
| joint | 6 | -0.2352 | 0.0561 | 3/3 |
| joint | 8 | -0.2152 | 0.1282 | 3/3 |
| joint | 12 | -0.1450 | 0.2056 | 2/3 |
| signflip | 4 | -0.2102 | 0.0116 | 3/3 |
| signflip | 6 | -0.2373 | 0.0495 | 3/3 |
| signflip | 8 | -0.1714 | 0.1624 | 2/3 |
| signflip | 12 | -0.1595 | 0.2085 | 2/3 |
| support | 4 | -0.1955 | 0.0528 | 3/3 |
| support | 6 | -0.2276 | 0.0801 | 3/3 |
| support | 8 | -0.1829 | 0.1452 | 2/3 |
| support | 12 | -0.1663 | 0.2440 | 2/3 |

## 3. Raw Best Per Offset

| Offset | Best family | k | Val Δ | Untouched Δ | Selected layers |
|---|---|---:|---:|---:|---|
| O0 | joint | 6 | -0.3226 | -0.2807 | `[22,16,13,19,12,17]` |
| O1 | signflip | 12 | -0.3406 | -0.4299 | `[17,13,19,11,22,4,9,12,5,7,6,16]` |
| O2 | joint | 4 | -0.2089 | -0.1944 | `[17,13,9,23]` |

## 4. Key Findings

1. The CE-gradient editing direction is robust.

   Every family at k = 4 and k = 6 improves untouched NLL in all three offsets. This is the first robust evidence that the method is not merely an offset-0 artifact.

2. Small layer budgets are the stable region.

   k = 4 and k = 6 are stable. k = 8 remains positive for joint but becomes unstable for support/signflip. k = 12 is offset-dependent and fails on O2 for several families.

3. Joint is useful but not universally dominant.

   Joint has the best mean at k = 6 among joint rows, while signflip k = 6 has a slightly better mean untouched delta overall (`-0.2373` vs joint `-0.2352`). The evidence supports a joint edit family, but not a claim that joint always wins.

4. Support-only is not enough.

   Support-only is positive at k = 4/6, but signflip is competitive or better in some offsets. The final method should include both ternary zero-support relocation and nonzero polarity correction.

5. The research direction is now better constrained, not randomly changed.

   The locked direction remains:

   > Quantized-point CE-gradient guided ternary editing for PTQ-only gap reduction.

   The experiments update module confidence inside that family; they do not change the main problem.

## 5. Decision

```text
Robustness gate: PASS.
Stable budget region: k = 4 or 6.
Method shape: CE-gradient ternary edit selection with support + signflip candidates.
Do not claim support-only.
Do not use all-layer editing.
Do not scale before testing C4 / second model.
```

## 6. Next Experiment

Recommended next run:

```text
CEGSP-03A: cross-data transfer
```

Minimal design:

- Keep OPT-350M.
- Use Wikitext calibration.
- Evaluate on Wikitext untouched and C4 validation if cached/available.
- Test only stable k: `4` and `6`.
- Compare:
  - direct ternary;
  - signflip top-k;
  - support top-k;
  - joint top-k.

Gate:

- If Wikitext gains transfer to C4, the method becomes much more paper-credible.
- If C4 fails, add holdout/data-alignment regularization before scaling.

