# CEGSP-V2-P0-OPT125M Analysis

Date: 2026-08-27

## Purpose

This run follows the v2 CEGSP framing from the optimized DOCX:

- CEGSP is a QAT-inspired, post-training, optimizer-free ternary discrete repair method.
- It uses the same information source emphasized by the QAT basin paper: CE gradients evaluated at quantized weights.
- The key question is not only whether CEGSP improves direct ternary PTQ, but whether its one-shot ternary support exchange has independent value relative to matched One-Step and short Multi-Step latent QAT controls.

## Configuration

- Remote: `root@xj-member.bitahub.com:42188`
- GPU: RTX 4090 24GB
- Run id: `CEGSP-V2-P0-OPT125M`
- Remote result: `/root/tqgsp-runs/CEGSP-V2-P0-OPT125M/result.json`
- Model: `facebook/opt-125m`
- Dataset: Wikitext-2 raw
- Fit / val / untouched batches: 8 / 8 / 16
- Quantization: direct ternary, group size 128, rho 0.7
- CEGSP: Q/K support exchange, alpha frozen, 64 edits per candidate layer, top-3 layers selected on validation
- QAT controls: latent FP Q/K update controls with 1 and 4 steps; eta sweep `{0, 0.01, 0.03, 0.1, 0.3, 1.0}`
- Score-validity: layers `{0, 6, 11}`, 24 candidates per layer, 72 candidates total

## Main Results

| Method | Val NLL | Untouched W2 NLL | Delta vs Direct Val | Delta vs Direct Untouched |
|---|---:|---:|---:|---:|
| FP | 4.0822 | 4.0961 | - | - |
| Direct ternary | 9.7031 | 9.6952 | 0 | 0 |
| CEGSP top-3 | 9.4238 | 9.4316 | -0.2793 | -0.2636 |
| One-Step QAT best | 9.0706 | 9.0134 | -0.6325 | -0.6818 |
| Four-Step QAT best | 8.7086 | 8.7166 | -0.9945 | -0.9787 |

Selected CEGSP layers: `[1, 11, 8]`.

Gap closure ratio on untouched Wikitext-2 relative to the best 4-step QAT control:

```text
R_gap = (Direct - CEGSP) / (Direct - best Multi-Step QAT)
      = 0.2693
```

So this run supports a positive but incomplete gap-recovery claim: CEGSP recovers about 27% of the short-QAT reachable untouched NLL improvement under this small OPT-125M setting.

## Score-Validity Result

The quantized-point CE score is strongly predictive in this run:

- Candidates evaluated: 72
- Spearman(score, actual improvement): `0.7555`
- Top-10% true improvement rate: `1.0000`
- All-candidate true improvement rate: `0.9028`
- Top-10% mean delta val NLL: `-0.00439`
- All-candidate mean delta val NLL: `-0.00128`

This is the cleanest evidence from the run. It directly supports the v2 mechanism statement:

> Gradients evaluated at deployed ternary weights contain actionable local information for legal ternary discrete moves.

## Interpretation

Positive evidence:

1. CEGSP still improves direct ternary PTQ on both validation and untouched Wikitext-2.
2. The predicted discrete move score is meaningfully aligned with actual candidate-level CE improvement.
3. The method remains optimizer-free and teacher-free; the QAT controls are only baselines, not part of CEGSP.

Negative or limiting evidence:

1. One-Step QAT is substantially stronger than current CEGSP in this setting.
2. Four-Step QAT is stronger still, so CEGSP cannot yet claim to close most of the PTQ-QAT gap.
3. Current fixed-cardinality Q/K support exchange is useful, but probably too constrained relative to latent QAT.

## Claim Boundary After This Run

Supported:

> CEGSP can convert quantized-point CE gradients into useful one-shot ternary support edits, improving direct ternary PTQ while avoiding QAT teachers, latent trajectories, and optimizer steps.

Not yet supported:

> CEGSP matches One-Step QAT.

Not yet supported:

> CEGSP closes most of the PTQ-QAT gap.

## Next Experiment

The next run should not pivot away from CEGSP. It should test whether the gap to One-Step QAT is caused by an overly narrow move space.

Recommended next experiment: `CEGSP-V2-P1-MOVE-SPACE`

- Same model/data budget as this run.
- Keep score-validity.
- Compare these same-gradient one-shot discrete actions:
  - support relocation only
  - nonzero signflip only
  - mixed support relocation + signflip
  - support relocation with alpha refit as a controlled variant
  - slightly larger top-k layer budget `{3, 6}`
- Keep One-Step QAT as the main upper control.

Gate:

- Mixed CEGSP should close at least 40% of One-Step QAT's untouched improvement, or beat support-only by a clear margin.
- If mixed CEGSP is still far below One-Step QAT, the paper should present CEGSP as a cheap partial repair method, not a near-QAT substitute.

