# AutoResearch Summary: PTQ 1.58-bit, 2026-08-22

## Outcome

The new hard-discrete `T` update has a real local mechanism but is not yet a valid model-level PTQ method. The block-level hypothesis passed; both model-transfer gates failed. The project should pivot from block activation NMSE acceptance to hierarchical layer/sequence-level function acceptance.

## Evidence table

| Run | Variant | Primary result | Gate |
|---|---|---|---|
| R042c | validation-gated `T`, all-calibration grid refit | unseen block NMSE median +5.83%, mean +12.14%, win 96.43%; matched ungated median +5.02%, win 87.50% | PASS |
| R043a | gated `T` on layers 0/10/20/31 inside official GPTQ | W2 NaN, C4 56.0225 vs 25.8104/66.7370 | FAIL |
| R044a | gated `T` on layer 0 only | W2/C4 26.2403/57.9333; W2 -1.67%, C4 +13.19% relative improvement | FAIL |
| R045 | layer-0 hard-`T` vs fixed-`T` inside FP16 rest | hard-`T` wins all predeclared held-out train sequence metrics; no metric rejects it | FAIL |
| R046 | same layer-0 hard-`T` inside full quantized context | strict gate fails; W2 mean NLL/CVaR worse (+0.0479/+0.1644), C4 better (-0.1143/-0.1328) | FAIL/DIAGNOSTIC |

R043a quantization took 144.6 seconds and peaked at 9.13 GiB. All 1,112 refined blocks had finite grids and zero fallback rows. R044a's stable-ITF integration also passed targeted degenerate-row tests, so the transfer failure cannot be dismissed as a numerical implementation artifact.

## Scientific interpretation

The useful finding is not simply that updating `T` works. It is that sparse hard reassignment can improve held-out local linear-map error while worsening or destabilizing autoregressive language modeling. This falsifies the assumption that a block Hessian objective is a sufficient proxy for function preservation at ternary precision.

The C4/WikiText2 split is also informative: both model runs strongly improved C4 while failing WikiText2. The update is distribution-sensitive and likely changes rare-token or long-tail activation trajectories that average covariance loss underweights.

## Revised method direction

Working name: hierarchical trajectory-gated ternarization.

1. Use Hessian/activation-aware hard `T` search only as a sparse proposal generator.
2. Group proposals into a bounded layer-level trust region; do not accept every row-wise local improvement.
3. Compare candidate versus fixed-`T` ATQ on disjoint validation sequences using full transformer-layer output plus mean and tail-token NLL.
4. Accept a candidate tranche only if it improves the higher-level objective; freeze `T`, then refit `alpha,mu` on all calibration data.
5. Keep final PPL and task data untouched until the hierarchy has selected all layers.

This differs from merely adding more losses to AGA: the higher-level signal controls discrete structure acceptance and can reject locally attractive but trajectory-harmful ternary codes.

## Immediate next experiment

R045/R046 show that the next gate cannot be a simple held-out reconstruction or single-distribution sequence score. The layer-0 hard-`T` candidate is locally and FP16-context better, while the full quantized context reproduces a distribution split: worse on WikiText2 windows and better on C4 windows. The next algorithmic step should make acceptance robust across calibration distributions, e.g. a minimax or Pareto gate over W2-like and C4-like held-out streams, before any larger all-layer search.

## Claim status

Deterministic claim boundary: local mechanism supported; model-level superiority unsupported. The mandatory external `result-to-claim` reviewer timed out after 300 seconds, so the formal paper-level verdict is `REVIEW_UNAVAILABLE`. Seven cited numeric values passed deterministic evidence pre-check; independent integrity status remains unavailable.

## Artifacts

- `results/remote-runs/hessian_gated_r042b_ns12_20260822/`
- `results/remote-runs/hessian_gated_r042c_ns12_20260822/`
- `results/remote-runs/gated_t_r043a_l0_10_20_31_ns8_20260822_retry/`
- `results/remote-runs/gated_t_r044a_l0_ns8_20260822/`
- `results/remote-runs/r045_trajectory_gate_20260822/`
- `results/remote-runs/r046_contextual_sequence_gate_20260822/`
- `remote-tools/hessian_gated_ternary_diagnostics_r042c.py`
- `remote-tools/gated_t_gptq_quantize.py`
- `remote-tools/r045_trajectory_gate.py`
- `remote-tools/r046_contextual_sequence_gate.py`
- `findings.md`
- `CLAIMS_FROM_RESULTS.md`
