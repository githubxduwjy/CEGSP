# Experiment Analysis: TQGSP-02A CE-Aware Layer Selection

日期：2026-08-26

## 1. Run Integrity

- Run ID: `TQGSP-02A`
- Remote: `root@xj-member.bitahub.com:42181`
- GPU: RTX 4090 24GB
- Model: `facebook/opt-350m`
- Layers: all decoder layers `0..23`
- Operator: Q/K composed operator
- Data source: `wikitext-2-raw-v1-arrow-cache-after-ImportError`
- Fit / val / untouched batches: `16 / 8 / 8`
- Elapsed: `20.02 s`
- Result path: `results/remote-runs/TQGSP-02A/result.json`
- Console path: `results/remote-runs/TQGSP-02A/console.log`

Clean-room invariants:

- `uses_qat_checkpoint = false`
- `uses_qat_logits = false`
- `uses_qat_latent_weights = false`
- `uses_qat_state_prior = false`
- `uses_path_barrier_or_tdbt_transport = false`
- `uses_quantized_point_operator_gradient = true`
- `uses_ce_selection_gate = true`

## 2. Main Results

Baseline NLL:

| Model | Val NLL | Untouched NLL |
|---|---:|---:|
| FP | 3.8039 | 3.9876 |
| direct ternary PTQ | 8.6946 | 8.9900 |

Patch-set NLL:

| Patch set | Layers | Val delta vs direct | Untouched delta vs direct |
|---|---:|---:|---:|
| all-tqgsp-qk | 24 | +0.1235 | +0.1143 |
| operator-topk-qk | 6 | +0.0261 | +0.0258 |
| ce-selected-qk | 9 | -0.0325 | -0.0157 |
| ce-topk-qk | 6 | -0.0322 | -0.0093 |

Correlation between operator gain and single-layer val NLL delta:

- Pearson: `+0.1439`
- Spearman: `+0.0496`

Because lower NLL delta is better, a useful operator proxy should have a negative correlation. The observed correlation is weak and wrong-signed.

## 3. Per-Layer Summary

| Layer | Operator gain val | Operator gain untouched | Single-patch val NLL delta | Selected |
|---:|---:|---:|---:|---|
| 0 | +6.68% | +6.53% | -0.000285 | yes |
| 1 | +13.08% | +15.60% | -0.000091 | yes |
| 2 | +49.14% | +52.13% | -0.001057 | yes |
| 3 | +52.75% | +57.08% | +0.009574 | no |
| 4 | +47.06% | +49.92% | -0.000329 | yes |
| 5 | +56.63% | +57.37% | +0.003094 | no |
| 6 | +14.48% | +15.34% | +0.009555 | no |
| 7 | +33.68% | +33.94% | +0.005202 | no |
| 8 | +33.97% | +32.81% | +0.013721 | no |
| 9 | +48.08% | +49.46% | +0.008725 | no |
| 10 | +55.02% | +56.42% | +0.007011 | no |
| 11 | +50.48% | +49.61% | -0.000826 | yes |
| 12 | +48.46% | +52.81% | +0.024382 | no |
| 13 | +41.80% | +43.97% | +0.007375 | no |
| 14 | +46.63% | +48.44% | +0.014419 | no |
| 15 | +52.72% | +55.12% | -0.007115 | yes |
| 16 | +49.41% | +45.12% | +0.015858 | no |
| 17 | +57.17% | +46.96% | -0.014328 | yes |
| 18 | +62.84% | +62.86% | +0.014364 | no |
| 19 | +48.80% | +54.91% | +0.004159 | no |
| 20 | +20.95% | +20.50% | -0.009091 | yes |
| 21 | +41.61% | +42.77% | -0.000633 | yes |
| 22 | +38.23% | +38.66% | +0.002219 | no |
| 23 | +56.26% | +55.24% | +0.006068 | no |

## 4. Findings

1. Operator proxy alone is not reliable.

   `operator-topk-qk` degrades untouched NLL by `+0.0258`, despite selecting layers with high operator gain. The correlation between operator gain and single-layer CE delta is near zero and wrong-signed.

2. Blindly patching all Q/K layers is harmful.

   `all-tqgsp-qk` degrades untouched NLL by `+0.1143`. This rules out the naive scaling strategy.

3. CE-aware selection is useful.

   `ce-selected-qk` improves val NLL by `-0.0325` and untouched NLL by `-0.0157`. This is the strongest current end-to-end result in the clean-room TQG-SP branch.

4. The method should pivot from operator-gradient support projection to CE-gradient support projection.

   The useful gate is CE/NLL, not operator NMSE. Since Li et al. emphasize gradients at quantized weights biasing toward low-loss basins, the next method should use CE gradients directly at the deployed ternary point.

## 5. Decision

```text
Proxy reliability gate: FAIL.
CE-aware selection gate: PASS.
Next direction: direct CE-gradient ternary support projection.
```

Do not scale operator-only TQG-SP to larger models yet.

## 6. Next Experiment

Recommended immediate next run:

```text
CEGSP-01A: CE-gradient support projection at the deployed ternary point
```

Minimal design:

- Keep `facebook/opt-350m`.
- Use all Q/K layers.
- Apply direct ternary PTQ to the whole model.
- Compute CE gradients at the deployed ternary point on small Wikitext calibration batches.
- For each layer, generate budget-matched ternary support swaps using CE gradients.
- Compare:
  - direct ternary
  - all CE-GSP Q/K
  - CE-selected CE-GSP Q/K
  - CE-topk CE-GSP Q/K

Gate:

- If CE-GSP selected set improves untouched NLL more reliably than operator TQG-SP selected set, the paper should pivot to CE-gradient support projection as the main method.

