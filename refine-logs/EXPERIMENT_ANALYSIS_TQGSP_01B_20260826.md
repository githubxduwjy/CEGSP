# Experiment Analysis: TQGSP-01B

日期：2026-08-26

## 1. Run Integrity

- Run ID: `TQGSP-01B`
- Remote: `root@xj-member.bitahub.com:42181`
- GPU: RTX 4090 24GB
- Model: `facebook/opt-350m`
- Layers: `0,7,15,23`
- Operator: `qk`
- Data source: `wikitext-2-raw-v1-arrow-cache-after-ImportError`
- Fit / val / untouched batches: 16 / 8 / 8
- Elapsed: `14.66 s`
- Result path: `results/remote-runs/TQGSP-01B/result.json`

Clean-room invariants:

- `uses_qat_checkpoint = false`
- `uses_qat_logits = false`
- `uses_qat_latent_weights = false`
- `uses_qat_state_prior = false`
- `uses_path_barrier_or_tdbt_transport = false`
- `uses_quantized_point_operator_gradient = true`

This is a strict PTQ validation run. It does not use a QAT teacher.

## 2. Why 01B Supersedes 01A

`TQGSP-01A` exposed an edit-budget mismatch: `NZ-signflip-G` accepted 64 edits, while `TQGSP-support-G` accepted only 11–14 support swaps because pairwise candidates reused donors/receivers. Therefore `01A` is treated as a harness diagnostic.

`01B` fixes only this fairness issue by using unique donor/receiver support exchanges. Model, data, layers, operator, and metrics are unchanged.

## 3. Operator NMSE Results

| Layer | Base val | Variant | Val NMSE | Val improvement | Untouched NMSE | Untouched improvement | Edits |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 0.102066 | support-random | 0.102160 | -0.09% | 0.102438 | -0.09% | 64 |
| 0 | 0.102066 | support-forward | 0.100782 | +1.26% | 0.101070 | +1.25% | 64 |
| 0 | 0.102066 | TQGSP-support-G | 0.095253 | +6.68% | 0.095660 | +6.53% | 64 |
| 0 | 0.102066 | NZ-signflip-G | 0.095152 | +6.77% | 0.095625 | +6.57% | 64 |
| 7 | 0.161582 | support-random | 0.161848 | -0.16% | 0.163831 | -0.15% | 64 |
| 7 | 0.161582 | support-forward | 0.163538 | -1.21% | 0.165092 | -0.92% | 64 |
| 7 | 0.161582 | TQGSP-support-G | 0.107157 | +33.68% | 0.108066 | +33.94% | 64 |
| 7 | 0.161582 | NZ-signflip-G | 0.111190 | +31.19% | 0.111245 | +32.00% | 64 |
| 15 | 0.176913 | support-random | 0.177015 | -0.06% | 0.184142 | -0.01% | 64 |
| 15 | 0.176913 | support-forward | 0.175228 | +0.95% | 0.177794 | +3.44% | 64 |
| 15 | 0.176913 | TQGSP-support-G | 0.083642 | +52.72% | 0.082629 | +55.12% | 64 |
| 15 | 0.176913 | NZ-signflip-G | 0.088606 | +49.92% | 0.088832 | +51.75% | 64 |
| 23 | 0.167581 | support-random | 0.167533 | +0.03% | 0.177248 | +0.37% | 64 |
| 23 | 0.167581 | support-forward | 0.164408 | +1.89% | 0.172211 | +3.20% | 64 |
| 23 | 0.167581 | TQGSP-support-G | 0.073308 | +56.26% | 0.079638 | +55.24% | 64 |
| 23 | 0.167581 | NZ-signflip-G | 0.087008 | +48.08% | 0.091135 | +48.78% | 64 |

Mechanism gate:

- `TQGSP-support-G` beats all three controls on untouched operator NMSE in `3/4` layers.
- The only loss is layer 0, where `NZ-signflip-G` is marginally better: `0.095625` vs `0.095660`.

Decision:

```text
Mechanism gate: PASS, with layer-0 caveat.
```

## 4. End-to-End NLL

| Variant | Val NLL | Delta vs direct | Untouched NLL | Delta vs direct |
|---|---:|---:|---:|---:|
| direct-ternary | 8.694630 | 0.000000 | 8.989967 | 0.000000 |
| support-random | 8.694603 | -0.000027 | 8.990345 | +0.000379 |
| support-forward | 8.697208 | +0.002578 | 8.999558 | +0.009591 |
| TQGSP-support-G | 8.696226 | +0.001597 | 8.984388 | -0.005579 |
| NZ-signflip-G | 8.703313 | +0.008683 | 8.994935 | +0.004969 |

Transfer gate:

- Untouched split improves for `TQGSP-support-G` by `-0.00558` NLL versus direct PTQ.
- Val split slightly worsens by `+0.00160` NLL.
- Both changes are small at this sample size.

Decision:

```text
Transfer gate: PASS_WEAK / MIXED_POSITIVE.
```

The result is enough to justify one follow-up, but not enough to claim end-to-end improvement yet.

## 5. Cost

Wall-clock breakdown:

| Component | Seconds |
|---|---:|
| tokenizer + data | 9.99 |
| model load | 1.49 |
| FP hidden collection | 0.45 |
| proxy validation | 1.64 |
| direct PTQ apply | 0.05 |
| direct PTQ NLL eval | 0.13 |
| each variant NLL eval | ~0.12 |
| total elapsed | 14.66 |

Peak CUDA memory:

- `1.21 GB`

Interpretation:

- This run is far from QAT-like cost.
- Most time is data/model I/O, not gradient projection.
- The method-specific proxy stage is `1.64 s` for 4 Q/K layer pairs with one gradient batch.

## 6. Key Findings

1. Budget matching materially strengthens TQG-SP.

   Once support swaps can consume the same 64-edit budget as controls, `TQGSP-support-G` improves Q/K operator NMSE by `+33.94%`, `+55.12%`, and `+55.24%` on layers 7/15/23 untouched, beating signflip at matched budget.

2. The ternary-zero support claim is plausible but not yet universal.

   Layer 0 remains ambiguous: signflip is marginally better than support projection. This suggests early layers may prefer nonzero sign correction, while deeper Q/K layers benefit more from support relocation.

3. Forward/magnitude salience is weak.

   `support-forward` is much weaker than gradient support projection and even degrades layer 7. The useful signal is specifically the quantized-point gradient, not just support movement.

4. End-to-end NLL transfer is real enough to continue, but not yet publishable.

   Untouched NLL improves slightly, but val NLL slightly worsens. This suggests the proxy is not nonsense, but larger validation and CE-aware selection are needed before making model-level claims.

5. Cost is safely PTQ-like in this small setting.

   There is no evidence of QAT-like cost explosion. The run used one quantized-point backward and completed quickly on 4090.

## 7. Next Experiment Direction

The next run should not expand blindly to all layers. It should test whether the operator-to-NLL transfer becomes more reliable when the support projection objective is closer to language-model loss.

Recommended next run:

```text
TQGSP-02A: CE-aware layer/operator selection
```

Minimal design:

- Keep `facebook/opt-350m`.
- Candidate layers: all Q/K layers or a denser layer grid.
- For each layer, compute TQG-SP operator gain on calibration.
- Patch one layer at a time and measure small validation CE/NLL delta.
- Select only layers whose operator gain and CE delta agree.
- Evaluate selected-layer patch set on untouched Wikitext.

Stop condition:

- If operator gain repeatedly fails to predict CE/NLL delta, abandon pure operator objective and switch to direct CE-gradient support projection.

