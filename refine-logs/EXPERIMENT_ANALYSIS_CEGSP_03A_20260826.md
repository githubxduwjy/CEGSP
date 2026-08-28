# Experiment Analysis: CEGSP-03A C4 Transfer

## Summary

- Run ID: `CEGSP-03A-C4TRANSFER`
- Result path: `results/remote-runs/CEGSP-03A-C4TRANSFER/result.json`
- Console path: `results/remote-runs/CEGSP-03A-C4TRANSFER/console.log`
- Status: `complete`
- Wall-clock: `51.31s`
- GPU: RTX 4090 24GB
- Peak CUDA memory allocated: `1.20 GB`
- Nonfinite: none detected

CEGSP-03A passes the pre-registered cross-data transfer gate. The layer/edit selection used only WikiText fit/validation. C4 validation was report-only and was not used to choose `k`, layers, threshold, or edit family.

## Clean-room invariants

The result file records:

- `uses_qat_checkpoint = false`
- `uses_qat_logits = false`
- `uses_qat_latent_weights = false`
- `uses_qat_state_prior = false`
- `uses_optimizer_steps = false`
- `uses_ce_gradient_at_quantized_weights = true`

This remains a PTQ-only diagnostic; no QAT teacher is used.

## Baselines

| split | FP NLL | direct ternary NLL |
|---|---:|---:|
| WikiText val | 3.803895 | 8.694630 |
| WikiText untouched | 3.987563 | 8.989967 |
| C4 untouched | 3.438045 | 8.222888 |

## Patch-set deltas vs direct ternary

Negative is better.

| patch set | layers | val Δ | WikiText untouched Δ | C4 untouched Δ |
|---|---:|---:|---:|---:|
| `ksweep-support-top4-qk` | 4 | -0.265143 | -0.226320 | -0.149488 |
| `ksweep-signflip-top4-qk` | 4 | -0.255191 | -0.218416 | -0.145943 |
| `ksweep-joint-top4-qk` | 4 | -0.271754 | -0.225032 | -0.145396 |
| `ksweep-support-top6-qk` | 6 | -0.297465 | -0.272000 | -0.192771 |
| `ksweep-signflip-top6-qk` | 6 | -0.302090 | -0.270071 | -0.198171 |
| `ksweep-joint-top6-qk` | 6 | -0.322584 | -0.280708 | -0.174876 |
| `cegsp-support-all-qk` | 24 | +0.121583 | +0.133234 | +0.016003 |
| `ce-signflip-all-qk` | 24 | +0.038583 | +0.052664 | +0.037108 |

Selected joint layers:

- `joint top4`: layer 22 signflip; layers 16, 13, 19 support.
- `joint top6`: layer 22 signflip; layers 16, 13, 19, 12 support; layer 17 signflip.

## Gate judgement

Primary transfer gate: pass.

- `ksweep-joint-top4-qk`: val `-0.271754`, WikiText untouched `-0.225032`, C4 untouched `-0.145396`.
- `ksweep-joint-top6-qk`: val `-0.322584`, WikiText untouched `-0.280708`, C4 untouched `-0.174876`.

Both pre-registered joint candidates improve both untouched distributions. Therefore the result is not merely a WikiText-only calibration artifact in this single offset.

## Interpretation

1. **Cross-data transfer is real in this run.**  
   C4 was not used for selection, yet all top4/top6 variants improve C4 NLL. This directly addresses the risk raised after earlier W2/C4 split failures.

2. **Layer budget remains essential.**  
   All-layer editing still degrades WikiText and C4. The method is not “apply CE edit everywhere”; it is “CE-gradient ternary edit plus small layer budget”.

3. **Support and signflip are both useful.**  
   On C4 top6, signflip-only is best (`-0.198171`), support-only is close (`-0.192771`), and joint is strongest on WikiText but not C4. This means the method claim should not be support-only; it should be ternary edit selection over support relocation and polarity correction.

4. **Cost is comfortably PTQ-diagnostic scale at OPT-350M.**  
   Total runtime is `51.31s`; CE gradient collection is only `0.18s`, and edit generation plus single-layer validation is `5.79s`. Data streaming/tokenization dominates (`39.12s`). The current result does not suggest a QAT-like cost explosion.

## What this does and does not prove

Supported:

- CE gradients at deployed ternary weights contain actionable information for improving ternary PTQ.
- The top-k layer budget is a real stabilizer, not an arbitrary cosmetic choice.
- At least one WikiText-selected configuration transfers to C4 in this run.

Not yet proven:

- Robust C4 transfer across multiple calibration offsets/seeds.
- Scalability from OPT-350M to larger LLMs.
- End-to-end PPL/zero-shot improvements after integrating into a stronger PTQ baseline rather than direct ternary diagnostic.
- Whether a conservative selection rule can pick between support/signflip/joint without looking at C4.

## Next minimal experiment

`CEGSP-03B`: repeat the same C4 transfer test for the other two CEGSP-02A offsets:

- `O1`: fit/val offset `4096`, C4 token offset `4096`
- `O2`: fit/val offset `8192`, C4 token offset `8192`
- `k ∈ {4, 6}` only
- C4 remains untouched/report-only

Success: top4 or top6 joint improves WikiText untouched and C4 untouched on both additional offsets. If joint fails but support/signflip consistently pass, freeze the safer family before scaling.
