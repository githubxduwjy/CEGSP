# Experiment Plan: TQGSP-02A CE-Aware Layer Selection

日期：2026-08-26

## Purpose

`TQGSP-01B` showed that budget-matched ternary support projection has a real Q/K operator signal: `TQGSP-support-G` beat random/forward/signflip controls on 3/4 tested layers. However, end-to-end NLL transfer was weak/mixed.

`TQGSP-02A` tests the next question:

> Can operator-level TQG-SP gains predict which layers actually improve language-model CE/NLL?

If yes, the next method can use cheap operator proxy plus a small CE validation gate. If no, we should stop expanding operator-only TQG-SP and move to direct CE-gradient support projection.

## Clean-Room Boundary

- No QAT checkpoint.
- No QAT logits.
- No QAT latent weights.
- No QAT state prior.
- No TDBT path/barrier.
- No post-hoc epsilon tuning after seeing results.

## Setup

- Run ID: `TQGSP-02A`
- Model: `facebook/opt-350m`
- GPU: RTX 4090 24GB
- Layers: all OPT-350M decoder layers `0..23`
- Operator: Q/K composed operator only
- Data:
  - Wikitext-2 train fit batches for gradient/operator calibration
  - Wikitext-2 validation first segment for CE layer selection
  - Wikitext-2 validation later segment as untouched final split
- Fit / val / untouched batches: `16 / 8 / 8`
- Sequence length: `128`
- Batch size: `2`
- Group size: `128`
- Threshold factor: `0.7`
- Max support edits per layer: `64`
- Gradient batches: `1`

## Compared Patch Sets

| Patch set | Definition | Role |
|---|---|---|
| `direct-ternary` | Direct PTQ all linear layers | Baseline |
| `all-tqgsp-qk` | Patch Q/K TQG-SP edits on all tested layers | Does scaling proxy blindly help? |
| `operator-topk-qk` | Patch top-k layers by untouched operator gain | Does operator ranking alone work? |
| `ce-selected-qk` | Patch layers with positive operator gain and non-worse val NLL delta | Main CE-aware selection |
| `ce-topk-qk` | Patch top-k layers by per-layer val NLL delta | Diagnostic upper bound using validation CE |

## Metrics

Primary:

- Per-layer operator NMSE improvement on val and untouched.
- Per-layer single-patch val NLL delta versus direct PTQ.
- Final untouched NLL for patch sets.

Secondary:

- Spearman/Pearson correlation between operator gain and val NLL delta.
- Wall-clock breakdown and peak GPU memory.

## Gate

Proxy reliability gate:

- Operator gain should negatively correlate with val NLL delta. More operator improvement should generally imply lower NLL.
- If correlation is weak or wrong-signed, pure operator proxy is not reliable.

CE-aware selection gate:

- `ce-selected-qk` should improve or at least not degrade untouched NLL versus direct PTQ.
- If `all-tqgsp-qk` degrades but `ce-selected-qk` improves, CE gating is necessary and useful.

Stop / pivot condition:

- If per-layer operator gain is not predictive and CE-selected also fails, stop operator-only TQG-SP and implement direct CE-gradient support projection next.

