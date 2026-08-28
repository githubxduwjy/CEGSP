# Experiment Plan: CEGSP-01A

日期：2026-08-26

## Purpose

`TQGSP-02A` showed that Q/K operator gain is not a reliable predictor of language-model NLL, while CE-aware layer selection improves untouched NLL. Therefore the next method should use CE gradients directly at the deployed ternary point.

`CEGSP-01A` tests:

> Can CE gradients at quantized ternary weights directly guide ternary support projection better than the operator-proxy version?

## Clean-Room Boundary

- No QAT checkpoint.
- No QAT logits.
- No QAT latent weights.
- No QAT state prior.
- No optimizer steps.
- No continuous latent-weight training.

## Setup

- Model: `facebook/opt-350m`
- Layers: all decoder layers `0..23`
- Matrices: Q/K only
- Data: Wikitext-2
- Fit / val / untouched batches: `8 / 8 / 8`
- CE gradient batches: `1`
- Max edits per layer: `64`
- Group size: `128`
- Threshold factor: `0.7`

## Compared Variants

| Variant | Description |
|---|---|
| direct ternary | Direct PTQ all linear layers |
| cegsp-support-all-qk | CE-gradient support swaps on all Q/K layers |
| cegsp-support-selected-qk | Support layers whose single-layer val NLL delta is non-positive |
| cegsp-support-topk-qk | Top-k support layers by single-layer val NLL delta |
| ce-signflip-all-qk | Nonzero-only CE-gradient sign flip control on all Q/K layers |
| ce-signflip-selected-qk | Signflip layers whose single-layer val NLL delta is non-positive |
| ce-signflip-topk-qk | Top-k signflip layers by single-layer val NLL delta |

## Gate

Primary:

- `cegsp-support-selected-qk` should improve untouched NLL versus direct PTQ.

Ternary specificity:

- If signflip selected outperforms support selected, the support-only claim is insufficient and the next method should become joint sign/support editing.
- If support selected outperforms signflip selected, ternary zero-support projection becomes the main method.

Cost:

- The run must remain PTQ-like: one CE backward plus discrete projection/eval, no iterative training.

