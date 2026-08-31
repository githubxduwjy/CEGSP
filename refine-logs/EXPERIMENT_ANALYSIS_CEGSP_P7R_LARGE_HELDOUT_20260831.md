# CEGSP P7-R: large-model held-out robustness analysis

Date: 2026-08-31  
Remote: `root@xj-member.bitahub.com:42067`  
GPU: NVIDIA A100-SXM4-80GB

## Purpose

P7-R was the pre-registered follow-up to P7-S0/P7-A/P7-B. P7-C was skipped as
requested. The experiment kept the affine CEGSP rule completely fixed and
tested whether the earlier Wikitext-only large-model signal transferred to
larger untouched Wikitext-2 and a bounded streamed C4 slice.

## Protocol audit

Both runs completed with `status=complete`. The configuration matched the
pre-registration: sequence length 128, batch size 1, fit 4, validation 32,
untouched Wikitext 32, streamed C4 16, one gradient batch, group size 128,
threshold factor 0.75, `layer_budgets=6`, 64 edits per selected layer, seed
20260831, BF16. All decoder layers were covered during ranking: 32 Llama layers
and 36 Qwen layers. Each layer retained 256 candidates (128 candidates per
Q/K module). The selected patch had 6 layers, 384 relocation pairs, and 768
changed coordinates.

The C4 source was `allenai/c4:en:validation:text:8000`, not a deterministic
fallback. All displayed result numerics were finite. The baseline and patched
states had zero illegal ternary states, zero codebook residual, and identical
active-support cardinality within each model.

## Results

NLL deltas are `CEGSP - affine baseline`; lower is better.

| Model | State | W2 NLL | C4 NLL | Δ W2 | Δ C4 |
|---|---|---:|---:|---:|---:|
| Llama-2-7B | affine baseline | 7.620776 | 7.566776 | — | — |
| Llama-2-7B | affine + CEGSP top-6 | 7.550108 | 7.401753 | -0.070668 | -0.165023 |
| Llama-2-7B | matched random | 7.621426 | 7.563252 | +0.000650 | -0.003523 |
| Qwen3-8B | affine baseline | 4.584134 | 4.661928 | — | — |
| Qwen3-8B | affine + CEGSP top-6 | 4.516931 | 4.565536 | -0.067203 | -0.096392 |
| Qwen3-8B | matched random | 4.584600 | 4.658326 | +0.000465 | -0.003602 |

## Gate and interpretation

The pre-registered strong cross-domain scaling gate passes for both models:
CEGSP improves W2 and C4 relative to the affine baseline, and is better than
the matched random relocation on both holdouts. This is stronger than the
previous P7 Wikitext-only result and supports the bounded claim:

> Under a frozen affine ternary rule, quantized-point CE-gradient ranking can
> produce legal, non-random, cross-domain held-out NLL improvement on the
> tested Llama/Qwen 7B–8B models.

The result does not support claims of SOTA ternary PTQ, strong-PTQ/PT²
compatibility, QAT-gap closure, downstream accuracy improvement, or multi-seed
robustness. Absolute affine baselines remain weak, so the main evidence is
the frozen-rule transfer and random-control separation, not recovered FP16
quality.

The external result-to-claim reviewer was unavailable after the required
fallback attempt. Accordingly, the formal external verdict remains
`REVIEW_UNAVAILABLE`; the gate above is a deterministic local analysis of
existing result files.

## Raw artifacts

- [Llama P7-R result](../results/remote-runs/cegsp_p7r_llama2_7b_heldout_a100_20260831_42067/p7_affine_scaling_result.json)
- [Llama P7-R log](../results/remote-runs/cegsp_p7r_llama2_7b_heldout_a100_20260831_42067/screen.log)
- [Qwen P7-R result](../results/remote-runs/cegsp_p7r_qwen3_8b_heldout_a100_20260831_42067/p7_affine_scaling_result.json)
- [Qwen P7-R log](../results/remote-runs/cegsp_p7r_qwen3_8b_heldout_a100_20260831_42067/screen.log)

## Next status

P8-A downstream screening is prepared locally and committed to GitHub, but it
was not started because the requested 42067 endpoint began refusing SSH
connections after P7-R completed. No alternate endpoint was assumed and no
P8 result is claimed. Resume with the same fixed P8-A plan once an active A100
endpoint is available.
