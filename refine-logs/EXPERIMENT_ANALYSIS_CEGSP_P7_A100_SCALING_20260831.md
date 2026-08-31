# CEGSP P7: A100 7B/8B Scaling Smoke and Frozen-Canonical Affine Test

Date: 2026-08-31

Code commit: `5271de6`

Remote: `root@xj-member.bitahub.com:42058`

GPU: NVIDIA A100-SXM4-80GB

## Purpose

This run follows the v4.0 route after P6-B: move from small-model mechanism
evidence to A100 scaling. The goal is not to tune CEGSP, but to test whether
the frozen affine whole-model rule can run on Llama/Qwen-size models and still
produce a non-random improvement signal.

Protocol held fixed:

- representation: affine ternary `Q = mu + alpha T`, `T in {-1,0,+1}`;
- scope: all decoder layers, attention `q_proj`/`k_proj` only;
- group size: 128;
- threshold factor: 0.75;
- selection signal: one fit-split quantized-point CE backward;
- budgets: top-4 and top-6 layers, 64 relocation pairs per selected layer;
- controls: matched random relocation on the same selected layer set;
- no QAT teacher, no latent optimizer, no multi-step training, no held-out
  selection.

This initial A100 run uses Wikitext-2 only (`c4=skipped`) to establish scaling
and architecture compatibility. C4/downstream remain future paper-table runs.

## Raw Artifacts

- Llama-2-7B smoke:
  `results/remote-runs/cegsp_p7s0_llama2_7b_a100_20260831_42058/`
- Llama-2-7B affine scaling:
  `results/remote-runs/cegsp_p7a_llama2_7b_affine_a100_20260831_42058/`
- Qwen3-8B smoke:
  `results/remote-runs/cegsp_p7s0_qwen3_8b_a100_20260831_42058_v2/`
- Qwen3-8B affine scaling:
  `results/remote-runs/cegsp_p7b_qwen3_8b_affine_a100_20260831_42058/`

## Environment Notes

The base environment initially matched the repository pin:
`torch==2.5.1+cu124`, `transformers==4.46.3`, `datasets==3.0.1`.

Llama-2-7B runs used the base environment.

Qwen3-8B failed under `transformers==4.46.3` because `model_type=qwen3` was
unrecognized. A separate `.venv-qwen3` with system-site packages was created
and upgraded only for Qwen:

- torch: `2.5.1+cu124`
- transformers: `5.16.1`

This isolates the Qwen architecture fix from the pinned Llama/OPT-compatible
environment.

## P7-S0 Smoke Results

| Model | Layers | Finite | Peak memory | One-batch FP val NLL | Notes |
|---|---:|---:|---:|---:|---|
| Llama-2-7B HF | 32 | true | 14.645 GB | 1.4872 | all Q/K grad norms finite |
| Qwen3-8B | 36 | true | 16.819 GB | 1.8725 | required newer transformers |

Interpretation: A100 80GB can load both models in BF16 and run one
seq_len=128, batch=1 quantized-point Q/K backward without OOM.

## P7-A/B Frozen-Canonical Affine Scaling Results

### Llama-2-7B

Data: Wikitext-2 cached/loaded through HuggingFace datasets; C4 skipped.

| Variant | Selected layers | Changed coords | Val NLL | Untouched W2 NLL | Delta Val | Delta W2 |
|---|---|---:|---:|---:|---:|---:|
| FP/BF16 | - | 0 | 1.882328 | 2.114392 | - | - |
| Affine baseline | all Q/K | 0 | 7.573157 | 7.657350 | - | - |
| Affine CEGSP top-4 | 1,0,30,31 | 512 | 7.467231 | 7.642597 | -0.105926 | -0.014753 |
| Random matched top-4 | 1,0,30,31 | 512 | 7.573446 | 7.660000 | +0.000289 | +0.002650 |
| Affine CEGSP top-6 | 1,0,30,31,29,25 | 768 | 7.467578 | 7.642538 | -0.105579 | -0.014812 |
| Random matched top-6 | 1,0,30,31,29,25 | 768 | 7.571535 | 7.658802 | -0.001622 | +0.001451 |

Audit: illegal states = 0, max codebook residual = 0, active support preserved
at 587,361,192. Runtime 163.73 s. Peak memory 14.645 GB.

Gate: PASS on Wikitext-only scaling. Top-6 improves validation and untouched
W2, and beats matched random on untouched W2.

### Qwen3-8B

Data: Wikitext-2 cached/loaded through HuggingFace datasets; C4 skipped.

| Variant | Selected layers | Changed coords | Val NLL | Untouched W2 NLL | Delta Val | Delta W2 |
|---|---|---:|---:|---:|---:|---:|
| FP/BF16 | - | 0 | 2.306142 | 2.505167 | - | - |
| Affine baseline | all Q/K | 0 | 3.963487 | 4.035729 | - | - |
| Affine CEGSP top-4 | 7,13,11,12 | 512 | 3.899504 | 4.055006 | -0.063983 | +0.019277 |
| Random matched top-4 | 7,13,11,12 | 512 | 3.965027 | 4.033957 | +0.001540 | -0.001771 |
| Affine CEGSP top-6 | 7,13,11,12,16,8 | 768 | 3.759392 | 3.999196 | -0.204095 | -0.036532 |
| Random matched top-6 | 7,13,11,12,16,8 | 768 | 3.962640 | 4.041554 | -0.000847 | +0.005825 |

Audit: illegal states = 0, max codebook residual = 0, active support preserved
at 411,143,095. Runtime 117.11 s. Peak memory 16.819 GB.

Gate: PASS for the primary top-6 rule on Wikitext-only scaling. The secondary
top-4 rule improves validation but hurts untouched W2, so it should remain a
diagnostic budget result rather than a new default.

## Main Interpretation

This is the first successful cross-family large-model scaling signal for the
affine CEGSP formulation. Under the frozen top-6 canonical rule, both
Llama-2-7B and Qwen3-8B reduce untouched Wikitext-2 NLL relative to the affine
ternary baseline, while matched random relocation does not reproduce the gain.

The result supports a stronger but still careful claim:

CEGSP's task-gradient-ranked affine ternary support relocation scales beyond
OPT/Pythia small models to Llama/Qwen-family 7B/8B models in a Wikitext-only
A100 setting.

It does not yet establish:

- C4 transfer at 7B/8B;
- downstream zero-shot recovery;
- compatibility with a healthy strong ternary PTQ baseline;
- final paper-table performance against state-of-the-art ternary PTQ.

## Next Step

Do not tune budget/sign/group size from these results. The next paper-critical
experiment should keep top-6 fixed and run the same Llama/Qwen models with:

- larger Wikitext-2 evaluation slices;
- C4 evaluation enabled through a bounded local/cached C4 subset;
- downstream zero-shot tasks after LM loss remains positive.

Qwen should continue using the isolated `.venv-qwen3` or the repository should
document a second environment profile for Qwen3-family models.
