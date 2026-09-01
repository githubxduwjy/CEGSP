# CEGSP P8-A: Downstream Screen Analysis

Date: 2026-09-01
Remote: `root@xj-member.bitahub.com:42168`
GPU: NVIDIA A100-SXM4-80GB

## Purpose

P8-A tests whether the frozen affine TernRefine/CEGSP rule that improved
Wikitext-2/C4 NLL in P7-R also transfers to downstream multiple-choice scoring.
No downstream example is used for layer selection, candidate selection, or
hyperparameter tuning.

## Protocol

- Models: Llama-2-7B and Qwen3-8B.
- Compared states: BF16, ordinary affine ternary, affine + CEGSP/TernRefine.
- Tasks: HellaSwag, PIQA, ARC-Easy, ARC-Challenge, WinoGrande, MMLU.
- Examples: first 128 validation/test-screen examples per task.
- Primary screen metric: mean gold normalized answer log-likelihood.
- Secondary diagnostics: accuracy and mean answer margin.
- Frozen rule: all-layer Q/K search, affine ternary `g=128`, threshold factor
  `0.75`, top-6 layers, 64 relocations per selected layer.

This is a bounded downstream screen, not an official lm-eval-harness result.
The remote environment did not have `lm_eval` installed.

## Environment and Preparation Notes

- Initial parallel launch exposed a broken `/opt/conda` pandas installation:
  `pandas.util.version` was missing. This was a harness/environment failure and
  was fixed by reinstalling `pandas==2.2.2` only; PyTorch, Transformers, and
  datasets were not changed.
- Llama used `/opt/conda/bin/python` with Transformers 4.46.3.
- Qwen3 required `/root/CEGSP-code/.venv-qwen3/bin/python` because the base
  Transformers version did not recognize `model_type=qwen3`. The Qwen venv uses
  Transformers 5.16.1 while reusing the same CUDA/PyTorch stack.
- Qwen3-8B path:
  `/model/bitahub-model/pice35408784b54431987c4d13c457b9cd/Qwen3-8B`.
- All six downstream datasets were available from cache after the earlier
  preflight.
- Parallel feasibility: both jobs were launched together. Llama completed while
  Qwen was still loading/evaluating; no OOM occurred. Peak recorded per-process
  memory was 14.645 GiB for Llama and 22.094 GiB for Qwen.

## Integrity

Both result files reported `status=complete`. Both runs preserved legal affine
ternary states:

| Model | Selected Layers | Relocations | Changed Coordinates | Illegal States | Max Codebook Residual | Active Support Preserved |
|---|---:|---:|---:|---:|---:|---:|
| Llama-2-7B | `[1,0,30,31,29,25]` | 384 | 768 | 0 | 0.0 | yes |
| Qwen3-8B | `[7,13,11,12,16,8]` | 384 | 768 | 0 | 0.0 | yes |

## Results

Deltas are `affine+CEGSP - affine baseline`. Higher is better for all three
downstream screen metrics below.

### Llama-2-7B

| Task | BF16 Acc | Affine Acc | CEGSP Acc | Delta Acc | Delta Gold NLL-Score | Delta Margin |
|---|---:|---:|---:|---:|---:|---:|
| HellaSwag | 0.570313 | 0.210938 | 0.218750 | +0.007813 | -0.015178 | -0.054730 |
| PIQA | 0.804688 | 0.562500 | 0.554688 | -0.007813 | +0.016802 | -0.007969 |
| ARC-Easy | 0.601563 | 0.273438 | 0.250000 | -0.023438 | -0.012465 | -0.048371 |
| ARC-Challenge | 0.429688 | 0.218750 | 0.257813 | +0.039063 | -0.001336 | -0.025302 |
| WinoGrande | 0.671875 | 0.531250 | 0.437500 | -0.093750 | +0.124145 | -0.040889 |
| MMLU | 0.257813 | 0.304688 | 0.242188 | -0.062500 | -0.040648 | -0.048248 |

Macro deltas:

- accuracy: -0.023438
- gold normalized log-likelihood score: +0.011887
- margin: -0.037585
- tasks improved on gold score: 2/6
- tasks improved on accuracy: 2/6

### Qwen3-8B

| Task | BF16 Acc | Affine Acc | CEGSP Acc | Delta Acc | Delta Gold NLL-Score | Delta Margin |
|---|---:|---:|---:|---:|---:|---:|
| HellaSwag | 0.531250 | 0.320313 | 0.343750 | +0.023438 | +0.015550 | +0.024914 |
| PIQA | 0.835938 | 0.648438 | 0.648438 | +0.000000 | +0.117989 | +0.030082 |
| ARC-Easy | 0.718750 | 0.328125 | 0.328125 | +0.000000 | +0.340820 | +0.156884 |
| ARC-Challenge | 0.523438 | 0.250000 | 0.320313 | +0.070313 | -0.034085 | -0.046622 |
| WinoGrande | 0.796875 | 0.546875 | 0.515625 | -0.031250 | -0.063254 | -0.002867 |
| MMLU | 0.414063 | 0.226563 | 0.218750 | -0.007813 | +0.216255 | +0.094719 |

Macro deltas:

- accuracy: +0.009115
- gold normalized log-likelihood score: +0.098879
- margin: +0.042852
- tasks improved on gold score: 4/6
- tasks improved on accuracy: 2/6

## Gate

P8-A is a mixed downstream screen.

- Qwen3-8B passes the screen gate: macro gold score improves, macro margin
  improves, macro accuracy slightly improves, and 4/6 tasks improve on the
  primary continuous score.
- Llama-2-7B is borderline: macro gold score is slightly positive, but only 2/6
  tasks improve on the primary score, while macro accuracy and margin decline.

Overall, P8-A supports a cautious statement that downstream continuous-choice
signals can improve under frozen affine CEGSP, especially on Qwen3-8B. It does
not support a broad downstream accuracy claim.

## Interpretation

The result is consistent with the earlier warning from P7-R: NLL improvement
does not automatically become robust zero-shot accuracy recovery. Qwen shows a
cleaner downstream-likelihood transfer; Llama shows task-dependent score
movement and worse discrete accuracy/margins. This should be reported as a
bounded capability/boundary result rather than tuned away.

## Artifacts

Remote artifacts:

- `/root/tqgsp-runs/cegsp_p8a_downstream_llama2_7b_a100_20260901_42168/p8_downstream_result.json`
- `/root/tqgsp-runs/cegsp_p8a_downstream_llama2_7b_a100_20260901_42168/screen.log`
- `/root/tqgsp-runs/cegsp_p8a_downstream_qwen3_8b_a100_20260901_42168/p8_downstream_result.json`
- `/root/tqgsp-runs/cegsp_p8a_downstream_qwen3_8b_a100_20260901_42168/screen.log`

Local pull was blocked during this turn by transient DNS resolution failure for
`xj-member.bitahub.com`; remote paths above were verified directly over the
already-open SSH session.
