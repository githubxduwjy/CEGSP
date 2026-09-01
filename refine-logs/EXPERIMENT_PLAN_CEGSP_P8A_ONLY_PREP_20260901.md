# CEGSP P8-A Only Preparation

Date: 2026-09-01

## Scope

This preparation is limited to P8-A downstream evaluation. It does not prepare,
modify, or launch P11, P10, P9, or any new algorithm branch.

## Paper Question

Does the held-out NLL improvement from the frozen affine TernRefine/CEGSP rule
translate into downstream multiple-choice recovery?

## Frozen Method

- models: Llama-2-7B and Qwen3-8B when both checkpoints are present;
- compared states: BF16, ordinary affine ternary, affine + TernRefine;
- quantization rule: affine ternary with group size 128 and threshold factor
  0.75;
- refinement rule: Q/K only, all decoder layers scored, top-6 layers selected,
  64 relocations per selected layer;
- selection signal: Wikitext train fit split quantized-point CE gradient;
- no downstream example is used for layer selection, candidate selection, or
  hyperparameter choice.

## Prepared Executor

P8 can be launched on the remote server with:

```bash
cd /root/CEGSP-code
bash remote-tools/run_p8a_downstream.sh check llama
bash remote-tools/run_p8a_downstream.sh launch llama
```

On the 42079 server, Qwen3-8B is stored at:

```bash
/model/bitahub-model/pice35408784b54431987c4d13c457b9cd/Qwen3-8B
```

Set it explicitly before running the Qwen half:

```bash
export QWEN_MODEL_PATH=/model/bitahub-model/pice35408784b54431987c4d13c457b9cd/Qwen3-8B
bash remote-tools/run_p8a_downstream.sh check qwen
bash remote-tools/run_p8a_downstream.sh launch qwen
```

The launcher defaults to:

- tasks: `hellaswag,piqa,arc_easy,arc_challenge,winogrande,mmlu` via the
  `all6` alias;
- max examples per task: 128;
- downstream scorer: normalized answer log-likelihood from the local P8
  evaluator;
- output root: `/root/tqgsp-runs`;
- logs: `${run_dir}/screen.log`.

## Important Boundary

The remote image currently does not provide `lm_eval`. Therefore this prepared
P8 executor is a bounded downstream screen, not an official lm-eval-harness
main-table result. It is valid for deciding whether downstream signal exists
under the frozen rule. A later paper-table confirmation should use a fixed
lm-eval-harness version and full/fixed validation splits.

## Pre-Launch Gate For Afternoon

Before launching, confirm:

- the A100 is idle or intentionally allocated to P8;
- `/root/Llama-2-7b-hf` resolves to the intended Llama-2-7B checkpoint;
- Qwen3-8B path is known before running the Qwen half. On the 42079 server,
  use `/model/bitahub-model/pice35408784b54431987c4d13c457b9cd/Qwen3-8B`;
- `bash remote-tools/run_p8a_downstream.sh check llama` passes;
- for Qwen, `QWEN_MODEL_PATH=... bash remote-tools/run_p8a_downstream.sh check qwen` passes;
- no task, edit budget, layer budget, threshold, sign rule, or calibration
  split is changed based on early downstream results.

## 42079 Preparation Notes

- A100 check: `NVIDIA A100-SXM4-80GB`, 0 MiB used at preparation time.
- Python package check: `torch`, `transformers`, and `datasets` available;
  `lm_eval` unavailable.
- Llama check: passed for `/root/Llama-2-7b-hf`.
- Qwen check: passed with
  `/model/bitahub-model/pice35408784b54431987c4d13c457b9cd/Qwen3-8B`.
- Dataset preflight: after adding `trust_remote_code=True` for PIQA and
  WinoGrande, all six tasks loaded one sample successfully on 42079:
  HellaSwag, PIQA, ARC-Easy, ARC-Challenge, WinoGrande, and MMLU.

## Success Criterion

For each model, compare `affine_cegsp` against `affine_baseline`.

Primary screen gate:

- macro average of mean gold normalized log-likelihood is non-degraded;
- at least one task improves on mean gold normalized log-likelihood.

Secondary diagnostics:

- accuracy;
- mean answer margin;
- per-task raw rows for later paired analysis.

Failure is a downstream boundary result, not a reason to tune the frozen rule.
