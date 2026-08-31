# P9-S0: Official PT2 Llama-2-7B health and state-parity audit

## Status

Pre-registered on 2026-08-31 before cloud launch on the A100 endpoint.

## Research question

Can the official PT2 Llama-2-7B pipeline produce a numerically healthy ternary
state under the released ATQ+SSR protocol, before any CEGSP post-processing is
attempted?

This experiment is a baseline audit only. It does not run CEGSP, does not tune
CEGSP budget/sign/layers, and does not use downstream or held-out metrics to
modify the quantizer.

## Frozen protocol

- Model: local Hugging Face Llama-2-7B directory `/CEGSP/model`.
- Hardware target: one A100 80GB.
- PT2 code: official PT2-LLM repository, with commit recorded in the run log.
- Quantization method: `atq`.
- Main strong setting: `--ssr`, `blocksize=128`, `nsamples=128`,
  `calib_seqlen=2048`, `ppl_seqlen=2048`, `percdamp=0.01`, `num_p=1`,
  `salient_metric=hessian`, GPTQ enabled.
- Calibration data: official PT2 Wikitext-2 calibration loader.
- Evaluation: official PT2 WikiText-2 and C4 PPL at sequence length 2048.
- Checkpoint saving: save fake-quantized PT2 checkpoint for later state audit
  and possible frozen compatibility test.

## Measurements

- Official W2/C4 PPL for the resulting PT2 checkpoint.
- Finite/nonfinite status from the log and saved state loading.
- Checkpoint existence and reloadability.
- PT2 commit, package versions, GPU type, peak GPU memory if available.
- Whether the result appears materially healthier than the previous OPT-350M
  PT2 audit failure.

## Decision gate

- `PT2_LLAMA7B_HEALTH_PASS`: official W2/C4 evaluation is finite, checkpoint is
  saved and reloadable, and no severe numerical pathology is visible from the
  run log or state inspection. Only after this gate may a separate frozen
  `PT2 -> PT2+CEGSP` compatibility experiment be planned.
- `PT2_LLAMA7B_HEALTH_FAIL`: official PT2 on the author's main model setting is
  not numerically healthy or cannot complete under the official protocol. Stop
  using PT2 as the main strong-baseline compatibility target and move to a
  state-exportable strong ternary PTQ candidate such as TWLA.

