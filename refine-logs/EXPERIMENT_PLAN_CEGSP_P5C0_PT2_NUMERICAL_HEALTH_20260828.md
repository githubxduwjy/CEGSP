# P5-C0: PT² numerical-health and evaluator-parity audit

## Status

Pre-registered before cloud launch on 2026-08-28. This is a baseline audit only. It does not run CEGSP, does not change any CEGSP rule, and does not select a layer or budget from evaluation results.

## Research question

Is the official OPT-350M PT² ATQ state numerically healthy and evaluator-consistent under the released configuration, or is the poor P5-C result confounded by an unhealthy strong baseline?

## Frozen protocol

- Model: `facebook/opt-350m`.
- GPU target: one RTX 4090 (24 GiB).
- Official calibration: Wikitext-2 train loader, `nsamples=128`, `seqlen=2048`, seed `0`.
- Official PT² configuration: `method=atq`, `percdamp=0.01`, `blocksize/group_size=128`, `num_p=1`, `salient_metric=hessian`, GPTQ enabled, official order `k_proj,v_proj,q_proj,out_proj,fc1,fc2`.
- Two baseline states: official ATQ (`ssr=False`) and official ATQ+SSR (`ssr=True`). SSR is the released flag, not a new implementation.
- Evaluation datasets: official Wikitext-2 and C4 loaders at sequence length 2048; additionally the compact evaluator on fixed Wikitext validation/untouched batches and C4 batches.
- No QAT checkpoint, QAT logits, optimizer step, CEGSP patch, budget sweep, sign-rule sweep, layer selection, or post-hoc epsilon.

## Measurements

For every quantized projection/block, record:

- finite/nonfinite status and `max|Q|`, `p50|Q|`, `p99|Q|`;
- inferred ternary support ratios for `-1/0/+1` and illegal-state ratio;
- affine `mu` and `alpha` statistics;
- Q/K/V/O identity and layer/module name;
- block output reconstruction MSE/RMSE/max error on the official GPTQ audit capture;
- official and compact evaluator metrics for the same resulting model state.

## Pre-registered health interpretation

1. A run is numerically complete only if all recorded tensors and both evaluators are finite and the expected 24-layer projection sequence is present.
2. Report, without changing the run, whether the maximum block `p99|Q|` exceeds the median block `p99|Q|` by more than the fixed 10x diagnostic ratio. This is a diagnostic flag, not a tuned acceptance threshold.
3. Report layer-0 Q/K separately and compare them with the global layer/module distribution. The audit does not discard outliers.
4. Evaluator parity means both evaluators are finite and agree on the direction of quantization degradation relative to clean FP16; their absolute NLL/PPL values are not expected to match because their samples and sequence lengths differ.

## Decision gate after completion

- **Healthy PT²:** finite complete audit, no material diagnostic outlier, and official/compact direction agreement. Then preregister one frozen replay of `PT² -> affine CEGSP` before any launch; do not redesign CEGSP.
- **Unhealthy PT²:** severe or localized numerical outlier remains under the official protocol. Stop treating this PT² state as the main strong-baseline comparison; document the reproduction limitation and identify a stable, state-exportable strong ternary PTQ separately.
- **Only a modified PT² setup is healthy:** freeze that exact setup as a new baseline and require a new preregistration before any CEGSP compatibility test.

This audit itself does not select or launch a follow-up CEGSP experiment.
