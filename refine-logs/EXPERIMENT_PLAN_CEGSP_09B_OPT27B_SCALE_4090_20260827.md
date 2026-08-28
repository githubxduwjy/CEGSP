# CEGSP-09B plan: OPT-2.7B scale validation on RTX 4090

日期：2026-08-27

## Motivation

The user correctly pointed out that the experiment loop had become too small and risked drilling into local details. CEGSP-09A already passed on OPT-1.3B. The next meaningful test is therefore **model scale**, not another small ablation.

## Hypothesis

If CEGSP is a general PTQ-side ternary edit mechanism rather than a small-model artifact, then the same clean-room CE-gradient editing rule should improve direct ternary PTQ on OPT-2.7B with acceptable RTX 4090 cost.

## Run

- Run ID: `CEGSP-09B-OPT27B-O0-U32-SCALE`
- Model: `facebook/opt-2.7b`
- Layers: 0--31
- Quantization: direct ternary PTQ, group size 128, threshold factor 0.7
- Edited modules: Q/K projection only
- fit batches: 8
- validation batches: 8
- untouched Wikitext batches: 24
- untouched C4 batches: 24
- k sweep: 12,16
- support-topk: 8
- signflip-topk: 8
- max edits per layer/module candidate: 64
- random controls: disabled for this scale run
- cloze eval: disabled for this scale run

## Gate

Primary gate:

- At least one CE joint top-k patch set improves both untouched Wikitext and untouched C4 NLL over direct ternary PTQ.

Secondary observations:

- Support relocation should remain competitive with or stronger than nonzero-only signflip.
- Runtime and memory should remain within a practical 4090 PTQ-style budget.

## Fallback

If OPT-2.7B fails due to memory or model-loading issues:

1. Reduce untouched Wikitext/C4 batches from 24 to 16.
2. If still failing, reduce validation batches from 8 to 4.
3. Do not change the algorithm, do not introduce QAT artifacts, and do not pivot the method.
