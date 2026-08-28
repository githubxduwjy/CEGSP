# CEGSP-08A Cloze Sanity Plan

日期：2026-08-27

## Purpose

CEGSP-01A through 07B established robust NLL improvements, random/edit controls, and ternary-specific support relocation evidence. The next paper-critical question is whether this NLL improvement has any task-level signal beyond perplexity tables.

CEGSP-08A is a small downstream sanity check, not a full benchmark.

## Claim Tested

> CEGSP's NLL improvement is not purely a metric artifact; it should at least not harm, and ideally improve, simple cloze prediction accuracy compared with direct ternary PTQ.

## Setup

- Models: `facebook/opt-125m`, `facebook/opt-350m`
- Quantization: direct ternary PTQ, group size 128, threshold factor 0.7
- Edited modules: Q/K only
- Calibration: WikiText-2 O0, 8 fit batches, 8 validation batches
- NLL holdouts: WikiText-2 32 batches and C4 32 batches
- Cloze task: LAMBADA-style last-token cloze, 128 examples
- Metrics: cloze last-token NLL, top1 accuracy, top5 accuracy
- CEGSP variants evaluated on cloze: `ksweep-joint*` only
- OPT-125M k: `2,3`
- OPT-350M k: `4,6`

## Gates

Primary sanity gate:

- Best CE joint top-k should improve W32/C4 NLL versus direct ternary.
- Best CE joint top-k should not degrade both cloze top1 and top5 versus direct ternary.

Positive task-signal gate:

- If top1 or top5 improves on either model while NLL improves, claim "early task-level signal".
- If cloze accuracy is flat but NLL improves, claim only "no obvious task harm in a small sanity check".
- If cloze accuracy drops on both models despite NLL improvements, do not use NLL as a proxy for downstream claims without further evaluation.

## Direction Rule

This run may influence evaluation priority. It must not change the locked CEGSP method family.
