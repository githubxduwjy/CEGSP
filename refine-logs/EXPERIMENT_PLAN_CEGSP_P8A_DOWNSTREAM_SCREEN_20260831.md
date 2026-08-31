# CEGSP P8-A: bounded downstream log-likelihood screen

Date: 2026-08-31

## Question

After P7-R's fixed-rule WikiText-2/C4 transfer, does affine CEGSP retain a
continuous task-level signal on multiple-choice benchmarks, rather than only
improving language-model NLL?

## Frozen protocol

- models: the already validated Llama-2-7B and Qwen3-8B checkpoints;
- states: BF16, affine ternary baseline, affine ternary + frozen CEGSP;
- CEGSP selection uses only the Wikitext train fit split;
- representation, group size 128, threshold 0.75, all decoder Q/K projections,
  top-6 layer ranking, and 64 legal relocations per selected layer are unchanged;
- no downstream example is used for layer selection, candidate selection, or
  hyperparameter choice;
- tasks: PIQA validation and ARC-Easy validation;
- first screen: first 128 examples from each task, in dataset order;
- primary metric: mean gold normalized answer log-likelihood; accuracy and
  answer margin are secondary diagnostics.

## Gate

P8-A is a screening gate, not a final benchmark table. Relative to affine
ternary baseline, CEGSP passes the task-level signal gate if the average primary
metric over the two tasks is non-degraded and at least one task improves. A
failure is recorded as a downstream boundary result; it does not invalidate the
P7-R language-model function-preservation claim.

The final paper still requires larger/full validation splits, additional tasks,
paired uncertainty, and a fair strong-ternary baseline before a broad
downstream claim.

## Cost and stop rule

The screen runs one quantized-point gradient and one candidate ranking per
model. Do not add tasks, tune the patch, or increase the edit budget based on
the first screen. If the evaluator or a dataset is unavailable, record the
environment/data failure and do not substitute synthetic examples.
