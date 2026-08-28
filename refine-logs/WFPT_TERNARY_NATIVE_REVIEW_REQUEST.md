# Adversarial review request: is WFPT genuinely ternary-native?

## Decision requested

Act as a hostile but technically precise reviewer. Decide whether the current proposal, "cross-layer function preservation + per-layer trust region" (WFPT), is publishable as a new 1.58-bit weight-only PTQ method. Do not try to rescue it by rhetoric. Answer three questions:

1. Does it exploit properties specific to ternary weights {-1,0,+1}, or is it a generic low-bit reconstruction method?
2. Does using a full-precision teacher and exact function-level evaluation make calibration cost approach optimization-heavy PTQ, ATQ, or QAT?
3. Given existing work already uses activation/output reconstruction, sliding-layer or blockwise joint optimization, distillation, Hessian/Fisher information, and learned ternarization, is the novelty sufficient?

Then recommend the smallest defensible pivot, if one exists.

## Current proposal

The current WFPT proposal quantizes weights as Wq = mu + alpha*T, T in {-1,0,+1}. It proposes a full-model teacher objective (token KL/NLL, tail CVaR and domain minimax), cross-layer/window function preservation, per-layer reconstruction trust-region constraints, a Hamming update budget, Hessian/Fisher candidate shortlisting, and exact full quantized-context evaluation before accepting discrete T updates.

The intended claim is that the global objective prevents harmful error propagation while the local trust region preserves layerwise fidelity.

## Existing evidence from our experiments

- R042c showed that hard ternary-code updates can improve local block metrics: median +5.83%, mean +12.14%, 96.43% wins.
- R043/R044 showed unstable model-level transfer: Wikitext-2 degraded or became nonfinite while C4 improved.
- R045 showed isolated FP16-rest diagnostics could not predict full quantized-context harm.
- R046 confirmed a Wikitext-2/C4 split under the full quantized context.
- R047-R058 tried increasingly elaborate gates. They were unstable. R058's fixed hard_l11 candidate passed an untouched local test but failed the Wikitext-2 gate.
- R054 falsified the proposed cancellation-risk/no-cancellation mechanism, so that cannot be treated as established motivation.

## Relevant prior art already identified

- PT²-LLM: asymmetric ternary grid, iterative ternary fitting/flexible rounding, activation-aware grid alignment, salience-based reordering.
- CAT-Q: learnable modulation, softened-to-hard ternarization, and sliding-layer output reconstruction that jointly optimizes ternary weights/scales; heavy calibration (reported default 60 epochs over 512 sequences).
- TernaryLLM: learnable ternarization plus feature/logit knowledge distillation; training-heavy.
- AQLM and related blockwise methods: joint optimization across transformer blocks exists for generic low-bit quantization.
- KronQ and second-order methods: input/output-side sensitivity, gradient covariance and limited backward passes already go beyond per-layer weight Frobenius error.

## Candidate ternary-native pivot to assess

Factorize each ternary code as T = S * M, where S in {-1,+1} is polarity and M in {0,1} is zero/nonzero support. Distinguish support birth/death (0 <-> +/-1) from polarity reversal (+1 <-> -1), rather than treating both as one Hamming flip. Optimize paired zero-support exchanges across adjacent layers/windows under a fixed active-code budget, refitting shared group scale/offset exactly after each exchange. Initially freeze polarity and optimize only support; allow a tiny separately budgeted set of sign reversals only if justified.

Use cached FP window-boundary targets once, not full teacher logits in every inner-loop step. Keep cross-layer reconstruction and local trust constraints as supporting safeguards, not as the claimed innovation.

Assess whether this pivot is genuinely tied to the three-state code, whether it merely becomes pruning, and what decisive experiments and cost limits are necessary.

## Required output

- Verdict on each of the three questions, with reject/weak accept style confidence.
- Exact overlap with prior art and what is not novel.
- Whether the support/polarity pivot is defensible and its nearest conceptual competitors.
- A compact mathematical core for a revised method.
- PTQ cost accounting and an explicit go/no-go cost ceiling.
- Three minimum experiments that could falsify the revised claim.
