# AutoResearch Summary: PTQ 1.58-bit after PT2-LLM

Date: 2026-08-21

## Executive Decision

Stop the local Haar-pairing mainline. Continue with activation-sorted structured
Hadamard rotation as the new mainline.

The evidence now includes two negative mechanism gates:

1. R019: weight-cosine similarity pairing reduced Haar high-band energy, but only
   improved activation-weighted NMSE by 0.35% vs random under the band grid.
2. R033b: activation-weighted pairing improved the objective more than weight-cosine
   pairing, but still reached only 0.7337% median improvement and 67.86% block win
   rate vs random. The preregistered gate was 5% and 70%.

This means the failure is not just "wrong pairing metric". The deeper issue is
that fixed local 2-column Haar is too weak a coordinate transform for 1.58-bit PTQ.

R034/R035 then tested the planned structured-rotation pivot. The signal is strong:

- `activation_rms_hadamard`: 9.7247% median activation-weighted NMSE improvement
  vs identity, 100% block win rate.
- `joint_norm_hadamard`: 9.1821% median improvement, 100% block win rate.
- `activation_rms_permutation`: approximately 0% improvement, so sorting alone is
  not the source.
- `random_perm_hadamard`: 0.7181% median improvement.
- `reverse_activation_rms_hadamard`: 0.0705% median improvement.

Decision: the new hypothesis is not "Haar pairing helps"; it is "activation-ordered
structured mixing produces ternary-friendly coordinates under activation-aware PTQ."

R036a/R036b tested whether the block-level signal transfers to PPL:

- R036a quantized only layers 0/10/20/31. Activation-sorted Hadamard improved
  Wikitext2/C4 PPL from 6.6303/8.8797 to 6.2003/8.3578, a relative improvement
  of 6.49%/5.88%.
- R036b quantized all 224 projections with the same naive block ATQ path.
  Identity ATQ collapsed on Wikitext2 (`NaN`) and gave C4 195.6563.
  Activation-sorted Hadamard stayed finite on Wikitext2 and gave 158.9003/191.6737.

Decision: the method has real transfer signal, but the all-layer naive ATQ path is
not the right final vehicle. The next experiment must integrate the rotation into
PT2's stronger GPTQ/SSR-style quantization path or a layerwise staged fake-quant
path with error propagation control.

R037 then embedded activation-rms sorted Hadamard directly into the official
sequential GPTQ path without SSR, using the same LLaMA-2-7B nsamples=8 setup:

- official GPTQ no-SSR: Wikitext2/C4 PPL 25.8104/66.7370.
- activation-rms sorted Hadamard GPTQ: Wikitext2/C4 PPL 26.4786/53.2413.
- Relative change: Wikitext2 regressed by 2.59%, while C4 improved by 20.22%.

Decision: the GPTQ integration is viable but not yet cleanly dominant. The result
is strong enough to keep the structured-rotation line alive, but the next run
should test SSR-compatible placement and a staged layer-count sweep rather than
claiming a final method.

R038 tested the most direct SSR-compatible placement: keep official SSR block
selection and insert activation-rms sorted Hadamard inside each selected SSR block.

- official SSR: Wikitext2/C4 PPL 20.3080/49.4199.
- activation-rms sorted Hadamard inside SSR blocks: Wikitext2/C4 PPL 24.4195/62.4480.
- Relative regression: 20.25% on Wikitext2 and 26.36% on C4.

Decision: do not stack Hadamard inside SSR blocks. The current evidence says SSR's
similarity-based grouping and the activation-Hadamard mixing are competing
coordinate choices, not complementary transforms at the same block location.

R039b tested a selective replacement strategy: use activation-rms sorted Hadamard
only on layers 0/10/20/31, the same layers that gave the positive reduced PPL
screen in R036a, while leaving the rest of the model on official GPTQ.

- selective activation-Hadamard GPTQ: Wikitext2/C4 PPL 25.0170/58.1993.
- Compared with official GPTQ no-SSR from R037, this improves Wikitext2 by 3.07%
  and C4 by 12.79%.
- Compared with full activation-Hadamard GPTQ from R037, it fixes the Wikitext2
  regression but gives up some C4 gain.
- It still does not beat official SSR, which remains 20.3080/49.4199 at nsamples=8.

Decision: the useful hypothesis is now selective replacement. The rotation should
not be applied everywhere, and it should not be stacked inside SSR; it should be
searched as a layer/projection-local alternative to official GPTQ/SSR choices.

R040a/R040b decomposed the R039b mask:

- layer 0 only: Wikitext2/C4 PPL 24.2900/55.4673, improving over official GPTQ
  no-SSR by 5.89%/16.89%.
- layers 10/20/31 only: Wikitext2/C4 PPL 27.3516/68.2298, regressing by
  5.97%/2.24%.

Decision: layer 0 is the dominant positive contributor. Layers 10/20/31 are not
neutral; they actively hurt this insertion strategy. The next decomposition should
split layer 0 by projection.

R041a/R041b split layer 0 into attention-only and MLP-only groups:

- layer-0 attention only (q/k/v/o): Wikitext2/C4 PPL 26.0009/56.3635. This
  improves C4 by 15.54% vs official GPTQ no-SSR but regresses Wikitext2 by 0.74%.
- layer-0 MLP only (up/gate/down): Wikitext2/C4 PPL 28.1206/66.1013. This
  regresses Wikitext2 by 8.95% and barely improves C4 by 0.95%.
- layer-0 all projections remains much better: 24.2900/55.4673.

Decision: the layer-0 gain is not explained by attention-only or MLP-only alone.
There is likely a cross-submodule interaction through GPTQ error propagation or
layer output coupling. The next split should test layer-0 individual projections
or pairs, with special attention to whether MLP projections are harmful alone but
helpful when attention projections are also rotated.

## What Worked

- PT2-LLM baseline reproduction is credible at the 64-sample LLaMA-2-7B scale:
  Wikitext2/C4 PPL 11.8878/26.0226 for ATQ+SSR.
- SSR itself is useful in the reproduced setting: vs ATQ without SSR, it improved
  Wikitext2/C4 PPL by 2.4724/5.6038.
- Negative controls behaved correctly: dissimilar pairing worsened high-frequency
  energy and activation-weighted error.
- Activation-aware pairing was directionally better than pure weight-cosine pairing,
  so activation-weighted objectives remain the right evaluation target.
- Activation-sorted Hadamard produced a clean positive mechanism signal that survived
  immediate controls.
- R036a showed reduced PPL transfer on both Wikitext2 and C4.
- R036b showed all-layer stability improvement even when naive identity ATQ collapsed.
- R037 showed the rotation can be inserted into the official GPTQ path and improve
  C4 substantially in an all-layer run.
- R038 confirmed that official SSR remains a strong low-sample baseline.
- R039b showed selective activation-Hadamard can beat official no-SSR GPTQ on both
  Wikitext2 and C4.
- R040a improved further with layer 0 only, giving the best no-SSR activation-
  Hadamard result so far.
- R041a preserved most of the C4 gain with attention-only layer 0.

## What Failed

- High-frequency energy is not a reliable success proxy.
- Local 2-column Haar pairing does not produce enough activation-weighted error
  reduction to justify extra metadata, transforms, or kernel complexity.
- Even after adding activation information to pairing, the median gain was below
  1%, far from the 5% mechanism gate.
- Therefore, full-model Haar PPL, bit-accounting, and kernel work are not justified
  for this formulation.
- Dense random rotations and random-permutation Hadamard were too weak to explain
  the new result, so the ordering criterion appears important.
- Naive all-layer ATQ is not sufficient for final model quality; the positive
  rotation must be embedded into the stronger PT2/GPTQ machinery.
- The first GPTQ insertion is dataset-split: C4 improves but Wikitext2 regresses,
  so rotation placement/objective still needs refinement.
- Hadamard inside SSR blocks regressed on both datasets; this insertion point is
  closed unless a new objective or transform changes the mechanism.
- Selective activation-Hadamard still trails official SSR, so it is not yet a
  PT2-level replacement.
- Later selected layers 10/20/31 are harmful under this strategy.
- Layer-0 MLP-only is harmful on Wikitext2, so projection interactions matter.

## Literature Position

PT2-LLM is still the closest baseline. Its core components are ATQ, ITF, AGA, and
SSR, and the arXiv page lists it as accepted at ICLR 2026:
https://arxiv.org/abs/2510.03267

TWLA is the most relevant next technical direction. It targets W1.58A4 PTQ using
layer-output-error-oriented ternarization, Kronecker-structured orthogonal rotation
for tri-modal weight shaping and activation outlier suppression, plus mixed-precision
activation allocation:
https://arxiv.org/abs/2606.13054

ScaleQ-1.58 / AYOT is the strongest calibration-data pivot. It argues that reasoning
LLM ternarization can collapse when calibration ignores reasoning traces, and uses
the target model's own reasoning traces and answers as calibration context:
https://arxiv.org/abs/2608.01078

BitNet b1.58 remains the north-star for native ternary training, not a PTQ baseline:
https://arxiv.org/abs/2402.17764
https://arxiv.org/abs/2504.12285

Undermind MCP status: `codex mcp list` shows `undermind` enabled with OAuth, but
this Codex session exposes no callable Undermind research tool and the CLI has only
management commands (`list/get/add/remove/login/logout`). The literature notes above
therefore come from direct web/arXiv lookup.

## Recommended Next AutoResearch Track

### Track A: Structured Orthogonal Ternary Shaping

Active hypothesis: activation-sorted structured mixing reshapes PT2 blocks into a
ternary-friendly coordinate system while reducing activation outlier sensitivity.

Mechanism screen:

- Candidate rotations: activation-rms sorted Hadamard, joint-norm sorted Hadamard,
  random-permutation Hadamard, reverse-order Hadamard, and identity/PT2 SSR controls.
- Objective: minimize activation-weighted output error after ternary quantization,
  not weight MSE or spectral energy.
- Scope: same 56-block LLaMA-2-7B screen first.
- Gate: at least 5% median activation-weighted NMSE improvement vs PT2 SSR/random
  rotation and at least 70% block win rate.

This gate has passed at the block mechanism level.

Completed validation:

- Reduced 4-layer PPL screen passed.
- All-layer naive stress test showed finite/stability improvement but poor absolute PPL.

Next validation:

- Run layer-0 single-projection/pair attribution before scaling it.
- Compare the best selective mask as an SSR replacement, not as an inner SSR addition.
- Run a staged layer-count sweep: 4 layers, 8 layers, 16 layers, all layers.
- Then do bit accounting and kernel feasibility.

If the PPL screen fails, keep the block-level result as a mechanism observation but
do not claim model-level quality.

### Track B: Reasoning-Trace Calibration for PT2/CAT

Hypothesis: for reasoning models, PTQ 1.58-bit quality is bottlenecked less by the
ternary codebook and more by calibration context mismatch.

Mechanism screen:

- Compare WikiText2/C4 calibration vs AYOT-style generated reasoning traces.
- Target models: Qwen3 family first, because the prior Qwen smoke results showed
  severe ternary degradation.
- Metrics: layer output error, PPL, and math/code task accuracy if feasible.
- Gate: consistent layer-output-error reduction plus end-task rescue relative to
  generic calibration.

## Concrete Next Run

Run ID: R041

`R042` should decompose layer 0 at finer granularity. R041 showed attention-only
helps C4 but not Wikitext2, MLP-only is poor, and all projections together are
best. The immediate question is which projection or pair creates the Wikitext2
rescue.

Minimum comparison set:

- official PT2 ATQ/AGA no-SSR baseline
- official PT2 ATQ/AGA + SSR baseline
- activation-rms sorted Hadamard no-SSR
- selective masks inside layer 0: individual q/k/v/o/up/gate/down projections and
  a few pairs such as attention+down or q/k/v/o+down
- random-permutation Hadamard control at the same selective masks

Primary metric:

- Wikitext2/C4 PPL, plus layerwise activation-weighted NMSE and NaN/fallback counts.

Gate:

- PPL improves over the matched PT2 baseline on both Wikitext2 and C4, or at least
  does not regress while improving layerwise output error.
- Storage accounting includes permutation metadata and transform cost.

## Current Artifacts

- R019 analysis: `results/remote-runs/haar_m2_band_r019_20260821_191000/ANALYSIS.md`
- R033b metrics: `results/remote-runs/activation_pairing_r033b_20260821/metrics.json`
- R033b pivot summary: `results/remote-runs/activation_pairing_r033b_20260821/pivot_summary.json`
- R034 metrics: `results/remote-runs/rotation_shaping_r034_20260821/metrics.json`
- R035 sanity metrics: `results/remote-runs/rotation_shaping_r035_20260821/metrics.json`
- R036a reduced PPL: `results/remote-runs/rotation_ppl_r036a_20260821/`
- R036b all-layer stress: `results/remote-runs/rotation_ppl_r036b_all_20260821/`
- R037 official GPTQ integration: `results/remote-runs/rotation_gptq_r037_ns8_20260821/`
- R038 SSR insertion screen: `results/remote-runs/rotation_ssr_r038_ns8_20260821/`
- R039b selective replacement: `results/remote-runs/rotation_selective_r039b_ns8_20260821/`
- R040a layer-0 attribution: `results/remote-runs/rotation_selective_r040a_l0_ns8_20260821/`
- R040b later-layer attribution: `results/remote-runs/rotation_selective_r040b_l10_20_31_ns8_20260821/`
- R041a layer-0 attention split: `results/remote-runs/rotation_selective_r041a_l0_attn_ns8_20260821/`
- R041b layer-0 MLP split: `results/remote-runs/rotation_selective_r041b_l0_mlp_ns8_20260821/`
- AutoResearch entry: `autoresearch/program.ptq158.md`
