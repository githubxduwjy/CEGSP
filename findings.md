# Research Findings

## 2026-08-22 — Validation-gated discrete ternary refinement

### Verdict

Paper-level result-to-claim verdict: `REVIEW_UNAVAILABLE`. The required Codex reviewer (`gpt-5.6-sol`, ultra) timed out after 300 seconds without returning a usable thread or verdict. Reviewer routing forbids downgrade/retry on timeout. The deterministic evidence pre-check verified all seven cited numeric values, but the existing independent experiment-integrity audit is unavailable.

### What was tested

- R042b/R042c tested validation-gated, activation-covariance-aware hard updates of the ternary assignment `T` on 56 LLaMA-2-7B blocks with disjoint fit/validation/untouched-test activations.
- R043a inserted the update into official no-SSR GPTQ on layers 0/10/20/31.
- R044a isolated layer 0 as a single causal model-transfer test.

### Deterministic findings

- R042c local mechanism gate passed: gated-refit improved untouched-test activation-weighted NMSE by 5.8289% median and 12.1401% mean, with a 96.43% block win rate. The matched ungated-fit-refit control achieved 5.0223% median and 87.50% wins.
- R043a model-transfer gate failed: WikiText2 PPL was NaN, while C4 improved from 66.7370 to 56.0225. All 1,112 refined blocks had zero grid fallback rows, so this was not a closed-form NaN artifact.
- R044a model-transfer gate failed: layer-0-only refinement produced W2/C4 26.2403/57.9333 versus official no-SSR 25.8104/66.7370. C4 improved 13.19%, but W2 worsened 1.67%.

### Supported boundary and failed claim

The evidence supports a narrow mechanism statement: validation-gated hard `T` updates can reduce held-out block output error more consistently than fixed-`T` grid refitting on the tested blocks. It does not support the stronger claim that this is already a better model-level ternary PTQ method. Block activation NMSE is not a sufficient acceptance objective for language-model function preservation.

### Constraints for future attempts

- Do not continue layer/projection enumeration, rotation combinations, or threshold tuning around the current block-NMSE gate.
- Do not launch all-layer or SSR integration of the current hard-`T` rule; the pre-registered transfer gates failed.
- Preserve fit/validation separation for discrete selection, but move acceptance to a cross-distribution criterion. R045 showed that FP16-rest held-out scoring still prefers hard-`T`; R046 showed that full quantized-context scoring reproduces the split: W2 windows prefer official while C4 windows prefer hard-`T`.
- Treat strong C4-only gains as evidence of calibration/distribution dependence, not general improvement.

### Next research direction

Develop a hierarchical trust-region ternarization method: block-level Hessian proposals generate sparse candidate changes, while a layer- or short-sequence validation objective accepts a bounded subset only when it is robust across calibration distributions. The next experiment should test a minimax or Pareto acceptance rule over W2-like and C4-like held-out streams before any all-layer hard-`T` search.

## 2026-08-22 — R045/R046 trajectory-gate diagnostics

R045 tested fixed layer-0 `T` versus the R044 hard-`T` layer-0 candidate inside an otherwise FP16 model. The strict pre-registered gate failed: hard-`T` had lower layer-output NMSE, lower cosine drift, lower mean NLL, and lower CVaR10 NLL increase on all four held-out train sequences.

R046 moved the comparison into the matched full quantized context. The strict all-sequence rejection gate still failed, but the directional split matched R044: on WikiText2 test windows, hard-`T` worsened mean NLL by 0.0479 and CVaR10 by 0.1644 versus official; on C4 validation windows, hard-`T` improved mean NLL by 0.1143 and CVaR10 by 0.1328.

Interpretation: the hard-`T` update is not simply bad; it is distribution-selective. The next method should frame discrete `T` acceptance as robust function preservation across calibration distributions, not as single-distribution reconstruction minimization.

## 2026-08-24 — R058 stop decision and direction pivot

R058 completed cleanly with 128/128 finite rows and the preregistered sequence split. Fixed H0/`hard_l11` improved all four untouched-test W2/C4 function metrics, but failed the W2 gate mean/CVaR by +0.018083/+0.056855. The positive checkpoint NMSE therefore cannot be isolated as an overconservative veto, and R057A's pattern did not replicate.

The current hard-T proposal plus finite-sample gate loop is stopped. The supported boundary remains: discrete support refinement has a real local mechanism, but current candidate formation and selection are not stable model-level PTQ. The preferred untested pivot is short-window cross-layer function-preserving support optimization with retained single-layer trust-region constraints; soft-to-hard continuation is a backup solver, and calibration-data alignment remains a separate task-adaptive axis.

Formal result-to-claim verdict remains `REVIEW_UNAVAILABLE`: the required Codex calls for result-to-claim, research-review, and research-refine each timed out after 300 seconds. Deterministic evidence precheck verified all five anchor values. No new experiment was launched.

## 2026-08-28 — P5-C official PT² state parity and affine compatibility

P5-C tested the preregistered compatibility question `PT² -> PT²+affine CEGSP` on OPT-350M with the official PT² ATQ (`9e943e6`, `ssr=False`), all 24 Q/K layers, fixed top-6 layer selection, 384 relocation pairs, and matched random control. A first parity attempt used an overly strict absolute FP32 residual and pre-cast comparison. Module-level diagnostics showed the discrepancy was confined to official layer-0 large values: the `fasterquant` return state already matched the final FP16 tensor after dtype cast. The same protocol was rerun with the corrected precision-aware parity rule.

The corrected run passed state parity: 48/48 Q/K modules, zero illegal/nonfinite T, capture codebook residual `0.0001220703125 < 1e-3`, and final-vs-dtype-cast capture residual `0.0`. Thus state recovery and group/layout mapping are demonstrated; no performance comparison was blocked by the harness.

The performance gate failed. PT² baseline NLL was `9.850781/10.170247/9.642318` on validation/W2/C4. PT²+CEGSP was `9.900243/10.303432/9.645273`, giving deltas `+0.049462/+0.133185/+0.002955`. Matched random was `9.853100/10.165869/9.649151`, deltas `+0.002318/-0.004378/+0.006833`. All variants were finite and legal, but CEGSP did not improve PT² validation or W2, and did not beat random on W2. Record the result as `STATE_PARITY_PASS; PT2_COMPATIBILITY_FAIL`, not as a failure of CEGSP overall.

Supported claim: CEGSP can legally consume a captured affine ternary state from the official PT² pipeline. Unsupported claim: CEGSP is a complementary performance improvement after strong PT². P5-B's positive affine diagnostic cannot be generalized to PT². The official PT² run also showed approximately `1e3` layer-0 Q/K values and extremely high PPL, so numerical baseline health remains a separate paper-level audit item. Do not respond with blind budget/sign/layer scans; first resolve whether that official numerical behavior is intended and whether the strong baseline is paper-comparable.

Formal reviewer result-to-claim call was unavailable in this turn (no response before termination); `.aris/CLAIMS_FROM_RESULTS.md` records `REVIEW_UNAVAILABLE`.

## 2026-08-28 — P5-C0 official PT² numerical-health audit

P5-C0 ran the released OPT-350M PT² ATQ and ATQ+SSR configurations with `nsamples=128`, calibration/evaluation sequence length 2048, group size 128, `percdamp=0.01`, and seed 0 on an RTX 4090. It did not run CEGSP or use QAT artifacts. Both states completed with 144/144 expected linear modules, 1728 blocks, finite recorded values, zero inferred illegal/nonfinite ternary states, and finite official plus compact evaluator metrics.

The numerical-health gate nevertheless failed. Official ATQ produced W2/C4 PPL 13044.43/11384.02 and ATQ+SSR produced 15917.38/13408.74, versus clean FP16 22.0046/22.5898. The compact evaluator agreed in direction: untouched W2 NLL was 9.5009 for ATQ and 9.8585 for ATQ+SSR versus clean 3.8903. Layer-0 Q/K values were the localized outliers: ATQ Q/K max 1166.0/1152.0; ATQ+SSR Q/K max 11696.0/7844.0. The maximum block p99-to-median ratio was 1.97e4/2.20e5, and layer-0 output reconstruction MSE reached 8.58e4/4.15e6 for ATQ/ATQ+SSR.

Interpretation: T legality and affine state recoverability are not sufficient evidence of a healthy strong PTQ state. P5-C remains `STATE_PARITY_PASS / PT2_COMPATIBILITY_FAIL`, but the negative performance result cannot be interpreted as a test on a healthy strong baseline. The current PT² reproduction is moved to appendix/reproduction limitation and no further CEGSP budget/sign/layer scan is authorized. The next strong-baseline effort must first identify a stable, state-exportable ternary PTQ and repeat a protocol/health audit.

The independent external integrity reviewer did not return before termination; deterministic local audit is recorded as `PASS_WITH_SCOPE_WARNING`, with external status `REVIEW_UNAVAILABLE`.

## 2026-08-28 — P6-A full-layer centered/affine score-validity

P6-A expanded the earlier limited-layer score-validity check to all 24 OPT-350M decoder layers and compared centered and affine ternary initialization under one frozen protocol. Each representation evaluated 192 gradient-ranked and 192 matched-random legal same-group support exchanges; all 768 total rows were finite, and candidates were generated before validation/untouched evaluation.

Both representations passed the preregistered mechanism gate. Centered achieved Spearman `rho=0.6982` on validation and `0.7678` on untouched W2; affine achieved `0.7199` and `0.7388`. The top-20% gradient candidates had validation mean Delta NLL `-0.005323` (centered) and `-0.005588` (affine), versus `-0.000044` and `+0.000286` for matched random; both top-20% improvement rates were `1.0`. The signal is therefore supported across the full layer set and across both codebook representations, not just in a selected layer.

This supports the narrow claim that quantized-point CE first-order scores can prioritize task-relevant legal ternary support relocations. It does not yet establish whole-model patch performance, strong-PTQ superiority, QAT-gap closure, or scaling. The next recommended step is preregistered multi-seed/offset robustness, followed by a fixed-budget global patch only if score validity remains stable.

## 2026-08-28 — P6-B seed/offset replication

P6-B repeated the frozen P6-A protocol at three preregistered seed/offset pairs: `(20260829, 0)`, `(20260830, 512)`, and `(20260831, 1024)`. All three runs completed successfully. Each centered/affine representation covered all 24 OPT-350M layers with 192 gradient and 192 matched-random candidates; all 2304 candidate rows and all summary numerics were finite.

The directionality gate passed in all 6 representation-replicate cells. Centered validation rho was `0.6897 +/- 0.0307` and affine validation rho was `0.7369 +/- 0.0206`; mean `Delta_rank` was `-0.005406 +/- 0.000222` and `-0.006118 +/- 0.000230`, respectively. Top-20% validation improvement rates were 1.000, 1.000, 1.000 for centered and 1.000, 1.000, 0.947 for affine. The affine random untouched-W2 improvement rate varied from `0.104` to `0.875`, so the P6-A split bias does not repeat as a universal effect.

This establishes `STABLE_SUPPORT_SCORE_VALIDITY`: quantized-point CE scores consistently rank useful legal ternary support relocations across fixed seed/offset changes and both centered/affine representations. It remains a mechanism result, not evidence of strong-PTQ superiority, final whole-model PPL improvement, QAT-gap closure, or large-model scaling. The score-validity branch should now stop; the next experiment should test fixed-budget high-score/random/low-score whole-model composition consistency.

## 2026-08-31 — P7-R large-model held-out robustness

P7-R re-evaluated the already frozen affine top-6 CEGSP rule on larger
Wikitext-2 and bounded streamed C4 slices for Llama-2-7B and Qwen3-8B. Both
runs completed on an A100 with finite metrics, real dataset sources, zero
illegal ternary states, zero codebook residual, unchanged support cardinality,
384 legal relocations, and 768 changed coordinates.

Relative to the affine ternary baseline, CEGSP changed untouched NLL by
Llama W2/C4 `-0.070668/-0.165023` and Qwen W2/C4 `-0.067203/-0.096392`.
Matched random controls changed them by Llama `+0.000650/-0.003523` and
Qwen `+0.000465/-0.003602`. Thus the pre-registered strong cross-domain
scaling gate passes for both model families: CEGSP improves both holdouts and
beats the matched random control on both.

Supported boundary: frozen affine CEGSP transfers a non-random, cross-domain
NLL improvement from the Wikitext fit split to these Llama/Qwen large-model
holdouts. This does not establish SOTA ternary PTQ, compatibility with the
unhealthy PT² checkpoint, QAT-gap closure, or broad downstream accuracy. The
external result-to-claim reviewer was unavailable after the required fallback
attempt, so the formal claim verdict remains `REVIEW_UNAVAILABLE`; the numbers
above are mechanically verified local evidence, not an external acquittal.

## 2026-08-31 — P9-S0 official PT² Llama-2-7B health audit

P9-S0 ran the official PT² repository (`9e943e68`) on Llama-2-7B using the
author-domain `ATQ+SSR` protocol: Wikitext-2 calibration, `nsamples=128`,
`blocksize=128`, `calib_seqlen=2048`, `ppl_seqlen=2048`, `percdamp=0.01`,
`num_p=1`, and Hessian saliency. CEGSP was not called. A symlink named
`/root/Llama-2-7b-hf` was created only so the official path-name based Llama
branch would trigger for the uploaded model at `/CEGSP/model`.

The run completed on an A100 with quantization time `1419.1s`, official
Wikitext-2 PPL `11.6425`, and official C4 PPL `24.3239`. It saved a 13.5GB
fake-quantized Hugging Face checkpoint with three safetensors shards. A shallow
state audit checked 291 saved tensors: all were finite, with max absolute value
`10.953125`.

Interpretation: the previous OPT-350M PT² numerical-health failure should not be
generalized to the official Llama-2-7B setting. This run is healthy enough to
justify a separate, frozen `PT² -> PT²+CEGSP` compatibility test. It does not
itself support any CEGSP-over-PT² performance claim. The external
result-to-claim reviewer remains unavailable, so this is a deterministic local
audit result only.
