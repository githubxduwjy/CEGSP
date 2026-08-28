# AutoResearch Program: PTQ 1.58-bit Continuation

Research direction: post-training ternarization for LLMs, close to PT2-LLM, with
strict negative-result handling.

## 2026-08-24 Autonomous Continuation Contract

The active loop is now stateful and must continue without asking the human after
each run:

```text
monitor current run
-> verify artifact integrity
-> parse all pre-registered metrics
-> update ptq_results.tsv and PTQ_AUTORESEARCH_STATE.json
-> choose exactly one smallest next experiment from the decision tree
-> preregister it in refine-logs
-> launch it on the available GPU
-> repeat
```

R047 established that `hard_l1` improves the first four W2 and C4 windows, while
`hard_l0` and `hard_l0_l1` exhibit distribution-dependent regressions. R048 is
the currently running untouched-holdout test: the first four windows per dataset
choose among `official`, `hard_l0`, `hard_l1`, and `hard_l0_l1`; the next four
windows are hidden until the choice is frozen.

### Machine-checkable R048 decision tree

1. If a non-official candidate is selected and all untouched W2/C4 mean-NLL,
   CVaR10, and nonfinite deltas are non-positive, mark R048 `keep` and run R049
   as a layer-window replication over `(10,11)`, `(20,21)`, and `(30,31)` using
   the same frozen gate.
2. If a non-official candidate is selected but any untouched metric regresses,
   mark R048 `discard`; run one cross-fit replication with swapped/disjoint
   windows before changing the objective. Do not tune epsilon after seeing test.
3. If the gate selects `official`, mark R048 `safe_fallback`; keep the gate but
   close layers `(0,1)` as an improvement claim and screen the three preregistered
   later windows with the same rule.
4. If the run crashes, repair only harness/environment faults. A repeated
   algorithmic failure is logged and the loop moves to the next preregistered
   window rather than hiding the negative result.

The loop must not revive projection-mask enumeration, all-layer hard-T, or
weighted-loss lambda sweeps. Direction changes require a result-supported
mechanism statement and a new preregistration.

### 2026-08-24 R050 decision

R050 used unseen windows 8--23 with an 8/8 gate-test split. `hard_l1` passed the
gate but failed untouched WikiText2 on both mean NLL (+0.032156) and CVaR10
(+0.069191), while C4 improved strongly. Close the `(0,1)` sample-size route:
do not add more samples or tune epsilon for this window.

The next mechanism question is position specificity. Run the three previously
declared adjacent windows sequentially, beginning with `(10,11)`, while keeping
the local hard-T proposal and the exact dual-distribution zero-regression gate.
Each window uses new sequence ranges and is never tuned after its test result.

### 2026-08-24 R051 decision

R051 is the first strict success. On window `(10,11)`, the frozen gate selected
`hard_l11`; untouched W2 mean/CVaR deltas were -0.036275/-0.168679 and C4
mean/CVaR deltas were -0.049213/-0.191124, with no nonfinite logits. The gate
also rejected `hard_l10` and the superficially tempting two-layer update.

Interpret this as evidence for layer-position dependence, not as a final method
claim. Replicate the exact protocol at `(20,21)` and then `(30,31)` before
changing the mechanism or promoting a general claim.

### 2026-08-24 R052 decision

R052 is a preregistered safe fallback. On `(20,21)`, `hard_l20` improved three
of four gate metrics but regressed C4 mean NLL by +0.001831, so the frozen
zero-regression gate selected `official`. All 128 rows were finite and the
untouched result is therefore exactly the official anchor.

This does not negate R051 or the hard-T direction. It rejects automatic transfer
from layer 11 to layers 20/21 and strengthens the position-specific hypothesis.
Do not tune epsilon at `(20,21)`. Complete the final declared `(30,31)` boundary
replication on unseen windows 56--71, then synthesize the depth evidence before
choosing a new mechanism test.

### 2026-08-24 R053 decision and NC-PTQ pivot

R053 selected `official` at `(30,31)`: every hard candidate violated at least
one frozen W2/C4 mean-NLL or CVaR constraint. Integrity was complete (128 rows,
zero nonfinite values). Across the four frozen depth windows, only layer 11 has
strictly generalized; stop layer enumeration.

The next test is not another layer search. R054 uses all four already-declared
windows on unseen sequences 72--87 to audit a no-cancellation hypothesis. The
current two-layer hard candidate is a composition of independent hard-T updates,
not a joint optimizer, so R054 may only claim evidence about cancellation in
composition. A true unconstrained/constrained joint solver belongs to R055 and
is permitted only if the machine-checkable R054 gate supports the mechanism.

### 2026-08-24 R054 decision and checkpoint-gate pivot

R054 had ample mechanism prevalence (47/64 gate, 48/64 test) and no nonfinite
values, but the preregistered cancellation-plus-regression score failed both
splits. Its average Spearman rho was 0.4016/0.3719 on gate/test, below the
boundary-only proxy at 0.4367/0.4316; test CVaR rho was also only 0.1655. Close
the error-cancellation explanation and do not implement the old R055
no-cancellation joint solver.

The simpler boundary checkpoint proxy was directionally stable, especially for
mean-NLL harm (rho 0.5901/0.6388). R057A therefore tests a preregistered five-
configuration OFAT sensitivity matrix on the frozen successful window `(10,11)`:
default, max_steps 2/8, nsamples 16, and blocksize 64. A machine rule may use only
gate windows 88--95 to freeze one non-official candidate/configuration before
reading untouched windows 96--103. If and only if R057A passes, R057B transfers
the frozen hyperparameters to the previously failed `(30,31)` window on fresh
windows 104--119. Epsilon, validation fraction, projection masks, and layer set
remain frozen; seed replication is deferred until both stages avoid regression.

### 2026-08-24 R057A decision and checkpoint-veto replication

R057A completed all five preregistered configurations with 640/640 score rows,
zero nonfinite values, and `selection_uses_test=false`. The machine decision was
`INCONCLUSIVE_OVERCONSERVATIVE`: no non-official candidate passed both the exact
checkpoint-zero-regression gate and the functional gate, so R057B is not run.

The result is nevertheless diagnostic. Default H0 `hard_l11` improved all four
functional gate metrics and all four untouched-test metrics; test deltas were
W2 mean/CVaR -0.025126/-0.119563 and C4 -0.086018/-0.275001. It was vetoed only
because W2 layer-11 checkpoint NMSE increased by +0.000106458. The 2/8-step,
nsamples16, and blocksize64 variants did not pass the functional gate, so do not
broaden the hyperparameter sweep.

R058 fixes H0 and `hard_l11`, changes only calibration seed to 1, and scores new
windows 120--135. It asks whether functional gate/test success can again coexist
with a positive checkpoint delta. This is a gate-validity replication, not an
epsilon sweep: R057A remains inconclusive, and the observed +0.000106458 must not
be copied into a tolerance. R058 does not authorize R057B; a new gate would need
its own preregistration and fresh-data validation.

### Human stop instruction after R058

R058 is the hard stop for the current autonomous experiment loop. After its
machine decision, collect and record the result, then perform a one-time
result-to-claim and research-direction review over R014--R058. Do not launch
R059, R057B, or any other experiment. The review must identify the strongest
supported claim, closed branches, the preferred pivot, an alternative pivot,
and explicit stop/go criteria. Save it to
`refine-logs/PTQ158_DIRECTION_REVIEW_AFTER_R058.md`, then pause both project
state and the heartbeat automation.

### 2026-08-24 R058 decision

R058 completed with exit code 0 and passed every integrity check: the frozen
configuration matched, all 128 expected score records were present, sequence
IDs were 120--135 with the declared 120--127 gate / 128--135 untouched-test
split, and all values were finite with zero nonfinite observations.

The fixed `hard_l11` candidate improved all four untouched-test function
metrics, but failed both Wikitext2 gate metrics (mean-token NLL `+0.0180829`,
CVaR10 NLL increase `+0.0568551`). The machine decision is therefore
`REJECT_CANDIDATE`. Layer-11 checkpoint NMSE was also positive on Wikitext2
and C4 (`+0.000440568` and `+0.000243718`), but checkpoint veto cannot be called
the cause because the functional gate had already failed. This does not
replicate R057A's clean overconservative-veto pattern and closes the current
strict hard-T gate loop pending the mandated direction review. No subsequent
experiment is authorized in this loop.

## 2026-08-22 Primary Pivot

The active hypothesis is no longer activation-Hadamard layer/projection search.
R042-R044 directly tested validation-gated hard updates of the ternary assignment
`T` and established a sharper boundary:

- R042c passed the local mechanism gate: untouched-test block NMSE improved
  5.83% median with a 96.43% block win rate.
- R043a (layers 0/10/20/31) gave W2 `NaN` and C4 56.0225.
- R044a (layer 0 only) gave W2/C4 26.2403/57.9333 versus official no-SSR
  25.8104/66.7370: strong C4 improvement but W2 regression.
- R045 showed the same hard-T layer-0 candidate is better than fixed-T under
  FP16-rest held-out train scoring; no predeclared trajectory metric rejected it.
- R046 put the comparison back into full quantized context and reproduced the
  split: W2 windows worsen while C4 windows improve.

Conclusion: block activation NMSE is not a sufficient acceptance objective for
hard ternary structure changes. Stop the current block-gated method, all-layer
integration, SSR integration, and layer/projection enumeration.

The new primary hypothesis is robust trajectory-gated ternarization:
Hessian-aware hard `T` updates generate sparse proposals, but discrete structure
is accepted only by a disjoint layer/sequence-level gate that is robust across
calibration distributions. Single-distribution layer output, mean token NLL, or
tail-token NLL is not enough.

The next run should test a minimax or Pareto acceptance rule over W2-like and
C4-like held-out streams. No all-layer hard-T search is justified until that
cross-distribution gate rejects candidates that improve one stream while harming
another.

## Context

The current project already ran a local-Haar extension of PT2-style ternary PTQ.
That line failed its mechanism gate:

- Weight-cosine similarity pairing reduced Haar high-frequency weight energy.
- The reduction did not produce enough activation-weighted quantization benefit.
- Band-specific low/high ternary grids improved only 0.35% median vs random.

Do not continue the old hypothesis by merely scaling it up. Treat it as falsified
unless a new objective changes what pairing/rotation optimizes.

## Latest Result

R033 tested the activation-weighted pairing pivot on the same 56-block LLaMA-2-7B
mechanism screen as R019.

- `activation_hf`: median activation-weighted NMSE improvement vs random was
  0.7337%, with 67.86% block win rate.
- Gate target was 5% median improvement and 70% win rate.
- Result: gate failed.

Do not continue local Haar-pairing as the mainline idea.

## Archived Hypothesis

Test activation-weighted pairing.

For each candidate input-column pair `(i, j)`, use:

```text
score(i, j) = ||W[:, i] - W[:, j]||_2^2 * E[(X[:, i] - X[:, j])^2]
```

Greedily choose low-score pairs inside each PT2 block, then evaluate the same
band-grid ternary quantizer used in R019.

## Gate

This hypothesis would have continued only if the new method beat random Haar +
band grid by:

- at least 5% median activation-weighted NMSE, and
- at least 70% block win rate.

It failed, so it is now closed.

## New Positive Result

R034/R035 tested structured rotation shaping after the Haar-pairing failures.

- `activation_rms_hadamard`: 9.7247% median activation-weighted NMSE improvement
  vs identity, with 100% block win rate.
- `joint_norm_hadamard`: 9.1821% median improvement, with 100% block win rate.
- `activation_rms_permutation`: approximately 0% improvement, so sorting alone is
  not the explanation.
- `random_perm_hadamard`: 0.7181% median improvement.
- `reverse_activation_rms_hadamard`: 0.0705% median improvement.

This is now the active mainline.

## PPL Transfer Result

R036a applied the method to a reduced PPL screen on LLaMA-2-7B layers 0/10/20/31:

- identity ATQ: Wikitext2/C4 PPL = 6.6303/8.8797
- activation-rms sorted Hadamard ATQ: Wikitext2/C4 PPL = 6.2003/8.3578
- relative improvement = 6.49%/5.88%

R036b applied the same naive ATQ path to all 224 projections:

- identity ATQ collapsed on Wikitext2 (`NaN`) and gave C4 195.6563
- activation-rms sorted Hadamard remained finite: Wikitext2/C4 = 158.9003/191.6737

Interpretation: activation-sorted Hadamard transfers beyond block metrics, but naive
all-layer ATQ is not good enough. The next step must integrate the rotation into
PT2/GPTQ/AGA or a staged layerwise quantization path.

## Official GPTQ Integration Result

R037 inserted activation-rms sorted Hadamard into the official sequential GPTQ path
without SSR on LLaMA-2-7B, nsamples=8, all 224 projections.

- official GPTQ no-SSR: Wikitext2/C4 PPL = 25.8104/66.7370
- activation-rms sorted Hadamard GPTQ: Wikitext2/C4 PPL = 26.4786/53.2413
- relative change: Wikitext2 regressed by 2.59%, C4 improved by 20.22%

Interpretation: this is not yet a clean win, but it is a real all-layer signal in
the stronger GPTQ path. Continue with insertion-point and SSR-compatibility tests.

## SSR Insertion-Point Result

R038 tested activation-rms sorted Hadamard inside official SSR-selected blocks on
LLaMA-2-7B, nsamples=8, all 224 projections.

- official SSR: Wikitext2/C4 PPL = 20.3080/49.4199
- activation-rms sorted Hadamard inside SSR blocks: Wikitext2/C4 PPL = 24.4195/62.4480
- relative regression: 20.25%/26.36%

Interpretation: do not stack this Hadamard transform inside SSR blocks. SSR and
activation-Hadamard appear to be competing coordinate choices at this insertion
point.

## Selective Replacement Result

R039b tested activation-rms sorted Hadamard only on layers 0/10/20/31, all seven
LLaMA projections in those layers, while leaving the rest of the model on official
GPTQ no-SSR.

- selective activation-Hadamard GPTQ: Wikitext2/C4 PPL = 25.0170/58.1993
- vs official GPTQ no-SSR: relative improvement = 3.07%/12.79%
- vs full activation-Hadamard GPTQ: Wikitext2 is rescued, but C4 gain is smaller
- vs official SSR: still worse than 20.3080/49.4199

Interpretation: selective replacement is the active positive direction. The method
should be searched over layers/projections instead of applied everywhere.

## Layer Attribution Result

R040a/R040b decomposed R039b into layer 0 versus layers 10/20/31.

- layer 0 only: Wikitext2/C4 PPL = 24.2900/55.4673
- layers 10/20/31 only: Wikitext2/C4 PPL = 27.3516/68.2298
- vs official GPTQ no-SSR, layer 0 improves 5.89%/16.89%, while layers 10/20/31
  regress 5.97%/2.24%

Interpretation: layer 0 is the active positive contributor. The previous four-layer
mask diluted the gain by adding harmful later layers.

## Layer-0 Projection-Group Result

R041a/R041b split layer 0 into attention-only versus MLP-only groups.

- layer-0 attention only (q/k/v/o): Wikitext2/C4 PPL = 26.0009/56.3635
- layer-0 MLP only (up/gate/down): Wikitext2/C4 PPL = 28.1206/66.1013
- layer-0 all projections from R040a remains best: 24.2900/55.4673

Interpretation: attention-only carries most of the C4 gain but fails Wikitext2.
MLP-only is harmful on Wikitext2 and nearly neutral on C4. The all-projection
layer-0 result likely depends on cross-submodule coupling, not a simple group
winner.

## Archived 2026-08-21 Hypothesis

Activation-ordered structured mixing creates a better ternary coordinate system
than PT2 SSR or local Haar pairing.

The proposed layer-0 projection decomposition is closed. R041 already showed
cross-submodule coupling, and R042-R044 show that local improvements do not
justify further mask enumeration. Do not compare additional projection masks
as an SSR replacement.

Secondary route: PT2/CAT-style ternarization with calibration contexts generated
from the target model's own reasoning traces, especially for math/coding tasks.

## Forbidden Shortcuts

- Do not claim 1.58-bit benefit without metadata accounting.
- Do not claim inference speed without a real factorized kernel or benchmark.
- Do not use high-frequency energy or zero rate alone as success.
- Do not proceed to full-model PPL if the block-level mechanism gate fails.
- Do not overwrite or reinterpret the prior negative result.

## Useful Local Files

- `refine-logs/EXPERIMENT_PLAN.md`
- `refine-logs/EXPERIMENT_TRACKER.md`
- `results/HAAR_M2_FINDINGS_20260821.md`
- `results/remote-runs/haar_m2_band_r019_20260821_191000/ANALYSIS.md`
- `refine-logs/AUTORESEARCH_SUMMARY_PTQ158_20260821.md`
- `remote-tools/activation_pairing_diagnostics.py`
- `remote-tools/run_activation_pairing_diagnostics.sh`
- `results/remote-runs/activation_pairing_r033b_20260821/metrics.json`
- `remote-tools/rotation_shaping_diagnostics.py`
- `remote-tools/run_rotation_shaping_diagnostics.sh`
- `remote-tools/rotation_ppl_screen.py`
- `remote-tools/run_rotation_ppl_screen.sh`
- `remote-tools/rotation_gptq_quantize.py`
- `remote-tools/run_rotation_gptq_quantize.sh`
- `results/remote-runs/rotation_shaping_r035_20260821/metrics.json`
- `results/remote-runs/rotation_ppl_r036a_20260821/`
- `results/remote-runs/rotation_ppl_r036b_all_20260821/`
- `results/remote-runs/rotation_gptq_r037_ns8_20260821/`
- `results/remote-runs/rotation_ssr_r038_ns8_20260821/`
- `results/remote-runs/rotation_selective_r039b_ns8_20260821/`
- `results/remote-runs/rotation_selective_r040a_l0_ns8_20260821/`
- `results/remote-runs/rotation_selective_r040b_l10_20_31_ns8_20260821/`
- `results/remote-runs/rotation_selective_r041a_l0_attn_ns8_20260821/`
- `results/remote-runs/rotation_selective_r041b_l0_mlp_ns8_20260821/`

## Backup Pivots

The active backup pivots are:

1. Decompose layer 0 by individual projection and small projection pairs.
2. Test the best layer-0 projection mask as an SSR replacement in the real PT2/GPTQ path.
3. Staged layer-count sweep before all-layer claims.
4. Learned or Kronecker/Givens rotations if fixed Hadamard fails against PT2.
5. Calibration-data alignment: PT2/CAT-style ternarization calibrated on task traces,
   especially reasoning traces, instead of generic language-modeling snippets.

Each backup pivot must begin with its own preregistered mechanism test.
