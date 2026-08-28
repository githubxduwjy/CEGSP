# PTQ 1.58-bit direction review brief after R058

## Problem anchor

- Bottom-line problem: design a post-training ternary (1.58-bit) method that preserves the pretrained LLM's function, not merely its weight matrix, without turning PTQ into QAT.
- Must-solve bottleneck: FP16-to-ternary support mismatch; locally attractive ternary assignments often change the quantized autoregressive trajectory and are distribution-sensitive.
- Non-goals: small cosmetic changes to PT²; projection-mask enumeration; post-hoc epsilon tuning; claiming speed or 1.58 bpw without implementation/accounting.
- Constraints: LLaMA-2-7B on a 24 GB RTX 4090; small calibration sets; frozen pretrained weights except quantization parameters/ternary assignments; PTQ-scale cost.
- Success condition: a preregistered method whose candidate selection transfers to untouched Wikitext2 and C4, improves or preserves model-level function relative to official PT²/GPTQ, and generalizes beyond one layer/window/seed.

## Intended claims to judge

1. Activation/Hessian-aware updating of the discrete ternary support `T` is a real local mechanism, but local reconstruction improvement alone is not sufficient for model-level function preservation.
2. The present `hard-T proposal + strict finite-sample gate` algorithm is not stable enough to be the paper's main method.
3. The most defensible next pivot is cross-layer function-preserving optimization with a retained single-layer trust-region term; soft-to-hard discrete optimization is a backup, and calibration-data alignment is a supporting axis rather than the dominant contribution.

## Evidence chronology

### Closed coordinate-transform branches, R014--R041

- R014--R019: local Haar pairing and band grids failed the 5% mechanism gate; best median gain over random was 0.35%.
- R033 activation-weighted pairing also failed (0.7337% median, 67.86% wins).
- R034/R035 activation-sorted Hadamard produced a real block-level signal (9.7247% median NMSE improvement, 100% wins), and R036a transferred on a reduced four-layer PPL screen (6.49% W2, 5.88% C4 relative improvement).
- Full/integrated results were inconsistent: R037 all-layer GPTQ hurt W2 2.59% but improved C4 20.22%; R038 stacking inside SSR hurt both 20.25%/26.36%. Selective layer-0 use in R040a improved both against no-SSR, but R041 projection decomposition showed interaction rather than an isolated winning mask. Further layer/projection enumeration was closed.

### Discrete-support mechanism and transfer, R042--R046

- R042c: validation-gated hard `T` refinement with matched refit improved untouched block activation-weighted NMSE by median 5.8289%, mean 12.1401%, with 96.43% wins; matched ungated was weaker.
- R043a: applying hard `T` to layers 0/10/20/31 caused W2 NaN, although C4 improved 16.06%; all 1,112 refined blocks were numerically finite.
- R044a: layer 0 only still hurt W2 1.67% while improving C4 13.19%.
- R045: every isolated layer-0/FP16-rest metric preferred hard `T`, so isolated reconstruction could not predict R044's harm.
- R046: scoring in the full quantized context reproduced the W2/C4 direction split, establishing that quantized trajectory and distribution matter.

### Contextual/cross-layer gate experiments, R047--R054

- R047 adjacent-layer screen found `hard_l1` improved both domains on the screen, while the composition was not uniformly safe.
- R048/R050 selected `hard_l1` but failed untouched W2 mean NLL (+0.004538 and +0.032156; R050 also CVaR +0.069191). R049 swapped-fold selection changed to official, showing instability.
- R051 at layers (10,11) selected `hard_l11` and improved untouched W2 mean/CVaR by -0.036275/-0.168679 and C4 by -0.049213/-0.191124.
- R052/R053 safely fell back to official at deeper windows rather than showing broad positive transfer.
- R054 falsified the proposed cancellation-risk mechanism: average cancellation correlation was lower, not higher, than the boundary-only comparator. R055/R056 were canceled/blocked.

### Hyperparameters and gate replication, R057A--R058

- R057A varied max steps 2/8, calibration samples 16, and block size 64 around H0. Only H0/default produced functional gate and untouched-test improvement across W2/C4. H1--H4 failed at least one functional criterion. H0 was vetoed by W2 layer-11 checkpoint NMSE +0.00010645762085914612 despite functional improvement. The formal result was inconclusive/overconservative.
- R058 froze H0/hard_l11, changed calibration seed to 1, and used fresh sequences 120--135. Integrity passed (128/128 rows, exact split, all finite, nonfinite=0). `hard_l11` improved all four untouched-test function metrics but failed W2 gate mean/CVaR by +0.018082916736602783/+0.05685514211654663. Thus the preregistered decision is `REJECT_CANDIDATE`; positive checkpoint NMSE cannot be isolated as the cause. This is gate/test sign instability, not a clean replication of checkpoint over-conservatism.

## Deterministic evidence precheck

All five cited anchor claims are verified in `.aris/evidence_precheck_after_r058.json`:

- r042c_local_support: verified
- r051_cross_distribution_success: verified
- r057a_checkpoint_veto: verified
- r058_gate_failure: verified
- r058_test_improvement: verified

## Integrity status

- R058 passed the deterministic artifact audit described above.
- R057A's external integrity reviewer timed out, but its local deterministic audit passed 640/640 rows, finite checks, and no test leakage in selection.
- The overall evidence is one model (LLaMA-2-7B), primarily two language-modeling distributions, small sequence windows, and mostly single-run comparisons; no broad zero-shot or multi-model evidence exists.

## Candidate directions to compare

### A. Continue current hard-T gate

Keep sparse hard flips as proposals and tune/replace acceptance gates. Risk: repeated gate/test instability and an expanding gate-engineering story; easy to become finite-sample threshold tuning rather than a method contribution.

### B. Cross-layer function preservation with retained local trust region

For a short window `l:l+w`, optimize ternary codes jointly against the FP16 window output while retaining local per-layer regularization:

`L = L_window(H_{l+w}^q, H_{l+w}^{fp}) + lambda * sum_j L_local(H_j^q, H_j^{fp}) + beta * R_flip(T,T0)`.

The key distinction from a gate is that cross-layer function loss shapes the discrete/relaxed update rather than merely accepting or rejecting a fully formed local candidate. Keep `w=2` initially and freeze all layers outside the window. Risk: novelty overlap with sliding-layer reconstruction and PTQ cost.

### C. Soft-to-hard discrete optimization

Replace abrupt hard flips with ternary logits/probabilities or a continuous proxy, optimize a function-aware objective, anneal temperature, then project to {-1,0,+1}; retain a proximity/KL term to the initializer and evaluate the hard endpoint. Risk: collapses into lightweight QAT, relaxation-to-projection gap, and calibration overfit.

### D. Calibration-data alignment

Use generic, model-generated, or task/reasoning-trace calibration streams and distributionally robust weighting. Risk: changing from general PTQ to task-adaptive PTQ and confounding method gains with better data.

## Required reviewer outputs

1. Result-to-claim verdict (`yes|partial|no`) for each intended claim and an allowed claim sentence.
2. Identify which branches are closed versus merely unsupported.
3. State whether the project is in a local dead end or the broader direction is invalid.
4. Rank A--D by scientific value, novelty, feasibility, and information gain per GPU-hour.
5. Select one preferred pivot and one backup pivot.
6. Give explicit go/no-go stop conditions.
7. Design no more than three minimal, preregisterable validation blocks; these are plans only and must not be executed.
8. Be adversarial about overlap with GPTQ/OBQ, CAT-Q/sliding reconstruction, PT² AGA, and soft ternary/QAT approaches. Avoid rescuing the current path by post-hoc hyperparameter tuning.

## Primary files to verify

- `/home/x1shan/文档/ChatGPT/PTQ_paper/refine-logs/EXPERIMENT_TRACKER.md`
- `/home/x1shan/文档/ChatGPT/PTQ_paper/autoresearch/program.ptq158.md`
- `/home/x1shan/文档/ChatGPT/PTQ_paper/results/remote-runs/hessian_gated_r042c_ns12_20260822/summary.json`
- `/home/x1shan/文档/ChatGPT/PTQ_paper/results/remote-runs/r051_mid_window_gate_20260824/metrics.json`
- `/home/x1shan/文档/ChatGPT/PTQ_paper/results/remote-runs/r057a_hparam_gate_20260824/summary.json`
- `/home/x1shan/文档/ChatGPT/PTQ_paper/results/remote-runs/r058_checkpoint_veto_seed1_20260824/r058_summary.json`
- `/home/x1shan/文档/ChatGPT/PTQ_paper/refine-logs/EXPERIMENT_ANALYSIS_R058_20260824.md`
