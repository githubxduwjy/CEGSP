# P9-D1: Ordinary-Affine vs PT² Residual Discrete Landscape

**Date:** 2026-09-01
**Run:** `cegsp_p9d1_residual_landscape_llama2_7b_a100_20260901_42028`
**Device:** NVIDIA A100-SXM4-80GB
**Result:** complete, exit code 0

## Question and protocol

P9-D1 asked whether the same frozen CEGSP single-relocation scoring rule sees
a different residual landscape under ordinary affine ternarization and the
healthy detached official PT² ATQ+SSR state. It was a diagnostic, not a new
full-patch method run.

Both initializers used Llama-2-7B, all 32 decoder layers, Q/K only, group size
128, BF16, sequence length 128, batch size 1, one Wikitext train fit batch and
one backward pass. Per layer, the script evaluated the fixed ranked positions
`{1,2,4,8,16,32,64,128}` plus eight deterministic random legal moves. Each
move was applied alone and restored before the next move. Validation used a
disjoint train slice; untouched W2 used `PT2-data/wikitext/testdata`. Neither
evaluation split selected candidates. `mu`, `alpha`, T, and PT² SSR were
frozen.

## Integrity audit

| Check | Result |
|---|---:|
| Candidate count | 512 / initializer; 1024 total |
| Q/K modules | 64 / 64 |
| Layers | 32 / 32 |
| Ranked sample | 256 / initializer |
| Random sample | 256 / initializer |
| Exit code | 0 |
| Finite candidate scores and deltas | yes |
| Nonfinite count | 0 |
| Legal moves | yes |
| Per-group cardinality preserved | yes, by one active-to-zero and zero-to-sign exchange |
| Validation/test used for selection | no |
| Peak GPU memory | 14.65 GiB |

The first remote launch exited before producing a log because of a screen
variable-expansion error. It created no result and did not execute the
experiment. The second launch is the recorded run above.

## Raw results

All numbers below are NLL deltas relative to the corresponding initializer;
negative is an improvement. The absolute baseline NLLs must not be compared
as a performance table because the two initializers are intentionally
different.

| Initializer | Split | Sample | N | Spearman `rho(S,-delta)` | Mean delta | Positive density |
|---|---|---:|---:|---:|---:|---:|
| Ordinary affine | validation | all | 512 | -0.01349 | -0.00016880 | 0.47461 |
| Ordinary affine | validation | rank sample | 256 | — | -0.00036470 | 0.46875 |
| Ordinary affine | validation | rank 1 | 32 | — | -0.00051342 | 0.50000 |
| Ordinary affine | validation | random | 256 | — | +0.00002710 | 0.48047 |
| Ordinary affine | untouched W2 | all | 512 | 0.12837 | -0.00043312 | 0.66992 |
| Ordinary affine | untouched W2 | rank sample | 256 | — | -0.00050481 | 0.68359 |
| Ordinary affine | untouched W2 | rank 1 | 32 | — | -0.00034197 | 0.65625 |
| Ordinary affine | untouched W2 | random | 256 | — | -0.00036143 | 0.65625 |
| PT² ATQ+SSR | validation | all | 512 | 0.10355 | -0.00057583 | 0.68555 |
| PT² ATQ+SSR | validation | rank sample | 256 | — | -0.00075260 | 0.72656 |
| PT² ATQ+SSR | validation | rank 1 | 32 | — | -0.00106890 | 0.71875 |
| PT² ATQ+SSR | validation | random | 256 | — | -0.00039905 | 0.64453 |
| PT² ATQ+SSR | untouched W2 | all | 512 | 0.11093 | +0.00056441 | 0.31836 |
| PT² ATQ+SSR | untouched W2 | rank sample | 256 | — | +0.00051314 | 0.32422 |
| PT² ATQ+SSR | untouched W2 | rank 1 | 32 | — | +0.00059369 | 0.43750 |
| PT² ATQ+SSR | untouched W2 | random | 256 | — | +0.00061568 | 0.31250 |

For the ranked W2 sample, the initializer contrast is clear: ordinary affine
has mean delta `-0.00050481` and 68.36% beneficial moves, while PT² has
`+0.00051314` and 32.42% beneficial moves. At the layer level, the ranked
sample mean is negative in 27/32 ordinary-affine layers but only 7/32 PT²
layers. On validation the direction reverses for PT²: ranked moves improve,
but this does not transfer to untouched W2.

## Interpretation against the pre-registered gates

### Supported deterministic finding

The strongest evidence is consistent with **residual depletion under the
strong initializer**: after optimized PT² ATQ+SSR, the current legal support
relocation space contains substantially fewer W2-beneficial single moves and
the ranked sample is mildly harmful on average. The random PT² control is also
harmed, so this is not evidence that the frozen rank rule can reliably improve
PT² on W2; it indicates that the residual is largely exhausted or no longer
aligned with this move space.

### Not supported

The result does not support a claim that PT² uniquely destroys the QGP ranking.
Both W2 Spearman correlations are weak (`0.12837` vs `0.11093`), and ordinary
affine is not a high-correlation reference in this compact protocol. Therefore
the QGP-boundary interpretation is not isolated. It also does not support a
robust PT²+CEGSP performance gain, a new canonical budget, or any comparison
of the two absolute baseline NLLs.

The validation/W2 disagreement is an important caveat: the PT² ranked sample
improves the two-batch validation slice but worsens untouched W2. This is
evidence that validation-local residuals are not sufficient for a strong-PTQ
compatibility claim under the present compact diagnostic.

## P9 routing decision

The conditional P9-D2 composition diagnostic is **not triggered**. Its
precondition was clear, predictive, and untouched-W2-beneficial PT² single
moves while the earlier full patch was mixed. Here PT² W2 `rho` is weak and
the ranked single moves are harmful on average. Running D2 would therefore
turn into a post-hoc search for a smaller budget, which the protocol forbids.

P9 can close at D1 with the qualified conclusion:

> The detached interface is valid, but after optimized PT² initialization the
> current CEGSP single-relocation space contains little transferable W2
> residual; the compact first-order ranking is weak for both initializers, so
> the data do not identify a PT²-specific QGP collapse.

This closes the strong-initializer branch without modifying the frozen CEGSP
rule. Future work may study a different residual parameterization, but it is
outside P9 and must not be presented as a continuation of this diagnostic.

## Claim-review status

The required external result-to-claim reviewer call was rejected/unavailable
in this session. Per the result-to-claim fail-closed rule, this document is a
deterministic result and integrity report, not an external paper-claim verdict.
`CLAIMS_FROM_RESULTS.md` remains `verdict: REVIEW_UNAVAILABLE`; no paper claim
has been upgraded on the basis of this run.

## Artifacts

- Raw result: `results/remote-runs/cegsp_p9d1_residual_landscape_llama2_7b_a100_20260901_42028/p9d1_result.json`
- Raw log: `results/remote-runs/cegsp_p9d1_residual_landscape_llama2_7b_a100_20260901_42028/screen.log`
- Exit code: `results/remote-runs/cegsp_p9d1_residual_landscape_llama2_7b_a100_20260901_42028/exit_code.txt`
- Plan: `refine-logs/EXPERIMENT_PLAN_CEGSP_P9D1_RESIDUAL_LANDSCAPE_20260901.md`
