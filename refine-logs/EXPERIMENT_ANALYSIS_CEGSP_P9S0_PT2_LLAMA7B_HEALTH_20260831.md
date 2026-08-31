# P9-S0: Official PT2 Llama-2-7B health audit

## Verdict

`PT2_LLAMA7B_HEALTH_PASS_CANDIDATE`

This run answers only the baseline-health question. It does not run CEGSP and
does not establish PT2+CEGSP compatibility.

## Protocol

- Remote: A100 80GB endpoint, port 42265.
- Model: `/root/Llama-2-7b-hf`, a symlink to `/CEGSP/model`.
- CEGSP code commit: `5f11e29`.
- PT2 code: official `XIANGLONGYAN/PT2-LLM`, commit
  `9e943e68bdb27469929a4fe7e5720926b9d952d7`.
- Quantization: `atq --ssr`, `nsamples=128`, `blocksize=128`,
  `calib_seqlen=2048`, `ppl_seqlen=2048`, `percdamp=0.01`, `num_p=1`,
  `salient_metric=hessian`.
- Data root: `/root/PT2-data`, formatted for PT2 `load_from_disk`.
- CEGSP called: false.

One non-algorithmic launch fix was required: the official PT2 loader detects
Llama models by checking whether the model path contains `llama`. The uploaded
model path `/CEGSP/model` did not, so a symlink named `/root/Llama-2-7b-hf` was
created. The underlying model files were not changed.

## Results

| Metric | Value |
|---|---:|
| Quantization time | 1419.1 s |
| WikiText-2 PPL | 11.6425 |
| C4 PPL | 24.3239 |
| Checkpoint size | 13,476,889,532 bytes |
| Saved safetensors shards | 3 |
| Tensors checked | 291 |
| All checked tensors finite | true |
| Max absolute saved weight | 10.953125 |

The saved checkpoint is:

`/root/PT2-LLM-full/output/Llama-2-7b-hf_wikitext2_atq_groupsize_128_ssr_True_nsamples_128.pt`

Only logs and summary files were pulled locally. The 13.5GB checkpoint remains
on the remote machine.

## Interpretation

This result is materially different from the earlier OPT-350M PT2 audit. The
official Llama-2-7B `ATQ+SSR` run completed, produced finite W2/C4 perplexity,
saved a reloadable Hugging Face checkpoint, and showed no saved-weight numerical
explosion in the shallow safetensors audit.

The result supports moving from "PT2 reproduction is unhealthy" to a narrower
statement: PT2 was unhealthy in our OPT-350M audit, but the official Llama-2-7B
configuration is healthy enough to serve as the next strong-baseline
compatibility target.

It does not support any claim that CEGSP improves PT2. The next experiment must
be separately pre-registered and must freeze this checkpoint and the existing
CEGSP top-6 rule before running `PT2 -> PT2+CEGSP`.

## Evidence files

- Raw summary:
  `results/remote-runs/cegsp_p9s0_pt2_llama2_7b_health_a100_20260831_42265/summary.json`
- Official PT2 log:
  `results/remote-runs/cegsp_p9s0_pt2_llama2_7b_health_a100_20260831_42265/pt2_official.log`
- Environment note:
  `results/remote-runs/cegsp_p9s0_pt2_llama2_7b_health_a100_20260831_42265/env.txt`

## Reviewer status

The external result-to-claim reviewer was unavailable in the current tool
surface. `CLAIMS_FROM_RESULTS.md` remains `verdict: REVIEW_UNAVAILABLE`.
Therefore this report records a deterministic local audit, not an external
claim acquittal.

## Next step

Pre-register P9-S1 only if we keep this remote checkpoint available:

`PT2 Llama-2-7B -> PT2 + frozen affine-index CEGSP -> PT2 + matched random`

No budget, sign rule, layer count, threshold, or calibration data should be
changed based on this baseline result.

