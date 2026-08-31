# CEGSP P7-R: Large-Model Held-Out Robustness

Date: 2026-08-31

## Decision

Skip P7-C.  This run is the paper-critical held-out robustness check for the
already completed P7-S0/P7-A/P7-B scaling signal.  It does not introduce a new
method, budget, sign rule, group size, or layer-selection rule.

## Question

Does the frozen affine CEGSP patch selected on the fit split remain beneficial
when evaluated on substantially larger untouched WikiText-2 and a bounded,
streamed C4 slice for both available large-model families?

## Frozen protocol

- models: the already validated `/CEGSP/model` Llama-2-7B and available Qwen3-8B;
- representation: `Q = mu + alpha*T`, `T in {-1, 0, +1}`;
- target: all decoder layers, attention Q/K projections;
- group size: 128;
- threshold factor: 0.75;
- selection: one fit-split quantized-point CE backward;
- layer selection: frozen top-6 ranking;
- edits: 64 legal same-group relocations per selected layer;
- controls: one matched random relocation control on the same selected layers;
- no QAT teacher, latent optimizer, multi-step training, held-out selection,
  post-hoc thresholding, or post-hoc budget choice;
- dtype: BF16; sequence length 128; batch size 1; fixed seed 20260831.

The only change from P7-A/P7-B is evaluation size and enabling a bounded
streaming C4 slice.  The fit, validation, and untouched split offsets remain
fixed by the script and are recorded in each result JSON.

## Planned evaluator sizes

- fit batches: 4;
- validation batches: 32;
- untouched WikiText-2 batches: 32;
- untouched C4 batches: 16 streamed examples;
- gradient batches: 1;
- only `--layer-budgets 6` is run; top-4 is deliberately not revisited.

## Primary comparisons and gates

For each model, report BF16, affine baseline, frozen CEGSP top-6, and matched
random top-6.  The primary quantity is the NLL change relative to the affine
baseline on untouched data.

- Strong cross-domain pass: CEGSP improves both WikiText-2 and C4 and beats the
  matched random control on both for both model families.
- Partial pass: CEGSP improves WikiText-2 and at least one model has a positive
  C4 transfer, with no integrity or finiteness issue.
- Negative result: a fixed-rule C4 or WikiText regression is recorded as a
  boundary condition; it does not justify tuning the frozen rule.

If the C4 source is a fallback or unavailable, the run is valid as a Wikitext
held-out enlargement but cannot support a C4 claim.  A result is invalid for
paper evidence if the data source is fallback, any metric is non-finite, the
codebook audit fails, or the selected layer/edit count differs from the
pre-registration.

## Planned commands

Llama uses the repository base environment. Qwen uses the isolated Qwen3
environment only because the pinned Transformers version does not recognize
`model_type=qwen3`.

```bash
CUDA_VISIBLE_DEVICES=0 python remote-tools/cegsp_p7_a100_scaling.py \
  --mode affine --model /CEGSP/model \
  --run-id cegsp_p7r_llama2_7b_heldout_a100_20260831_42067 \
  --seq-len 128 --batch-size 1 --fit-batches 4 --val-batches 32 \
  --untouched-batches 32 --c4-untouched-batches 16 --grad-batches 1 \
  --layer-budgets 6 --edits-per-layer 64 --dtype bf16
```

```bash
. .venv-qwen3/bin/activate
CUDA_VISIBLE_DEVICES=0 python remote-tools/cegsp_p7_a100_scaling.py \
  --mode affine --model /model/bitahub-model/pice35408784b54431987c4d13c457b9cd/Qwen3-8B \
  --run-id cegsp_p7r_qwen3_8b_heldout_a100_20260831_42067 \
  --seq-len 128 --batch-size 1 --fit-batches 4 --val-batches 32 \
  --untouched-batches 32 --c4-untouched-batches 16 --grad-batches 1 \
  --layer-budgets 6 --edits-per-layer 64 --dtype bf16
```

## Stop conditions

Do not launch P8 downstream or any new tuning experiment from a single
negative result.  First audit source, offsets, layer/edit counts, and finite
metrics.  Do not alter the canonical rule based on the held-out results.
