# CEGSP-04B：OPT-125M second-model sanity check

## Motivation

CEGSP-03B/04A show that CE-gradient ternary top-k editing is robust across WikiText offsets, transfers to C4, and is not tied to a single edit budget on `facebook/opt-350m`. CEGSP-04B tests whether the signal survives on a second cached model in the same OPT family.

## Fixed method

No method redesign is allowed in this experiment.

- Method: CE gradient at deployed ternary weights
- Edit families: support relocation, signflip, joint per-layer best
- Quantization: direct ternary PTQ, group size 128, threshold factor 0.7
- QAT checkpoint/logits/latent weights/optimizer steps: forbidden
- C4: report-only untouched transfer, not used for selection

## Model and proportional layer budget

- Model: `facebook/opt-125m`
- Layers: `0..11`（12 layers）
- `max-edits`: `64`
- OPT-350M used `k=4/6` over 24 layers. Proportional budgets for 12 layers are:
  - `k=2`
  - `k=3`

## Runs

| run | WikiText fit offset | WikiText val offset | C4 token offset |
|---|---:|---:|---:|
| `CEGSP-04B-OPT125M-O0` | 0 | 0 | 0 |
| `CEGSP-04B-OPT125M-O1` | 4096 | 4096 | 4096 |
| `CEGSP-04B-OPT125M-O2` | 8192 | 8192 | 8192 |

## Gate

Primary:

- At least one of the pre-registered top-k families (`support`, `signflip`, `joint`) at `k ∈ {2,3}` improves validation, WikiText untouched, and C4 untouched in at least 2/3 offsets.

Stronger pass:

- A single family/k improves all three splits in 3/3 offsets.

Failure interpretation:

- If OPT-125M fails while OPT-350M passes, CEGSP remains a valid OPT-350M diagnostic but cannot yet claim model-family robustness.
- If WikiText passes but C4 fails, the next action is selection regularization or multi-distribution validation, not direction change.
