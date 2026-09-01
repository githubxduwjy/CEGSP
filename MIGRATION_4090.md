# CEGSP 4090 migration bundle

This repository contains the code and environment specification needed for
the next diagnostic/evaluation stage on a single RTX 4090. Model weights,
datasets, caches, and large PyTorch artifacts are intentionally not tracked by
Git.

## Runtime

Use the validated CUDA 12.4 environment in
`env/requirements-cegsp-cu124.txt`. The expected stack is Python 3.11,
PyTorch 2.5.1+cu124, Transformers 4.46.3, Datasets 3.0.1, and Accelerate
1.0.1. Install PyTorch from the CUDA index first, then install the requirements
file. Run the smoke test in `env/README.md` before loading Llama-2-7B.

## 4090-ready entry points

The following scripts are already in the repository and do not require an
A100:

- `remote-tools/cegsp_ce_gradient_4090.py` — one-step CE-gradient candidate
  generation at a deployed ternary point.
- `remote-tools/tqgsp_ce_selection_4090.py` — CE-aware layer selection and
  support projection diagnostics.
- `remote-tools/cegsp_p8_downstream.py` — bounded downstream evaluation using
  the frozen affine CEGSP rule.
- `remote-tools/cegsp_p6a_score_validity_4090.py` and
  `remote-tools/cegsp_p6b_replication_4090.py` — score-validity and replication
  experiments.

P9-D1/D2 are diagnostic designs, not yet frozen executable protocols. They
must not be inferred from P9-S2 or launched as a new cloud experiment until
their script, data split, candidate count, and gate are reviewed.

## P9-S2 detached sidecar

P9-S2 exported the real PT2 state needed by detached diagnostics:
`T`, `mu`, `alpha`, validity masks, group metadata, and full SSR
permutations. It also exported the FP Q/K snapshot and the PT2 Q/K checkpoint.
The artifact metadata is recorded locally in:

`results/remote-runs/cegsp_p9s2_detached_pt2_llama2_7b_plugin_a100_20260901_42071_retry1/metadata.json`

The binary files remain on the A100 run host because they are too large for
Git. Their recorded sizes are approximately 2.22 GB (`ternary_state.pt`),
2.15 GB (`fp_qk.pt`), and 2.15 GB (`model/qk_checkpoint.pt`). Copy them to a
4090 machine only when a reviewed diagnostic explicitly needs them. The
source run directory and exact paths are in `metadata.json` and
`p9s2_result.json`.

## Suggested 4090 layout

Keep code, model/data paths, and run outputs separate:

```text
/workspace/CEGSP/                 # git clone
/workspace/models/Llama-2-7b-hf/  # local model, not Git
/workspace/data/                  # calibration/evaluation data
/workspace/cegsp-runs/            # generated outputs
```

For every run, record the model revision, tokenizer, split/offset, sequence
length, seed, GPU name, CUDA/PyTorch versions, and the exact command in the
result JSON. Never use untouched evaluation data to select layers, budgets,
thresholds, or candidates.

